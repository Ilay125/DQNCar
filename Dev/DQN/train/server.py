from datetime import datetime
import socket
import threading
import queue
import os

import cv2
from DQN import *
from tensorflow import keras
import numpy as np
from time import sleep

print(tf.config.list_physical_devices('GPU'))


def val_converter(v):
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v[-1] == "f":
        return float(v[:-1])
    return int(v)


def read_conf(path):
    c = {}
    with open(path, "r") as conf:
        config_txt = conf.read()
    config_txt = config_txt.split("\n")

    for conf in config_txt:
        if conf.startswith("#") or conf == "":
            continue
        key, val = conf[:conf.find("=")], conf[conf.find("=") + 1:]
        c[key] = val_converter(val)
    return c


config = read_conf("config.txt")

s = socket.socket()
s.bind((config["ip"], config["port"]))
s.listen(1)

data_backlog = {"image": queue.Queue(), "info": queue.Queue()}

input_shape = (config["img_height"], config["img_width"], config["n_channels"])


def preprocess(img, in_shape):
    img_h, img_w, n_channels = in_shape
    img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    img = img.reshape(img_h, img_w, n_channels)
    return img


def listener(c):
    sample = {}
    SHOW_IMAGE = False
    while True:
        try:
            data = c.recv(64).decode()
            data = data.replace("reset", "")
        except UnicodeDecodeError:
            data_backlog["image"].put(sample["image"])
            data_backlog["info"].put(sample["info"])
            continue
        if "image" in data:
            print(f"image data length: Expected: {int(data[data.find(' ') + 1:])}", end=" ")
            expected_length = int(data[data.find(" ") + 1:])
            data = bytearray()
            while len(data) < expected_length:
                packet = c.recv(expected_length - len(data))
                data.extend(packet)
            print(f"Real: {len(data)}", end="\t")
            data = np.asarray(bytearray(data), dtype="uint8")
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = preprocess(img, input_shape)
            data_backlog["image"].put(img)
            if len(sample) < 2:
                sample["image"] = img.copy()

            if SHOW_IMAGE:
                cv2.imshow("received", img)
                cv2.waitKey(1)

            continue
        if "info" in data:
            data = data[data.find(" ") + 1:]
            data_backlog["info"].put(data.split(" "))
            if len(sample) < 2:
                sample["info"] = data.split(" ").copy()
            print(f"info {data.split(' ')}")
            continue

    cv2.destroyAllWindows()


print(f"Connected to {config['ip']}:{config['port']}")
print("Waiting for connection...")

client = None
while client is None:
    client, addr = s.accept()
    print(f"Established connection with {addr[0]}:{addr[1]}")
    client.send(b"Connection established successfully!")
    threading.Thread(target=listener, args=(client,)).start()


def state_generator():
    while True:
        if data_backlog["image"].qsize() < 1 or data_backlog["info"].qsize() < 1:
            sleep(0.1)
            continue
        img = data_backlog["image"].get()
        info = data_backlog["info"].get()
        yield {"image": img, "info": info}


### PATH MANAGER ###
curr_dir = os.path.dirname("__file__")
log_dir = os.path.join(curr_dir, "logs")
saves_dir = os.path.join(curr_dir, "saves")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(saves_dir):
    os.makedirs(saves_dir)


### BACKUP MANAGER ###
if config["path"] != '':
    backup_file = config["path"]
    backup_episode = config["start_episode"]
    policy = keras.models.load_model(backup_file)
else:
    policy = create_model(input_shape, config["n_outputs"], config["model_loss"], config["learning_rate"])  # makes a policy model
    backup_episode = -1
target_model = copy_model(policy)  # copy policy to target

replay_buffer = ReplayBuffer(config["replay_buffer_size"])  # set a replay buffer
epsilon = config["starting_epsilon"]  # set epsilon to greedy-epsilon
step_count = 0  # tracks no. of steps until termination
average_reward_tracker = AverageRewardTracker(100)  # track average
file_logger = FileLogger(log_dir, file_name=f"progress-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log")

tic = time.time()

state_gen = state_generator()

for episode in range(backup_episode + 1, config["max_episodes"]):
    print(f"Starting episode {episode} with epsilon {epsilon}")

    episode_reward = 0  # reset episode reward
    state = next(state_gen)  # first observation
    print(f"first state of episode {episode}")
    first_q_values = get_q_values(policy, state["image"])
    print(f"Q values: {first_q_values}")
    print(f"Max Q: {max(first_q_values)}")
    prev_dist = float(state["info"][2])
    for step in range(1, config["max_steps"] + 1):
        step_count += 1

        q_values = get_q_values(policy, state["image"])
        delta_dist = round(prev_dist - float(state["info"][2]), 2)

        prev_dist = float(state["info"][2])

        action = select_action_epsilon_greedy(q_values, epsilon)  # select action with greedy-epsilon
        print(f"Action chosen: {action}")
        reward, done = make_step(state["info"], delta_dist, config)  # make a step with the chosen action
        client.send(f"{action}".encode())

        new_state = next(state_gen)

        episode_reward += reward

        if step == config["max_steps"]:  # force termination
            print(f"Episode reached the maximum number of steps. {config['max_steps']}")
            done = True

        state_transition = StateTransition(state["image"], action, reward, new_state["image"],
                                           done)  # define new Q(s, a, r, s')
        replay_buffer.add(state_transition)  # add state transition to buffer

        state = new_state  # update current state

        if step_count % config["target_network_replace_frequency_steps"] == 0:  # update target model to policy
            print("Updating target model")
            target_model = copy_model(policy)

        if replay_buffer.length() >= config["training_start"] and step_count % config["train_every_x_steps"] == 0:  # checks if a batch can be made
            batch = replay_buffer.get_batch(batch_size=config["training_batch_size"])  # make batch
            targets = calculate_target_values(policy, target_model, batch,
                                              config["discount_factor"])  # calculate q-values for evaluation
            states = np.array([state_transition.old_state for state_transition in batch])  # make input from states
            train_model(policy, states, targets)  # train

        if done:
            client.send(b"4")
            break

    average_reward_tracker.add(episode_reward)
    average = average_reward_tracker.get_average()

    print(
        f"episode {episode} finished in {step} steps with reward {episode_reward}. "
        f"Average reward over last 100: {average}")

    file_logger.log(episode, step, episode_reward, average)

    toc = time.time()
    # Calculate training time and format as min:sec
    minutes = format((toc - tic) // 60, '.0f')
    sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
    print(f"Total training time (min:sec): {minutes}:{sec}")

    if episode != 0 and episode % config["model_backup_frequency_episodes"] == 0:  # backup model
        backup_file = f"model_{episode}.h5"
        print(f"Backing up model to {backup_file}")
        policy.save(os.path.join(saves_dir, backup_file))

    epsilon *= config["epsilon_decay_factor_per_episode"]  # epsilon decay
    epsilon = max(config["minimum_epsilon"], epsilon)

# Calculate training time and format as min:sec
minutes = format((toc - tic) // 60, '.0f')
sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
print(f"Total training time (min:sec): {minutes}:{sec}")

policy.save(os.path.join(saves_dir, 'final_model.h5'))

'''
if cv2.waitKey(1) & 0xFF == ord('q'):
    done = True

cv2.imshow('stream', img)
cv2.destroyAllWindows()
print("Done!")
s.send(b"disconnect")
s.close()
'''
