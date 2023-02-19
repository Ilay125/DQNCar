import os
import cv2
from DQN import *
from tensorflow import keras
import numpy as np
import pickle
#import matplotlib.pyplot as plt


### PATH MANAGER ###
data_dir = r"C:\Users\ilayb\Desktop\simulation_train_build\data"


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

input_shape = (config["img_height"], config["img_width"], config["n_channels"])


def preprocess(img, in_shape):
    img_h, img_w, n_channels = in_shape
    img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    img = img.reshape(img_h, img_w, n_channels)
    return img


def cvt_bytes_to_img(arr):
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def state_generator():
    files = os.listdir(data_dir)
    print(files)
    for i_f, f in enumerate(files):
        print(f"{f} {i_f}/{len(files)}")
        with open(os.path.join(data_dir, f), "rb") as handle:
            data = pickle.load(handle)

        for i in range(len(data["state"])):
            state, info, new_state = cvt_bytes_to_img(data["state"][i]), data["info"][i], cvt_bytes_to_img(data["new_state"][i])
            state = preprocess(state, input_shape)
            yield {"image": state,
                   "info": info,
                   "new_image": new_state,
                   "last": (i == (len(data["state"]) - 1) and i_f == (len(files) - 1))
                            }


### PATH MANAGER ###
curr_dir = os.path.dirname("__file__")
saves_dir = os.path.join(curr_dir, "saves_pretrain")

if not os.path.exists(saves_dir):
    os.makedirs(saves_dir)


policy = create_model(input_shape, config["n_outputs"], config["model_loss"], config["learning_rate"])  # makes a policy model
backup_episode = -1

replay_buffer = ReplayBuffer(config["replay_buffer_size"])  # set a replay buffer
epsilon = config["starting_epsilon"]  # set epsilon to greedy-epsilon
step_count = 0  # tracks no. of steps until termination
average_reward_tracker = AverageRewardTracker(100)  # track average

tic = time.time()

state_gen = state_generator()

done_files = False

while not done_files:
    print(f"starting with epsilon: {epsilon}")
    episode_reward = 0  # reset episode reward
    state = next(state_gen)  # first observation

    first_q_values = get_q_values(policy, state["image"])

    print(f"Q values: {first_q_values}")
    print(f"Max Q: {max(first_q_values)}")

    prev_dist = float(state["info"][2])
    for step in range(1, config["max_steps"] + 1):
        step_count += 1

        q_values = get_q_values(policy, state["image"])
        delta_dist = round(prev_dist - float(state["info"][2]), 2)

        prev_dist = float(state["info"][2])

        #action = select_action_epsilon_greedy(q_values, epsilon)  # select action with greedy-epsilon
        action = select_best_action(q_values)
        print(f"Action chosen: {action}")
        reward, done = make_step_manual(state["info"], delta_dist, config)  # make a step with the chosen action

        new_state = next(state_gen)

        if new_state["last"]:
            done_files = True
            break
        episode_reward += reward

        if step == config["max_steps"]:  # force termination
            print(f"Episode reached the maximum number of steps. {config['max_steps']}")
            done = True

        state_transition = StateTransition(state["image"], action, reward, new_state["image"],
                                           done, info=state["info"])  # define new Q(s, a, r, s')
        replay_buffer.add(state_transition)  # add state transition to buffer

        state = new_state  # update current state

        if replay_buffer.length() >= config["training_start"] and step_count % config["train_every_x_steps"] == 0:  # checks if a batch can be made
            batch = replay_buffer.get_batch(batch_size=config["training_batch_size"])  # make batch
            targets = np.array([int(i.info[-1]) for i in batch])
            states = np.array([state_transition.old_state for state_transition in batch])  # make input from states
            train_model(policy, states, targets)  # train

        if done:
            break

    average_reward_tracker.add(episode_reward)
    average = average_reward_tracker.get_average()

    print(
        f"finished in {step} steps with reward {episode_reward}. "
        f"Average reward over last 100: {average}")


    toc = time.time()
    # Calculate training time and format as min:sec
    minutes = format((toc - tic) // 60, '.0f')
    sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
    print(f"Total training time (min:sec): {minutes}:{sec}")

    epsilon *= config["epsilon_decay_factor_per_episode"]  # epsilon decay
    epsilon = max(config["minimum_epsilon"], epsilon)

# Calculate training time and format as min:sec
minutes = format((toc - tic) // 60, '.0f')
sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
print(f"Total training time (min:sec): {minutes}:{sec}")
print("done")
policy.save(r"C:\Users\ilayb\Desktop\DQNCar\saves_pretrain\pretrain.h5")

'''
policy = create_model(input_shape, config["n_outputs"])  # makes a policy model

replay_buffer = ReplayBuffer(config["replay_buffer_size"])  # set a replay buffer
step_count = 0  # tracks no. of steps until termination
average_reward_tracker = AverageRewardTracker(100)  # track average reward

tic = time.time()

state_gen = state_generator()
done_files = False
episode = -1
while not done_files:
    episode += 1
    episode_reward = 0  # reset episode reward
    state = next(state_gen)  # first observation
    print(f"first state of episode {episode}")
    prev_dist = float(state["info"][2])
    for step in range(1, config["max_steps"] + 1):
        step_count += 1

        delta_dist = round(prev_dist - float(state["info"][2]), 2)
        prev_dist = float(state["info"][2])

        action = int(state["info"][-1])  # select action with greedy-epsilon
        reward, done = make_step_manual(state["info"], delta_dist, config)  # make a step with the chosen action

        new_state = next(state_gen)

        if new_state["last"]:
            done_files = True
            break

        episode_reward += reward

        if step == config["max_steps"]:  # force termination
            print(f"Episode reached the maximum number of steps. {config['max_steps']}")
            done = True

        state_transition = StateTransition(state["image"], action, reward, new_state["image"],
                                           done)  # define new Q(s, a, r, s')
        replay_buffer.add(state_transition)  # add state transition to buffer

        state = new_state  # update current state

        if replay_buffer.length() >= config['training_start'] and step_count % config['train_every_x_steps'] == 0:  # checks if a batch can be made
            batch = replay_buffer.get_batch(batch_size=config['training_batch_size'])  # make batch
            targets = np.array([i.action for i in batch])
            states = np.array([state_transition.old_state for state_transition in batch])  # make input from states
            train_model(policy, states, targets)  # train

        if done:
            break

    average_reward_tracker.add(episode_reward)
    average = average_reward_tracker.get_average()

    print(
        f"episode {episode} finished in {step} steps with reward {episode_reward}. "
        f"Average reward over last 100: {average}")

    #file_logger.log(episode, step, episode_reward, average)

    toc = time.time()
    # Calculate training time and format as min:sec
    minutes = format((toc - tic) // 60, '.0f')
    sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
    print(f"Total training time (min:sec): {minutes}:{sec}")

# Calculate training time and format as min:sec
minutes = format((toc - tic) // 60, '.0f')
sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
print(f"Total training time (min:sec): {minutes}:{sec}")

policy.save(os.path.join(saves_dir, 'pretrained_model.h5'))
'''

