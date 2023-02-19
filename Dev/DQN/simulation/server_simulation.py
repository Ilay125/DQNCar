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


config = read_conf("config_sim.txt")
curr_dir = os.path.dirname("__file__")

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
            img_processed = preprocess(img, input_shape)
            data_backlog["image"].put(img_processed)

            if config["show_preprocess"] == 1:
                cv2.imshow("received", img)
                cv2.imshow("preprocessed", img_processed)
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


def cvt_locs(locs):
    new_locs = []
    for loc in locs:
        pos, target = loc
        pos = pos.split(",")
        target = target.split(",")
        new_locs.append(np.array([round(float(pos[0])), round(float(pos[1])),
                                  round(float(target[0])), round(float(target[1]))]))

    return np.array(new_locs)


def fix_scale(locs, arena_w, arena_h, img_w, img_h):
    locs[:, ::2] += arena_w // 2
    locs[:, 1::2] += arena_h // 2

    locs[:, ::2] *= img_w // arena_w
    locs[:, 1::2] *= img_h // arena_h

    return locs


np.set_printoptions(suppress=True)


def draw_image(locs, arena_w, arena_h, img_w, img_h, ep, hit_target):
    img_dir = os.path.join(curr_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img = np.full((img_h, img_w, 3), (230, 170, 10))

    locs = fix_scale(locs, arena_w, arena_h, img_w, img_h)
    target = locs[0, 2], locs[0, 3]
    start = locs[0, 0], locs[0, 1]

    img = cv2.line(img, (target[0] - 15, target[1] - 15), (target[0] + 15, target[1] + 15), (0, 0, 0), 6)
    img = cv2.line(img, (target[0] - 15, target[1] + 15), (target[0] + 15, target[1] - 15), (0, 0, 0), 6)

    for i in range(1, len(locs)):
        img = cv2.arrowedLine(img, (locs[i-1, 0], locs[i-1, 1]), (locs[i, 0], locs[i, 1]), (255, 255, 255), 2)

    img = cv2.rectangle(img, start, (start[0] + 30, start[1] - 50), (0, 0, 255), -1)

    if hit_target:
        print("hit")
        img = cv2.arrowedLine(img, (locs[-1, 0], locs[-1, 1]), (locs[0, 2], locs[0, 3]), (255, 255, 255), 2)

    cv2.imwrite(os.path.join(img_dir, f"{ep}.jpg"), img)


def state_generator():
    while True:
        if data_backlog["image"].qsize() < 1 or data_backlog["info"].qsize() < 1:
            sleep(0.1)
            continue
        img = data_backlog["image"].get()
        info = data_backlog["info"].get()
        yield {"image": img, "info": info}


policy = keras.models.load_model(config["path"])
state_gen = state_generator()

for episode in range(0, config["max_episodes"]):
    locations = []
    state = next(state_gen)  # first observation
    hit = False
    for step in range(1, config["max_steps"] + 1):
        q_value = get_q_values(policy, state["image"])
        action = select_best_action(q_value)
        done = make_step_sim(state["info"])  # make a step with the chosen action
        client.send(f"{action}".encode())

        if step == config["max_steps"]:  # force termination
            done = True

        locations.append(state["info"][-2:])

        state = next(state_gen)  # update current state
        if state["info"][3] == "1":
            hit = True
        if done:
            if config["draw_image"] == 1:
                draw_image(cvt_locs(locations[:-1]), config["arena_w"], config["arena_h"], config["img_w"], config["img_h"], episode, hit)
            client.send(b"4")
            break


