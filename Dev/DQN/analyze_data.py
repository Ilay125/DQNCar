import math
import pickle
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

### PATH MANAGER ###
data_path = r"C:\Users\ilayb\Desktop\DQNCar\data"
log_path = r"C:\Users\ilayb\Desktop\simulation_train_build\logs_pre1\progress-11-02-2023-15-40-31.log"
img_path = r"C:\Users\ilayb\Desktop\Untitled.png"


input_shape = (180, 320, 1)


def load_data(data_path):
    datas = os.listdir(data_path)
    idx = 0
    print(datas[idx])
    with open(os.path.join(data_path, datas[idx]), "rb") as handle:
        data = pickle.load(handle)

    return data


def cvt_bytes_to_img(arr):
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocess(img, in_shape):
    img_h, img_w, n_channels = in_shape
    img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
    img = img.reshape(img_h, img_w, n_channels)
    return img


def show_image(path):
    img = cv2.imread(path)
    pre_img = preprocess(img, input_shape)
    cv2.imshow("image", img)
    cv2.imshow("preprocess", pre_img)

    if cv2.waitKey() == ord("q"):
        cv2.destroyAllWindows()


def show_sample(state, info, new_state):
    print(info)

    state, new_state = cvt_bytes_to_img(state), cvt_bytes_to_img(new_state)
    preprocess_state = preprocess(state, input_shape)
    cv2.imshow("state", state)
    cv2.imshow("preprocess", preprocess_state)

    if cv2.waitKey() == ord("q"):
        cv2.destroyAllWindows()


def split_episodes(data):
    episodes = []
    curr_episode = []
    for i, x in enumerate(data["info"]):
        curr_episode.append((data["state"][i], data["info"][i]))
        if x[3] == "1" or x[1] == "1":
            episodes.append(curr_episode)
            curr_episode = []
    return episodes


def load_log(log_path):
    df = pd.read_csv(log_path, sep=";")
    return df


def plot_rewards(df):
    plt.figure(figsize=(20, 10))
    plt.ylim((-110, 110))
    plt.plot(df['reward'])
    plt.title('Reward')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.axhline(0, color="black", linestyle="dashed")
    plt.show()


def plot_average_rewards(df, freq):
    plt.figure(figsize=(20, 10))
    data = calc_average_reward(df, freq)
    plt.plot(data)
    plt.title('Reward')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.axhline(0, color="black", linestyle="dashed")
    plt.ylim((-110, 110))
    plt.xlim((0, len(data)))
    plt.show()


def calc_average_reward(df, freq):
    arr = df['reward'].to_numpy()
    avg = np.zeros(arr.shape[0])

    for i in range(1, avg.shape[0]):
        if i < freq:
            avg[i] = np.sum(arr[:i+1]) / i
            continue
        avg[i] = np.sum(arr[i-freq:i]) / freq

    return avg


def find_max_reward_episode(df, freq):
    rewards = calc_average_reward(df, freq)[::freq]
    idx = np.argmax(rewards)
    return idx * freq, rewards[idx]

'''
data = load_data(data_path)
episode = split_episodes(data)
lengths = list(map(lambda x: len(x), episode))
print(sum(lengths) / len(lengths))
'''


#sample = data["state"][0], data["info"][0], data["new_state"][0]
#show_sample(*sample)


freq = 50
data = load_log(log_path)
#show_image(img_path)
print(find_max_reward_episode(data, freq))
plot_average_rewards(data, freq)


