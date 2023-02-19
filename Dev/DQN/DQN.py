import tensorflow as tf
import os
import random
import time

from tensorflow import keras
import numpy as np
import scipy
import uuid
import shutil

import pandas as pd

import socket

input_shape = (320, 180, 1)
outputs = 4


class StateTransition:
    def __init__(self, old_state, action, reward, new_state, done, info=None):
        self.old_state = old_state  # s
        self.action = action  # a
        self.reward = reward  # r
        self.new_state = new_state  # s'
        self.done = done
        self.info = info


class ReplayBuffer:
    current_index = 0

    def __init__(self, size=10000):
        self.size = size  # limits size of buffer replay
        self.transitions = []  # collection of the transitions

    def add(self, transition):  # add a transition
        if len(self.transitions) < self.size:  # checks if there's empty space in buffer
            self.transitions.append(transition)
        else:  # if not - clears the oldest value - FIFO
            self.transitions[self.current_index] = transition
            self.__increment_current_index()

    def length(self):
        return len(self.transitions)

    def get_batch(self, batch_size):  # get random batch of samples
        return random.sample(self.transitions, batch_size)

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self.size - 1:
            self.current_index = 0


class AverageRewardTracker:
    current_index = 0

    def __init__(self, num_rewards_for_average=100):
        self.num_rewards_for_average = num_rewards_for_average
        self.last_x_rewards = []

    def add(self, reward):
        if len(self.last_x_rewards) < self.num_rewards_for_average:
            self.last_x_rewards.append(reward)
        else:
            self.last_x_rewards[self.current_index] = reward
            self.__increment_current_index()

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self.num_rewards_for_average:
            self.current_index = 0

    def get_average(self):
        return np.average(self.last_x_rewards)


class FileLogger:
    def __init__(self, n_dir, file_name='progress.log', new=True):
        self.file_name = os.path.join(n_dir, file_name)
        if new:
            self.clean_progress_file()

    def log(self, episode, steps, reward, average_reward):
        f = open(self.file_name, 'a+')
        f.write(f"{episode};{steps};{reward};{average_reward}\n")
        f.close()

    def clean_progress_file(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        f = open(self.file_name, 'a+')
        f.write("episode;steps;reward;average\n")
        f.close()


def create_model(in_shape, n_outputs, loss, learning_rate):
    # INSPIRED BY VGG-16
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=in_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(n_outputs, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model

'''
def create_model(in_shape, n_outputs, loss):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, 3, activation="relu", input_shape=in_shape))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Conv2D(16, 3, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Conv2D(16, 3, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.MaxPool2D(2))

    
    for i in range(1, 4):
        n_filters = 2**(5+i)
        model.add(keras.layers.Conv2D(n_filters, 3, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Conv2D(n_filters, 3, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Conv2D(n_filters, 3, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.MaxPool2D(2))

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation="softmax"))  # output

    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss)

    return model
'''

def get_q_values(model, state):
    inpt = state[np.newaxis, :]  # make pseudo-batch of inputs
    return model.predict(inpt)[0]  # takes the output of the first input (the real one in the pseudo-batch)


def get_multiple_q_values(model, states):
    return model.predict(states)


def select_action_epsilon_greedy(q_values, epsilon):
    random_value = random.uniform(0, 1)
    if random_value < epsilon:
        return random.randint(0, len(q_values) - 1)  # exploration
    else:
        return np.argmax(q_values)  # exploitation


def select_best_action(q_values):
    return np.argmax(q_values)


def calculate_target_values(policy, target, state_transitions, discount_factor):
    states = []  # all s in Q(s, a, r, s')
    new_states = []  # all s' in Q(s, a, r, s')
    for transition in state_transitions:
        states.append(transition.old_state)
        new_states.append(transition.new_state)

    new_states = np.array(new_states)

    q_values_new_state = get_multiple_q_values(policy, new_states)  # predicts the action according to policy
    q_values_new_state_target = get_multiple_q_values(target, new_states)  # predicts the action according to target

    targets = []  # collection of Q-value
    for index, state_transition in enumerate(state_transitions):
        best_action = select_best_action(
            q_values_new_state[index])  # takes best action according to policy (max of ouput 4)
        best_action_next_state_q_value = q_values_new_state_target[index][
            best_action]  # takes best action according to target (max of ouput 4)

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_next_state_q_value  # bellman equation - target's q_value

        target_vector = [0] * outputs
        target_vector[state_transition.action] = target_value  # considers only highest q-value
        targets.append(target_vector)

    return np.array(targets)


def train_model(model, states, targets):
    model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)


def copy_model(model):
    backup_file = 'backup_' + str(uuid.uuid4())
    model.save(backup_file)
    new_model = keras.models.load_model(backup_file)
    shutil.rmtree(backup_file)
    return new_model


def make_step_manual(info, delta_dist, reward_manager):

    try:
        target, wall, dist, targets, ang_dist = info[:-1]
    except ValueError:
        target, wall, dist, targets, ang_dist = 0, 0, 0, 0, 0
        print(info)

    ang_dist = float(ang_dist)
    ang_dist = round(ang_dist, 2)
    reward = delta_dist * reward_manager["dist_mult"]
    reward += ((reward_manager["ang_dist_offset"] - ang_dist) / (180 - reward_manager["ang_dist_offset"])) * reward_manager["ang_dist_mult"]
    if target == "1":
        reward += reward_manager["target"]
    if wall == "1":
        reward += reward_manager["wall"]
    return reward, (targets == "1" or wall == "1")


def make_step(info, delta_dist, reward_manager):
    try:
        target, wall, dist, targets, ang_dist = info
    except ValueError:
        target, wall, dist, targets, ang_dist = 0, 0, 0, 0, 0
        print(info)

    ang_dist = float(ang_dist)
    ang_dist = round(ang_dist, 2)
    reward = delta_dist * reward_manager["dist_mult"]
    reward += ((reward_manager["ang_dist_offset"] - ang_dist) / (180 - reward_manager["ang_dist_offset"])) * reward_manager["ang_dist_mult"]
    if target == "1":
        reward += reward_manager["target"]
    if wall == "1":
        reward += reward_manager["wall"]
    return reward, (targets == "1" or wall == "1")


def make_step_sim(info):
    try:
        target, wall, dist, targets, ang_dist, *args = info
    except ValueError:
        target, wall, dist, targets, ang_dist = 0, 0, 0, 0, 0
        print(info)

    return targets == "1" or wall == "1"


if __name__ == "__main__":
    print(tf.__version__)
    print(keras.__version__)
    print(tf.config.list_physical_devices('GPU'))
