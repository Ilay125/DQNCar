import math
import numpy as np

def val_converter(v):
    if v == '':
        return ''
    if v[-1] == "s":
        return v[:-1]
    if v[-1] == "f":
        return float(v[:-1])
    return int(v)


def read_conf():
    config = {}
    with open("config.txt", "r") as conf:
        config_txt = conf.read()
    config_txt = config_txt.split("\n")

    for conf in config_txt:
        if conf.startswith("#") or conf == "":
            continue
        key, val = conf[:conf.find("=")], conf[conf.find("=")+1:]
        config[key] = val_converter(val)
    return config


def calc_average_reward(arr, freq):
    sums = np.array([np.sum(arr[i*freq:(i+1)*freq]) for i in range(len(arr) // freq)])
    reminder = len(arr) % freq
    sums = np.append(sums, (np.sum(arr[-reminder:]) / reminder) * freq)
    return sums / freq, reminder


a = np.arange(0, 23)
print(a)
print(calc_average_reward(a, 5))
