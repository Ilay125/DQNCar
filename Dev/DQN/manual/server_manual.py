import socket
import threading
import pickle
import os
import numpy as np
from datetime import datetime

###
ip = "10.100.102.5"
port = 6000
###

s = socket.socket()
s.bind((ip, port))
s.listen(1)

saved_q_val = {"state": [], "info": [], "new_state": []}

### PATH MANAGER ###
curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(data_dir)
print(f"Connected to {ip}:{port}")
print("Waiting for connection...")


def data_backup(backup_f):
    global saved_q_val
    while True:
        if len(saved_q_val["state"]) >= backup_f:
            now = datetime.now()
            file_path = os.path.join(data_dir, f"{now.strftime('%d-%m-%Y-%H-%M-%S')}.pickle")
            with open(file_path, 'wb') as handle:
                pickle.dump(saved_q_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
            saved_q_val = {"state": [], "info": [], "new_state": []}


def listener(c):
    done_samples = {"state": [], "info": [], "new_state": []}
    sample = {"state": None, "info": None, "new_state": None}
    ditch = False
    while True:
        try:
            data = c.recv(64).decode()
        except UnicodeDecodeError:
            ditch = True
            continue

        if "image" in data:
            try:
                expected_length = int(data[data.find(" ") + 1:])
            except ValueError:
                ditch = True
                continue
            print(f"image data length: Expected: {expected_length}", end=" ")
            img = bytearray()
            while len(img) < expected_length:
                packet = c.recv(expected_length - len(img))
                img.extend(packet)
            print(f"Real: {len(img)}", end="\t")
            img_arr = np.asarray(bytearray(img), dtype="uint8")

            '''
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            '''

            if sample["state"] is not None:
                sample["new_state"] = img_arr

                if not ditch:
                    done_samples["state"].append(sample["state"])
                    done_samples["info"].append(sample["info"])
                    done_samples["new_state"].append(sample["new_state"])

                    if sample["info"][3] == "1" or sample["info"][1] == "1":
                        saved_q_val["state"].extend(done_samples["state"])
                        saved_q_val["info"].extend(done_samples["info"])
                        saved_q_val["new_state"].extend(done_samples["new_state"])

                        done_samples = {"state": [], "info": [], "new_state": []}

                    sample = {"state": None, "info": None, "new_state": None}
                else:
                    print("ditch")
                    ditch = False

            sample["state"] = img_arr

            continue

        if "info" in data:
            data = data[data.find(" ") + 1:]
            sample["info"] = data.split(" ")
            print(sample["info"])


backup_freq = 100
threading.Thread(target=data_backup, args=(backup_freq, )).start()


while True:
    client, addr = s.accept()
    print(f"Established connection with {addr[0]}:{addr[1]}")
    client.send(b"Connection established successfully!")
    threading.Thread(target=listener, args=(client,)).start()


