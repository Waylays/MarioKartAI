# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

region = (130,120,530,530)
WIDTH = 60
HEIGHT = 80


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    output = [0, 0, 0]

    if 'N' in keys:
        output[0] = 1
    elif 'M' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data_15.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = True
    while (True):
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=region)
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            #cv2.imshow('window', screen)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()
            #    break

            # resize to something a bit more acceptable for a CNNt
            keys = key_check()
            output = keys_to_output(keys)
            print(keys)
            training_data.append([screen, output])

            if len(training_data) % 500 == 0:
                print(len(training_data))
                #np.save(file_name, training_data)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        if 'P' in keys:
            print('Save...')
            np.save(file_name, training_data)
            break


main()