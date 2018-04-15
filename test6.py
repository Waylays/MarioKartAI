import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, N, M
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 60
HEIGHT = 80
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'mario-kart-{}-{}-{}-epochs-test-data-v2.model'.format(LR, 'alexnet_v2', EPOCHS)

t_time = 0.15


# def straight():
#     PressKey(W)
#     ReleaseKey(A)
#     ReleaseKey(D)


def left():
    PressKey(N)
    ReleaseKey(M)
    time.sleep(t_time)
    ReleaseKey(N)


def right():
    PressKey(M)
    ReleaseKey(N)
    time.sleep(t_time)
    ReleaseKey(M)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

region = (130,120,530,530)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = True
    while (True):

        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=region)
            #print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            # print(prediction)

            # turn_thresh = .75
            # fwd_thresh = 0.70
            idx = np.argmax(prediction)

            if idx == 0:
                left()
                print('left')
            elif idx == 2:
                right()
                print('right')
            else:
                print('straight')
                #straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                paused = True
                print('Pausing!')
                ReleaseKey(N)
                ReleaseKey(M)
                time.sleep(1)


main()
