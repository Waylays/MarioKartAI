import numpy as np
from alexnet import alexnet
WIDTH = 60
HEIGHT = 80
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'mario-kart-{}-{}-{}-epochs-test-data-v2.model'.format(LR, 'alexnet_v2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

# douczanie
model.load(MODEL_NAME)

# train_data = np.load('training_data_final.npy')
# train = train_data[:-100]
# test = train_data[-100:]
#
# X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
# Y = [i[1] for i in train]
#
# test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
# test_y = [i[1] for i in test]

hm_data = 15
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('training_data_final_{}.npy'.format(i))
        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:C:/path/to/log