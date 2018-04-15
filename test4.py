import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data_15.npy')
#shuffle(train_data)

df = pd.DataFrame(train_data)
print(df[1].apply(str).value_counts())
print(df.head())
# print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []



for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 0, 1]:
        rights.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])

shuffle(forwards)
shuffle(lefts)
shuffle(rights)

print(len(forwards),len(lefts),len(rights))

forwards = forwards[:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

print(len(forwards),len(lefts),len(rights))

final_data = forwards + lefts + rights
shuffle(final_data)
df2 = pd.DataFrame(final_data)
print(df2[1].apply(str).value_counts())

np.save('training_data_final_15.npy', final_data)