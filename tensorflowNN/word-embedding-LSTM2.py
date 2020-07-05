
import fasttext

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils

# CSV_OUTPUT_FILE_NAME = 'D:\MSc\Chat Parser Script\chat-data\extracted-features\chat3-AbbasJafferjee-HamzaNajmudeen-normalized-train-set.csv'
CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat1-MustafaAbid-MurtazaAn-normalized-train-set.csv'

dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
dataframe.head()


# # Train/test and variable split
# split = 0.8 # 80% train, 20% test
# split_idx = int(dataframe.shape[0]*split)

# # ...train
# x_train = dataframe.values[0:split_idx,0:dataframe.shape[1]]
# y_train = dataframe.values[0:split_idx,dataframe.shape[1] - 1]

# # ...test
# x_test = dataframe.values[split_idx:-1,0:dataframe.shape[1]]
# y_test = dataframe.values[split_idx:-1,dataframe.shape[1] - 1]

# ...train
x_train = dataframe.values[0:dataframe.shape[0],0:dataframe.shape[1]]
y_train = dataframe.values[0:dataframe.shape[0],dataframe.shape[1] - 1]

look_back = 20
num_features = x_train.shape[1]
nb_hidden_neurons = 32
nb_samples = x_train.shape[0] - look_back

x_train_reshaped = np.zeros((nb_samples, look_back, num_features))
y_train_reshaped = np.zeros((nb_samples))

for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped[i] = x_train[i:y_position]
    y_train_reshaped[i] = y_train[y_position]



# x_test_reshaped = np.zeros((nb_samples, look_back, num_features))
# y_test_reshaped = np.zeros((nb_samples))

# for i in range(nb_samples):
#     y_position = i + look_back
#     x_test_reshaped[i] = x_test[i:y_position]
#     y_test_reshaped[i] = y_test[y_position]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(nb_hidden_neurons, input_shape=(look_back,num_features)))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn')])

model.fit(x_train_reshaped, y_train_reshaped, batch_size = 32, epochs=500, validation_split=0.2)

# loss, accuracy, tp, fp, tn, fn,   = model.evaluate(test_data)
# print('Test Loss: {}'.format(loss))
# print('Test Accuracy: {}'.format(accuracy))
# print('Test TP: {}'.format(tp))
# print('Test FP: {}'.format(fp))
# print('Test TN: {}'.format(tn))
# print('Test FN: {}'.format(fn))

# print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))

# https://stackoverflow.com/questions/45435049/lstm-understand-timesteps-samples-and-features-and-especially-the-use-in-resha
# https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM
