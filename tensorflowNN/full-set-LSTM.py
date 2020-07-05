
import fasttext

import tensorflow as tf

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow import feature_column

import pandas as pd

# Imports the chat and trains it
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('result')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

INPUT_FILE_NAME = 'D:\MSc\Chat Parser Script\chat-data\extracted-features\chat3-AbbasJafferjee-HamzaNajmudeen-full-normalized.csv'

dataframe = pd.read_csv(INPUT_FILE_NAME)
dataframe.head()

# STEP 2 - This sets up the test and train data
data = []

# STEP 4 - Setup and Train the LSTM
BUFFER_SIZE = 100
BATCH_SIZE = 64
TAKE_SIZE = 50

# train, test = train_test_split(train_dataframe, test_size=test_split)
train, val = train_test_split(dataframe, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
# print(len(test), 'test examples')


# Done to extract feature keys
train_ds = df_to_dataset(train, batch_size=1)
val_ds = df_to_dataset(val, shuffle=False, batch_size=1)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=1)

feature_keys = []
for feature_batch, label_batch in train_ds.take(1):
  feature_keys = list(feature_batch.keys())
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of targets:', label_batch )

feature_columns = []

# numeric cols
for header in feature_keys:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# batch_size = 32
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(500, 64))
# model.add(tf.keras.layers.DenseFeatures(feature_columns))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# One or more dense layers.
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

model.fit(train_ds, epochs=100, validation_data=val_ds)

# loss, accuracy, tp, fp, tn, fn,   = model.evaluate(test_data)
# print('Test Loss: {}'.format(loss))
# print('Test Accuracy: {}'.format(accuracy))
# print('Test TP: {}'.format(tp))
# print('Test FP: {}'.format(fp))
# print('Test TN: {}'.format(tn))
# print('Test FN: {}'.format(fn))

# print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
