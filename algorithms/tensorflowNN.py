# pylint: disable=unused-variable
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

class TensorflowNN:
    
  __instance = None
  models = {}

  @staticmethod 
  def getInstance():
    """ Static access method. """
    if TensorflowNN.__instance == None:
        TensorflowNN()
    return TensorflowNN.__instance

  def __init__(self):
    """ Virtually private constructor. """
    if TensorflowNN.__instance != None:
        raise Exception("This class is a singleton!")
    else:
        TensorflowNN.__instance = self
    # for root, dirs, files in os.walk("D:\\MSc\\Chat Parser Script\\models\\tensorflow-RNN"):
    #   for foldername in dirs:
    #     model = tf.keras.models.load_model("D:\\MSc\\Chat Parser Script\\models\\tensorflow-RNN\\" + foldername, compile=False)
    #     self.models[foldername] = model
    # print("TensorflowNN Models loaded")

# A utility method to create a tf.data dataset from a Pandas Dataframe
  def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('result')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # if shuffle:
    #   ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

  def run_tensorflow_nn(
    self,
    CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat1-MustafaAbid-MurtazaAn-normalized-train-set.csv',
    MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/tensorflow-RNN/chat1-model',
    batch_size = 32,
    test_split=0.2,
    epochs_no=500
  ):
    # A small batch sized is used for demonstration purposes  
    train_dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
    train_dataframe.head()
    # test = pd.read_csv("D:\MSc\Chat Parser Script\chat-data\extracted-features\chat2-MurtazaAn-MustafaAbid-normalized-test-set.csv")
    # test.head()

    # train, test = train_test_split(train_dataframe, test_size=test_split)
    train, val = train_test_split(train_dataframe, test_size=test_split)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    # print(len(test), 'test examples')


    # Done to extract feature keys
    train_ds = self.df_to_dataset(train, batch_size=1)
    val_ds = self.df_to_dataset(val, shuffle=False, batch_size=1)
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
    train_ds = self.df_to_dataset(train, batch_size=batch_size)
    val_ds = self.df_to_dataset(val, shuffle=False, batch_size=batch_size)
    # test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
    ]

    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'), # cck ouut
      # layers.Dense(1, activation='relu')
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn')])

    history = model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs_no)
    model.save(MODEL_FILE_NAME) 
    # loss, accuracy, tp, fp, tn, fn,  = model.evaluate(test_ds)
    # print('Test Loss: {}'.format(loss))
    # print('Test Accuracy: {}'.format(accuracy))
    # print('Test TP: {}'.format(tp))
    # print('Test FP: {}'.format(fp))
    # print('Test TN: {}'.format(tn))
    # print('Test FN: {}'.format(fn))

    # pyplot.plot(history.history['loss'])
    # pyplot.plot(history.history['val_loss'])
    # pyplot.title('model train vs validation loss')
    # pyplot.ylabel('loss')
    # pyplot.xlabel('epoch')
    # pyplot.legend(['train', 'validation'], loc='upper right')
    # pyplot.show()
  
  def evaluate_tensorflow_nn(
    self,
    CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat7-Jonty-AliShabbir-normalized-test-set.csv',
    MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/tensorflow-RNN/chat7-model',
    batch_size = 32,
    test_split=0.2,
    epochs_no=500
  ):
    # A small batch sized is used for demonstration purposes  
    # URL = CSV_OUTPUT_FILE_NAME
    test = pd.read_csv(CSV_OUTPUT_FILE_NAME)
    test.head()

    print(len(test), 'test examples')


    # Done to extract feature keys
    # test_ds = df_to_dataset(test, shuffle=False, batch_size=1)

    # feature_keys = []
    # for feature_batch, label_batch in test_ds.take(1):
    #   feature_keys = list(feature_batch.keys())
    #   print('Every feature:', list(feature_batch.keys()))
    #   print('A batch of targets:', label_batch )

    # feature_columns = []

    # numeric cols
    # for header in feature_keys:
    #   feature_columns.append(feature_column.numeric_column(header))

    model = tf.keras.models.load_model(MODEL_FILE_NAME)
    test_ds = self.df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # labels = test_df.pop('result')
    # test_list = test_df.values.tolist()

    loss, accuracy, tp, fp, tn, fn  = model.evaluate(test_ds)
    print('Test Loss: {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))
    print('Test TP: {}'.format(tp))
    print('Test FP: {}'.format(fp))
    print('Test TN: {}'.format(tn))
    print('Test FN: {}'.format(fn))

    return test_ds, tn, tp, fn, fp

    # res = model.predict_classes(test_ds)
    # print(res)

  def predict_tensorflow_nn(self, MODEL_FILE_NAME, test):
      tf.keras.backend.clear_session()
      test_ds = self.df_to_dataset(test, shuffle=False, batch_size=1)
      model = self.load_model(MODEL_FILE_NAME)
      res = model.predict_classes(test_ds, 1, 0)
      return res

  def get_tf_model(self, MODEL_FILE_NAME):
    model = tf.keras.models.load_model(MODEL_FILE_NAME)
    return model

  def load_model(self, MODEL_FILE_NAME): 
    if MODEL_FILE_NAME in self.models:
      return self.models[MODEL_FILE_NAME]
    else:
      model = tf.keras.models.load_model(MODEL_FILE_NAME, compile=False)
      self.models[MODEL_FILE_NAME] = model
      print("Loaded model for tensorflowNN " + MODEL_FILE_NAME)
      return model

# run_tensorflow_nn()