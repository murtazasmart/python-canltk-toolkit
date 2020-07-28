# pylint: disable=unused-variable
import pickle
from algorithms.mlp import MLP
from algorithms.svm import SVM
from algorithms.tensorflowNN import TensorflowNN
from algorithms.naive_bayes import NaiveBayes
# from tensorflowRNN.tensorflowRNNTest import TensorflowNN
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
import sklearn.utils
# tensorflow_rnn()

import os
import re
import json
import time
# batch_size = 32
# test_split=0.2
# epochs_no=500

class Ensembling:
    
  __instance = None
  models = {}

  @staticmethod 
  def getInstance():
    """ Static access method. """
    if Ensembling.__instance == None:
        Ensembling()
    return Ensembling.__instance

  def __init__(self):
    """ Virtually private constructor. """
    if Ensembling.__instance != None:
        raise Exception("This class is a singleton!")
    else:
        Ensembling.__instance = self
    # for root, dirs, files in os.walk("D:\\MSc\\Chat Parser Script\\models\\ensemble"):
    #   for foldername in dirs:
    #     model = tf.keras.models.load_model("D:\\MSc\\Chat Parser Script\\models\\ensemble\\" + foldername, compile=False)
    #     self.models[foldername] = model
    self.svmInstance = SVM.getInstance()
    self.mlpInstance = MLP.getInstance()
    self.tensorflowNNInstance = TensorflowNN.getInstance()
    self.naiveBayesInstance = NaiveBayes.getInstance()

  def create_csv_file(self, file_name, dictionary):
    with open(file_name, mode='w', encoding="utf8", newline='') as csv_file:
      fieldnames = dictionary[0].keys()
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()

  #         for row in dictionary:
      writer.writerows(dictionary)

  def build_model_n_test(self):
    full_d = []
    allTimeStats = []
    for root, dirs, files in os.walk("./chat-data/extracted-features"):
      for filename in files:
        # print(filename[:-4])
        BASE_NAME = filename[:-4]
        if re.search("(normalized-train-set)", BASE_NAME):
          CHAT_NO = BASE_NAME.split("-")[0]
          CHAT_NAME = BASE_NAME.split("-")[1] + '-' + BASE_NAME.split("-")[2]
          FEATURES_BASE_FOLDER = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/'
          CSV_TRAIN_FILE_NAME = CHAT_NO + '-' + CHAT_NAME + '-normalized-train-set.csv'
          CSV_TEST_FILE_NAME = CHAT_NO + '-' + CHAT_NAME + '-normalized-test-set.csv'
          MODEL_FILE_NAME = CHAT_NO + '-model',
          MODEL_BASE_FOLDER = 'D:/MSc/Chat Parser Script/models/'
          print(CHAT_NO)
          print(CHAT_NAME)
          print(CSV_TRAIN_FILE_NAME)
          print( CSV_TEST_FILE_NAME)
          timeStats = {}
          timeStats["model"] = CHAT_NO
          # if not os.path.exists(MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model'):
          #   timeStats = {}
          #   startTimeTFNN = time.time()
          #   self.tensorflowNNInstance.run_tensorflow_nn(
          #       FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
          #       MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model'
          #   )
          #   endTimeTFNN = time.time()
          #   timeStats["tfNNTime"] = startTimeTFNN - endTimeTFNN
          #   jsond = json.dumps(timeStats)
          #   f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "-build-time-tfNN.json", "w")
          #   f.write(jsond)
          #   f.close()
          if not os.path.exists(MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl'):
          # timeStats = {}
            startTimeMLP = time.time()
            self.mlpInstance.run_mlp(
                FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
                MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl'
            )
            endTimeMLP = time.time()
            timeStats["mlpTime"] = endTimeMLP - startTimeMLP
          # jsond = json.dumps(timeStats)
          # f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "-build-time-mlp.json", "w")
          # f.write(jsond)
          # f.close()
          if not os.path.exists(MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl'):
          # timeStats = {}
            startTimeSVM = time.time()
            self.svmInstance.run_svm(
                FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
                MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl'
            )
            endTimeSVM = time.time()
            timeStats["svmTime"] = endTimeSVM - startTimeSVM
          # jsond = json.dumps(timeStats)
          # f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "-build-time-svm.json", "w")
          # f.write(jsond)
          # f.close()
          # if not os.path.exists(MODEL_BASE_FOLDER + 'naivebayes/' + CHAT_NO + '-model.pkl'):
          # # timeStats = {}
          #   startTimeNaiveBayes = time.time()
          #   self.naiveBayesInstance.run_naivebayes(
          #     FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
          #     MODEL_BASE_FOLDER + 'naivebayes/' + CHAT_NO + '-model.pkl')
          #   endTimeNaiveBayes = time.time()
          #   timeStats["naivebayesTime"] = endTimeNaiveBayes - startTimeNaiveBayes
          # jsond = json.dumps(timeStats)
          # f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "-build-time-naivebayes.json", "w")
          # f.write(jsond)
          # f.close()
          # if not os.path.exists(MODEL_BASE_FOLDER + 'ensemble/' + CHAT_NO + '-model'):
          # timeStats = {}
          startTimeEnsemble = time.time()
          self.run_ensemble(FEATURES_BASE_FOLDER, CSV_TRAIN_FILE_NAME, MODEL_BASE_FOLDER, CHAT_NO)
          endTimeEnsemble = time.time()
          timeStats["ensembleTime"] = endTimeEnsemble - startTimeEnsemble
          # jsond = json.dumps(timeStats)
          # f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "-build-time-ensemble.json", "w")
          # f.write(jsond)
          # f.close()

          # tf_ds, tf_tn, tf_tp, tf_fn, tf_fp = self.tensorflowNNInstance.evaluate_tensorflow_nn(
          #     FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
          #     MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model'
          # )
          mlp_pred, mlp_tn, mlp_tp, mlp_fn, mlp_fp = self.mlpInstance.test_mlp(
              FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
              MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl'
          )
          svm_pred, svm_tn, svm_tp, svm_fn, svm_fp = self.svmInstance.test_svm(
              FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
              MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl'
          )
          # naivebayes_pred, naivebayes_tn, naivebayes_tp, naivebayes_fn, naivebayes_fp = self.naiveBayesInstance.test_naivebayes(
          #     FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
          #     MODEL_BASE_FOLDER + 'naivebayes/' + CHAT_NO + '-model.pkl'
          # )
          multi_ds, multi_tn, multi_tp, multi_fn, multi_fp = self.test_ensemble(FEATURES_BASE_FOLDER, CSV_TEST_FILE_NAME, MODEL_BASE_FOLDER, CHAT_NO)
          
          # tf_accuracy= 0
          # tf_recall = 0
          # tf_precision = 0
          # tf_far = 0
          # tf_frr = 0
          mlp_accuracy = 0
          mlp_recall = 0
          mlp_precision = 0
          mlp_far = 0
          mlp_frr = 0
          # naivebayes_accuracy= 0
          # naivebayes_recall = 0
          # naivebayes_precision = 0
          # naivebayes_far = 0
          # naivebayes_frr = 0
          svm_accuracy = 0
          svm_recall = 0
          svm_precision = 0
          svm_far = 0
          svm_frr = 0
          multi_accuracy = 0
          multi_recall = 0
          multi_precision = 0
          multi_far = 0
          multi_frr = 0
          # if not (tf_tp+tf_tn+tf_fn+tf_fp) == 0:
          #   tf_accuracy = ((tf_tp+tf_tn)/(tf_tp+tf_tn+tf_fn+tf_fp))
          # if not (tf_tp+tf_fn) == 0:
          #   tf_recall = ((tf_tp)/(tf_tp+tf_fn))
          # if not (tf_tp+tf_fp) == 0:
          #   tf_precision = ((tf_tp)/(tf_tp+tf_fp))

          if not (mlp_tp+mlp_tn+mlp_fn+mlp_fp) == 0:
            mlp_accuracy = ((mlp_tp+mlp_tn)/(mlp_tp+mlp_tn+mlp_fn+mlp_fp))
          if not (mlp_tp+mlp_fn) == 0:
            mlp_recall = ((mlp_tp)/(mlp_tp+mlp_fn))
          if not (mlp_tp+mlp_fp) == 0:
            mlp_precision = ((mlp_tp)/(mlp_tp+mlp_fp))

          if not (svm_tp+svm_tn+svm_fn+svm_fp) == 0:
            svm_accuracy = ((svm_tp+svm_tn)/(svm_tp+svm_tn+svm_fn+svm_fp))
          if not (svm_tp+svm_fn) == 0:
            svm_recall = ((svm_tp)/(svm_tp+svm_fn))
          if not (svm_tp+svm_fp) == 0:
            svm_precision = ((svm_tp)/(svm_tp+svm_fp))

          # if not (naivebayes_tp+naivebayes_tn+naivebayes_fn+naivebayes_fp) == 0:
          #   naivebayes_accuracy = ((naivebayes_tp+naivebayes_tn)/(naivebayes_tp+naivebayes_tn+naivebayes_fn+naivebayes_fp))
          # if not (naivebayes_tp+naivebayes_fn) == 0:
          #   naivebayes_recall = ((naivebayes_tp)/(naivebayes_tp+naivebayes_fn))
          # if not (naivebayes_tp+naivebayes_fp) == 0:
          #   naivebayes_precision = ((naivebayes_tp)/(naivebayes_tp+naivebayes_fp))

          if not (multi_tp+multi_tn+multi_fn+multi_fp) == 0:
            multi_accuracy = ((multi_tp+multi_tn)/(multi_tp+multi_tn+multi_fn+multi_fp))
          if not (multi_tp+multi_fn) == 0:
            multi_recall = ((multi_tp)/(multi_tp+multi_fn))
          if not (multi_tp+multi_fp) == 0:
            multi_precision = ((multi_tp)/(multi_tp+multi_fp))

          
          train_dataframe = pd.read_csv(FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME)
          count_row = train_dataframe.shape[0]

          d = {
            "no": CHAT_NO,
            "name": CHAT_NAME,
            "no_of_chats": count_row,
          #   "tf_tn": tf_tn,
          #   "tf_tp": tf_tp,
          #   "tf_fn": tf_fn,
          #   "tf_fp": tf_fp,
          #   "tf_accuracy": tf_accuracy,
          #   "tf_recall": tf_recall,
          #   "tf_precision": tf_precision,
          #   "tf_far": (tf_fp/count_row),
          #   "tf_frr": (tf_fn/count_row),
            "mlp_tn": mlp_tn,
            "mlp_tp": mlp_tp,
            "mlp_fn": mlp_fn,
            "mlp_fp": mlp_fp,
            "mlp_accuracy": mlp_accuracy,
            "mlp_recall": mlp_recall,
            "mlp_precision": mlp_precision,
            "mlp_far": (mlp_fp/count_row),
            "mlp_frr": (mlp_fn/count_row),
            "svm_tn": svm_tn,
            "svm_tp": svm_tp,
            "svm_fn": svm_fn,
            "svm_fp": svm_fp,
            "svm_accuracy": svm_accuracy,
            "svm_recall": svm_recall,
            "svm_precision": svm_precision,
            "svm_far": (svm_fp/count_row),
            "svm_frr": (svm_fn/count_row),
            # "naivebayes_tn": naivebayes_tn,
            # "naivebayes_tp": naivebayes_tp,
            # "naivebayes_fn": naivebayes_fn,
            # "naivebayes_fp": naivebayes_fp,
            # "naivebayes_accuracy": naivebayes_accuracy,
            # "naivebayes_recall": naivebayes_recall,
            # "naivebayes_precision": naivebayes_precision,
            # "naivebayes_far": (naivebayes_fp/count_row),
            # "naivebayes_frr": (naivebayes_fn/count_row),
            "multi_tn": multi_tn,
            "multi_tp": multi_tp,
            "multi_fn": multi_fn,
            "multi_fp": multi_fp,
            "multi_accuracy": multi_accuracy,
            "multi_recall": multi_recall,
            "multi_precision": multi_precision,
            "multi_far": (multi_fp/count_row),
            "multi_frr": (multi_fn/count_row),
          }
          full_d.append(d)
          allTimeStats.append(timeStats)
          print(BASE_NAME)
          self.create_csv_file("./statistics/full-stats-v2(200 with mlp-svm only).csv", full_d)
          jsond = json.dumps(allTimeStats)
          f = open("D:/MSc/Chat Parser Script/chat-data/timings/full-build-time-stats(200 with mlp-svm only).json", "w")
          f.write(jsond)
          f.close()
          # break

  def run_ensemble(
    self,
    FEATURES_BASE_FOLDER,
    CSV_TRAIN_FILE_NAME,
    MODEL_BASE_FOLDER,
    CHAT_NO, 
    batch_size = 32,
    test_split=0.2,
    epochs_no=200):
    dataframe = pd.read_csv(FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME)
    dataframe.head()
    dataframe = sklearn.utils.shuffle(dataframe)

    x_dataframe = dataframe.to_numpy().tolist()
    y_dataframe = []
    for i in x_dataframe:
      y_dataframe.append(i.pop())

    svm_model = self.svmInstance.get_svm_model(MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl')
    svm_result = svm_model.predict(x_dataframe)

    mlp_model = self.mlpInstance.get_mlp_model(MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl')
    mlp_result = mlp_model.predict(x_dataframe)

    # naivebayes_model = self.naiveBayesInstance.get_naivebayes_model(MODEL_BASE_FOLDER + 'naivebayes/' + CHAT_NO + '-model.pkl')
    # naivebayes_result = naivebayes_model.predict(x_dataframe)

    # tf_model = self.tensorflowNNInstance.get_tf_model(MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model')
    # tf_data = self.tensorflowNNInstance.df_to_dataset(dataframe, shuffle=False)
    # tf_result = tf_model.predict_classes(tf_data)


    # tf_pred, tf_tn, tf_tp, tf_fn, tf_fp = tensorflow_rnn_test_evaluate(
    #     FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model'
    # )
    # mlp_pred, mlp_tn, mlp_tp, mlp_fn, mlp_fp = test_mlp(
    #     FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl'
    # )
    # svm_pred, svm_tn, svm_tp, svm_fn, svm_fp = test_svm(
    #     FEATURES_BASE_FOLDER + CSV_TRAIN_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl'
    # )
    # tf_result_formatted = []
    # for elem in tf_result:
    #     tf_result_formatted.extend(elem)
    # data = {'tf_pred':tf_result_formatted, 'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}
    # data = {'naivebayes_pred':naivebayes_result, 'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}
    data = {'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}

    # Create DataFrame
    train_dataframe = pd.DataFrame(data)
    train, val = train_test_split(train_dataframe, test_size=test_split)

    train_ds = self.tensorflowNNInstance.df_to_dataset(train, batch_size=1)
    val_ds = self.tensorflowNNInstance.df_to_dataset(val, shuffle=False, batch_size=1)

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

    train_ds = self.tensorflowNNInstance.df_to_dataset(train, batch_size=batch_size)
    val_ds = self.tensorflowNNInstance.df_to_dataset(val, shuffle=False, batch_size=batch_size)
    
    # Print the output.
    print(train_ds)
    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(8, activation='relu'),
      layers.Dense(8, activation='relu'),
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
    model.save(MODEL_BASE_FOLDER + 'ensemble/' + CHAT_NO + '-model') 

  def test_ensemble(self,
    FEATURES_BASE_FOLDER,
    CSV_TEST_FILE_NAME,
    MODEL_BASE_FOLDER,
    CHAT_NO,
    batch_size = 32,
    test_split=0.2):
    dataframe = pd.read_csv(FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME)
    dataframe.head()
    dataframe = sklearn.utils.shuffle(dataframe)

    x_dataframe = dataframe.to_numpy().tolist()
    y_dataframe = []
    for i in x_dataframe:
      y_dataframe.append(i.pop())

    svm_model = self.svmInstance.get_svm_model(MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl')
    svm_result = svm_model.predict(x_dataframe)

    mlp_model = self.mlpInstance.get_mlp_model(MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl')
    mlp_result = mlp_model.predict(x_dataframe)

    # naivebayes_model = self.naiveBayesInstance.get_naivebayes_model(MODEL_BASE_FOLDER + 'naivebayes/' + CHAT_NO + '-model.pkl')
    # naivebayes_result = naivebayes_model.predict(x_dataframe)

    # tf_model = self.tensorflowNNInstance.get_tf_model(MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model')
    # tf_data = self.tensorflowNNInstance.df_to_dataset(dataframe, shuffle=False)
    # tf_result = tf_model.predict_classes(tf_data)


    # tf_pred, tf_tn, tf_tp, tf_fn, tf_fp = tensorflow_rnn_test_evaluate(
    #     FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'tensorflow-RNN/' + CHAT_NO + '-model'
    # )
    # mlp_pred, mlp_tn, mlp_tp, mlp_fn, mlp_fp = test_mlp(
    #     FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'mlp/' + CHAT_NO + '-model.pkl'
    # )
    # svm_pred, svm_tn, svm_tp, svm_fn, svm_fp = test_svm(
    #     FEATURES_BASE_FOLDER + CSV_TEST_FILE_NAME,
    #     MODEL_BASE_FOLDER + 'svm/' + CHAT_NO + '-model.pkl'
    # )
    # tf_result_formatted = []
    # for elem in tf_result:
    #     tf_result_formatted.extend(elem)
    # data = {'tf_pred':tf_result_formatted, 'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}
    # data = {'naivebayes_pred':naivebayes_result, 'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}
    data = {'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': y_dataframe}

    # Create DataFrame
    test_dataframe = pd.DataFrame(data)
    test_ds = self.tensorflowNNInstance.df_to_dataset(test_dataframe, shuffle=False, batch_size=batch_size)

    model = tf.keras.models.load_model(MODEL_BASE_FOLDER + 'ensemble/' + CHAT_NO + '-model')
    
    loss, accuracy, tp, fp, tn, fn  = model.evaluate(test_ds)
    print('Test Loss: {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))
    print('Test TP: {}'.format(tp))
    print('Test FP: {}'.format(fp))
    print('Test TN: {}'.format(tn))
    print('Test FN: {}'.format(fn))

    return test_ds, tn, tp, fn, fp

  def predict_ensemble(self, MODEL_FILE_NAME, test):
    tf.keras.backend.clear_session()
    test_ds = self.tensorflowNNInstance.df_to_dataset(test, shuffle=False, batch_size=1)
    model = 0
    if MODEL_FILE_NAME in self.models:
          model = self.models[MODEL_FILE_NAME]
    else:
          model = tf.keras.models.load_model(MODEL_FILE_NAME, compile=False)
          self.models[MODEL_FILE_NAME] = model
    res = model.predict_classes(test_ds, 1, 0)
    return res

# e = Ensembling.getInstance()
# e.build_model_n_test()
# TODO iterete through each chat and build model, run test and gather data, 
# data isn't returned, need to fix that

# TODO need to build a CSV with the test outputs

# build_model_n_test()