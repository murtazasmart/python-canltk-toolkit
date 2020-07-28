# pylint: disable=unused-variable
import os
import json
import pandas as pd
from sklearn import preprocessing
import numpy as np
import re
import sys

from feature_extraction import FeatureExtraction
from algorithms.svm import SVM
from algorithms.mlp import MLP
from algorithms.tensorflowNN import TensorflowNN
from ensembling import Ensembling

class PredictionController:

    def __init__(self):
        self.svmInstance = SVM.getInstance()
        self.tensorflowNNInstance = TensorflowNN.getInstance()
        self.mlpInstance = MLP.getInstance()
        self.ensemblingInstance = Ensembling.getInstance()

    def t(self):
        print("heckers")

    def predict(self, chat_message, modelType = "ensemble", chatModel = "chat1-model"):
        NAME = ""
        for root, dirs, files in os.walk("./chat-data/processed-chat"):
            for filename in files:
                # print(filename[:-4])
                BASE_NAME = filename[:-4]
                chatModel = chatModel.split(".")[0]
                chatModelRegex = "(" + chatModel.split("-")[0] + "-)" 
                if re.search(chatModelRegex, BASE_NAME):
                    CHAT_NO = BASE_NAME.split("-")[0]
                    CHAT_NAME = BASE_NAME.split("-")[1] + '-' + BASE_NAME.split("-")[2]
                    NAME = CHAT_NO + "-" + CHAT_NAME
        feature_table = []
        static_feature_table = []
        dynamic_feature_table = []

        CONFIG = FeatureExtraction.set_variables(NAME)

        # f = open("D:\MSc\Chat Parser Script\chat-data\extracted-features\chat1-MustafaAbid-MurtazaAn-feature-set.json", encoding="utf8")
        f = open("D:\\MSc\\Chat Parser Script\\chat-data\\extracted-features\\" + NAME + "-feature-set.json", encoding="utf8")

        data_dictionary = json.load(f)
        print(data_dictionary)
        f.close()

        CSV_OUTPUT_FILE_NAME = 'D:\\MSc\\Chat Parser Script\\chat-data\\extracted-features\\' + NAME + '-partial.csv'
        CHAT_MODEL_BASE_PATH = "D:\\MSc\\Chat Parser Script\\models\\"
        # chat_message = "hahah. no. have to go home. Bro is there a format to send invites? No no you invite him. Your more close to him"
        # chat_message = "I explained it to u yesterday that the reason we didn't call because there was no update to give, we itself were looking for places, and when u called we were still not planned but eventually then and their we Decided to go to Ramada. How is it obvious that ull were getting wet? There are 2 possibilities either u got shelter and werent getting wet or ul didn't find any and got wet. So the obvious part gets eliminated when there are 2 possibilities.. Didn't no where.to go. Cause I know if it was my vehicle what ever the situ I wouldve taken ull inside.. 👆🏽. Judgement"
        # chat_message = "Let me know pricing. Also gym is empty these days. Let's play badminton"
        # chat_message = "No bro. Ill join after dinner. Let me know where ull r going."

        feature_table, static_feature_table, dynamic_feature_table = FeatureExtraction.generate_values(CONFIG, chat_message, data_dictionary, feature_table, static_feature_table, dynamic_feature_table, 1)

        dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
        train_dict = dataframe.to_dict('records')

        train_dict.append(feature_table[0])

        normalizedData = FeatureExtraction.normalize_data(train_dict)
        chat_features = train_dict[len(normalizedData) - 1]

        chat_features = normalizedData[len(normalizedData) - 1]

        if modelType == "svm":
            df = pd.DataFrame(chat_features, index=[0])
            
            result = self.svmInstance.predict_svm(CHAT_MODEL_BASE_PATH + "svm\\" + chatModel + ".pkl", df)
        if modelType == "mlp":
            df = pd.DataFrame(chat_features, index=[0])
            result = self.mlpInstance.predict_mlp(CHAT_MODEL_BASE_PATH + "mlp\\" + chatModel + ".pkl", df)
        if modelType == "svm-rbf":
            df = pd.DataFrame(chat_features, index=[0])
            result = self.svmInstance.predict_svm(CHAT_MODEL_BASE_PATH + "svm-rbf\\" + chatModel + ".pkl", df)
        if modelType == "tensorflow-RNN":
            df = pd.DataFrame(chat_features, index=[0])
            result = self.tensorflowNNInstance.predict_tensorflow_nn(CHAT_MODEL_BASE_PATH + "tensorflow-RNN\\" + chatModel, df)[0][0]
            result = result.tolist()
        if modelType == "ensemble":
            df = pd.DataFrame(chat_features, index=[0])
            svm_result = self.svmInstance.predict_svm(CHAT_MODEL_BASE_PATH + "svm\\" + chatModel + ".pkl", df)
            mlp_result = self.mlpInstance.predict_mlp(CHAT_MODEL_BASE_PATH + "mlp\\" + chatModel + ".pkl", df)
            # tf_result = self.tensorflowNNInstance.predict_tensorflow_nn(CHAT_MODEL_BASE_PATH + "tensorflow-RNN\\" + chatModel, df)
            # tf_result_formatted = []
            # for elem in tf_result:
            #     tf_result_formatted.extend(elem)
            # data = {'tf_pred':tf_result_formatted[0], 'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': 0}
            data = {'mlp_pred':mlp_result, 'svm_pred': svm_result, 'result': 0}
            test_dataframe = pd.DataFrame(data, index=[0])
            result = self.ensemblingInstance.predict_ensemble(CHAT_MODEL_BASE_PATH + "ensemble\\" + chatModel, test_dataframe)[0][0]
            result = result.tolist()
        print(result)
        return result

# predict(NAME="chat3-AbbasJafferjee-HamzaNajmudeen", modelType="ensemble", chatModel="chat3-model")