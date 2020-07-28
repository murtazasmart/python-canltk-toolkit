from feature_extraction import FeatureExtraction as fe
from algorithms.svm import SVM
from algorithms.mlp import MLP
from algorithms.naive_bayes import NaiveBayes
from algorithms.tensorflowNN import TensorflowNN
from ensembling import Ensembling

import time
import json

class BuildController:
    
    def __init__(self):
        self.svmInstance = SVM.getInstance()
        self.tensorflowNNInstance = TensorflowNN.getInstance()
        self.mlpInstance = MLP.getInstance()
        self.naiveBayesInstance = NaiveBayes.getInstance()
        self.ensemblingInstance = Ensembling.getInstance()

    def extract(self, NAME):
        """
        this function will return square(x) value
        :param NAME: chat name e.g. chat3-AbbasJaf-HamzaNaj
        :return: void
        """
        fe.feature_extraction(False, NAME)

    def build(self, NAME):
        timeStats = {}
        startTimeSVM = time.time()
        self.svmInstance.run_svm(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/svm/' + NAME.split("-")[0] + '-model.pkl')
        print("Finished building SVM models")
        endTimeSVM = time.time()
        timeStats["svmTime"] = endTimeSVM - startTimeSVM
        startTimeMLP = time.time()
        self.mlpInstance.run_mlp(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/mlp/' + NAME.split("-")[0] + '-model.pkl')
        print("Finished building MLP models")
        endTimeMLP = time.time()
        timeStats["mlpTime"] = endTimeMLP - startTimeMLP
        startTimeNaiveBayes = time.time()
        self.naiveBayesInstance.run_naivebayes(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/naivebayes/' + NAME.split("-")[0] + '-model.pkl')
        print("Finished building Naive Bayes models")
        endTimeNaiveBayes = time.time()
        timeStats["naiveBayesTime"] = endTimeNaiveBayes - startTimeNaiveBayes
        # startTimeTFNN = time.time()
        # self.tensorflowNNInstance.run_tensorflow_nn(
        #     'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
        #     'D:/MSc/Chat Parser Script/models/tensorflow-RNN/' + NAME.split("-")[0] + '-model'
        # )
        # print("Finished building tensorflowNN models")
        # endTimeTFNN = time.time()
        # timeStats["tfNN"] = endTimeTFNN - startTimeTFNN
        startTimeTFEnsemble = time.time()
        self.ensemblingInstance.run_ensemble(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/',
            NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/',
            NAME.split("-")[0]
        )
        print("Finished building ensemble models")
        endTimeTFEnsemble = time.time()
        timeStats["tfEnsemble"] = endTimeTFEnsemble - startTimeTFEnsemble
        jsond = json.dumps(timeStats)
        f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + NAME + "build-time-stats.json", "w")
        f.write(jsond)
        f.close()
