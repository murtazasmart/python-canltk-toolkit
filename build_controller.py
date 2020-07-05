from feature_extraction import FeatureExtraction as fe
from algorithms.svm import SVM
from algorithms.mlp import MLP
from algorithms.tensorflowNN import TensorflowNN
from ensembling import Ensembling

class BuildController:
    
    def __init__(self):
        self.svmInstance = SVM.getInstance()
        self.tensorflowNNInstance = TensorflowNN.getInstance()
        self.mlpInstance = MLP.getInstance()
        self.ensemblingInstance = Ensembling.getInstance()

    def extract(self, NAME):
        """
        this function will return square(x) value
        :param NAME: chat name e.g. chat3-AbbasJaf-HamzaNaj
        :return: void
        """
        fe.feature_extraction(False, NAME)

    def build(self, NAME):
        self.svmInstance.run_svm(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/svm/' + NAME.split("-")[0] + '-model.pkl')
        print("Finished building SVM models")
        self.mlpInstance.run_mlp(
            'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
            'D:/MSc/Chat Parser Script/models/mlp/' + NAME.split("-")[0] + '-model.pkl')
        print("Finished building MLP models")
        # self.tensorflowNNInstance.run_tensorflow_nn(
        #     'D:/MSc/Chat Parser Script/chat-data/extracted-features/' + NAME + '-normalized-train-set.csv',
        #     'D:/MSc/Chat Parser Script/models/tensorflow-RNN/' + NAME.split("-")[0] + '-model'
        # )
        # print("Finished building tensorflowNN models")
        # self.ensemblingInstance.run_ensemble(
        #     'D:/MSc/Chat Parser Script/chat-data/extracted-features/',
        #     NAME + '-normalized-train-set.csv',
        #     'D:/MSc/Chat Parser Script/models/',
        #     NAME.split("-")[0]
        # )
        # print("Finished building ensemble models")
