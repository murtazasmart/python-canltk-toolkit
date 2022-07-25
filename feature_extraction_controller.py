from feature_extraction import FeatureExtraction as fe
from algorithms.svm import SVM
from algorithms.mlp import MLP
from algorithms.naive_bayes import NaiveBayes
from algorithms.tensorflowNN import TensorflowNN
from ensembling import Ensembling

import time
import json

class FeatureExtractionController:

    def extract(self, NAME):
        """
        this function will return square(x) value
        :param NAME: chat name e.g. chat3-AbbasJaf-HamzaNaj
        :return: void
        """
        fe.feature_extraction(False, NAME)
