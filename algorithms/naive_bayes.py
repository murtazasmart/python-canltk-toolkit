# pylint: disable=unused-variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
import sklearn.utils
import pickle
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import VarianceThreshold

class NaiveBayes:

    __instance = None
    models = {}
    # feature_selection = {}

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if NaiveBayes.__instance == None:
            NaiveBayes()
        return NaiveBayes.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if NaiveBayes.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            NaiveBayes.__instance = self
        for root, dirs, files in os.walk("D:\\MSc\\Chat Parser Script\\models\\naivebayes"):
            for filename in files:
                pickle_model = 0
                with open("D:\\MSc\\Chat Parser Script\\models\\naivebayes\\" + filename, 'rb') as file:
                    pickle_model = pickle.load(file)
                self.models[filename] = pickle_model
        print("NaiveBayes Models loaded")

    def run_naivebayes(self,
        CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat10-Jonty-AliShabbir-normalized-train-set.csv',
        MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/naivebayes/chat10-model.pkl',
    ):

        dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
        dataframe.head()
        dataframe = sklearn.utils.shuffle(dataframe)

        # train, test = train_test_split(dataframe, test_size=0.2)
        train = dataframe
        print(len(train), 'train examples')

        x_train = train.to_numpy().tolist()
        y_train = []
        for i in x_train:
            y_train.append(i.pop())

        # feature_selection = VarianceThreshold()
        # x_train = feature_selection.fit_transform(x_train)
        # self.feature_selection = feature_selection

        gnb = BernoulliNB(alpha=1e-5, binarize=0.0)
        # gnb = GaussianNB()
        # gnb = MultinomialNB()
        # gnb = ComplementNB(alpha=1e-5, norm=True)

        gnb.fit(x_train, y_train)

        pkl_filename = MODEL_FILE_NAME
        with open(pkl_filename, 'wb') as file:
            pickle.dump(gnb, file)

    def test_naivebayes(self,
        CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat10-Jonty-AliShabbir-normalized-test-set.csv',
        MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/naivebayes/chat10-model.pkl',
    ):

        dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
        dataframe.head()
        dataframe = sklearn.utils.shuffle(dataframe)

        x_dataframe = dataframe.to_numpy().tolist()
        y_dataframe = []
        for i in x_dataframe:
            y_dataframe.append(i.pop())

        # feature_selection = VarianceThreshold()
        # feature_selection.set_params(self.params)
        # x_dataframe = self.feature_selection.transform(x_dataframe)
            
        pickle_model = 0
        with open(MODEL_FILE_NAME, 'rb') as file:
            pickle_model = pickle.load(file)

        test_result = pickle_model.predict(x_dataframe)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, val in enumerate(test_result):
            if y_dataframe[i] == val and val == 0:
                tn += 1
            if y_dataframe[i] == val and val == 1:
                tp += 1
            if y_dataframe[i] != val and val == 0:
                fn += 1
            if y_dataframe[i] != val and val == 1:
                fp += 1
        print()
        print("True negative: " + str(tn))
        print("True positive: " + str(tp))
        print("False negative: " + str(fn))
        print("False positive: " + str(fp))
        print()
        if not (tp+tn+fn+fp) == 0:
            print("Accuracy: " + str((tp+tn)/(tp+tn+fn+fp)))
        if not (tp+fn) == 0:
            print("Recall: " + str((tp)/(tp+fn)))
        if not (tp+fp) == 0:
            print("Precision: " + str((tp)/(tp+fp)))

        return test_result, tn, tp, fn, fp

    def predict_naivebayes(self, MODEL_FILE_NAME, dataframe):

        x_dataframe = dataframe.to_numpy().tolist()
        y_dataframe = []
        for i in x_dataframe:
            y_dataframe.append(i.pop())
            
        pickle_model = 0
        with open(MODEL_FILE_NAME, 'rb') as file:
            pickle_model = pickle.load(file)

        test_result = pickle_model.predict(x_dataframe)

        return test_result[0]

    def get_naivebayes_model(self, MODEL_FILE_NAME):
        pickle_model = 0
        with open(MODEL_FILE_NAME, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

# nb = NaiveBayes()
# nb.run_naivebayess()
# nb.test_naivebayes()

# run_naivebayes()

# test_naivebayes()