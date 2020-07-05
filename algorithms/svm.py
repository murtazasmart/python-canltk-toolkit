# pylint: disable=unused-variable
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
import pickle
import os
from sklearn.feature_selection import VarianceThreshold

class SVM:

    __instance = None
    models = {}

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if SVM.__instance == None:
            SVM()
        return SVM.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if SVM.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SVM.__instance = self
        for root, dirs, files in os.walk("D:\\MSc\\Chat Parser Script\\models\\svm"):
            for filename in files:
                pickle_model = 0
                with open("D:\\MSc\\Chat Parser Script\\models\\svm\\" + filename, 'rb') as file:
                    pickle_model = pickle.load(file)
                self.models[filename] = pickle_model
        print("SVM Models loaded")

    def run_svm(self,
        CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat1-MustafaAbid-MurtazaAn-normalized-train-set.csv',
        MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/svm/chat1-model.pkl',
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

        # x_test = test.to_numpy().tolist()
        # y_test = []
        # for i in x_test:
        #     y_test.append(i.pop())


        clf = svm.SVC()
        # clf = svm.SVC(kernel='rbf')
        # clf = LinearSVC()
        # clf = NuSVC()
        # feature_selection = VarianceThreshold()
        # x = feature_selection.fit_transform(x_train)

        clf.fit(x_train, y_train)

        # test_result = clf.predict(x_test)
        # tp = 0
        # tn = 0
        # fp = 0
        # fn = 0
        # for i, val in enumerate(test_result):
        #     if y_test[i] == val and val == 0:
        #         tn += 1
        #     if y_test[i] == val and val == 1:
        #         tp += 1
        #     if y_test[i] != val and val == 0:
        #         fn += 1
        #     if y_test[i] != val and val == 1:
        #         fp += 1
        # print()
        # print("True negative: " + str(tn))
        # print("True positive: " + str(tp))
        # print("False negative: " + str(fn))
        # print("False positive: " + str(fp))
        # print()
        # if not (tp+tn+fn+fp) == 0:
        #     print("Accuracy: " + str((tp+tn)/(tp+tn+fn+fp)))
        # if not (tp+fn) == 0:
        #     print("Recall: " + str((tp)/(tp+fn)))
        # if not (tp+fp) == 0:
        #     print("Precision: " + str((tp)/(tp+fp)))
        
        pkl_filename = MODEL_FILE_NAME
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

    def test_svm(self,
        CSV_OUTPUT_FILE_NAME = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/chat3-AbbasJafferjee-HamzaNajmudeen-normalized-test-set.csv',
        MODEL_FILE_NAME = 'D:/MSc/Chat Parser Script/models/svm/chat3-model.pkl',
    ):

        dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
        dataframe.head()
        dataframe = sklearn.utils.shuffle(dataframe)

        x_dataframe = dataframe.to_numpy().tolist()
        y_dataframe = []
        for i in x_dataframe:
            y_dataframe.append(i.pop())
            
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

    def predict_svm(self, MODEL_FILE_NAME, dataframe):

        # dataframe = pd.read_csv(CSV_OUTPUT_FILE_NAME)
        # dataframe.head()
        # dataframe = sklearn.utils.shuffle(dataframe)

        x_dataframe = dataframe.to_numpy().tolist()
        y_dataframe = []
        for i in x_dataframe:
            y_dataframe.append(i.pop())
            
        pickle_model = 0
        with open(MODEL_FILE_NAME, 'rb') as file:
            pickle_model = pickle.load(file)

        test_result = pickle_model.predict(x_dataframe)

        return test_result[0]

    def get_svm_model(self, MODEL_FILE_NAME):
        pickle_model = 0
        with open(MODEL_FILE_NAME, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

# run_svm()

# test_svm()