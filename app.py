# pylint: disable=unused-variable
from flask import Flask, Response
from flask import request, jsonify, redirect
import sys
import os
from flask_cors import CORS
import time
import json

from predict_controller import PredictionController
from build_controller import BuildController

# TODO CHECK HOW TO PRELOAD MODELS

app = Flask(__name__)
CORS(app)
predictionController = PredictionController()
buildController = BuildController()
# if __name__ == "__main__":
#     print("yes")

# predict_controller.t()

@app.route('/models')
def get_models():
    BASE_PATH = "D:\\MSc\\Chat Parser Script\\models"
    l = []
    if 'type' in request.args:
        modelType = request.args['type']
        for root, dirs, files in os.walk(BASE_PATH + "\\" + modelType):
            if (root == BASE_PATH + "\\" + modelType):
                if len(dirs) == 0:
                    l = files
                else: 
                    l = dirs
    else:
        for root, dirs, files in os.walk(BASE_PATH):
            if (root == BASE_PATH):
                l = dirs
    return jsonify(l)

@app.route('/feature-extraction')
def feature_extraction():
    if 'firstModel' in request.args and 'secondModel' in request.args:
        firstModel = request.args['firstModel']
        secondModel = request.args['secondModel']
        isBuildModel = request.args['buildModel']
        print("Request feature extraction " + firstModel + " " + secondModel)
        timeStats = {}
        startTimeFirstModel = time.time()
        buildController.extract(firstModel)
        endTimeFirstModel = time.time()
        timeStats["firstModelStartTime"] = endTimeFirstModel - startTimeFirstModel
        startTimeSecondModel = time.time()
        buildController.extract(secondModel)
        endTimeSecondModel = time.time()
        timeStats["secondModelStartTime"] = endTimeSecondModel - startTimeSecondModel
        timeStats["totalTime"] = ((endTimeFirstModel - startTimeFirstModel) + endTimeSecondModel - startTimeSecondModel)
        print("Finished all feature extractions for " + firstModel + " " + secondModel)
        jsond = json.dumps(timeStats)
        f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + firstModel + "feature-extraction-time-stats.json", "w")
        f.write(jsond)
        f.close()
        if isBuildModel == "true":
            buildController.build(firstModel)
            buildController.build(secondModel)
            print("Finished building all models for " + firstModel + " " + secondModel)
        return jsonify({"result": "Started feature extraction & building models"})

@app.route('/predict')
def predict():
    if 'modelType' in request.args and 'chatModel' in request.args and 'chatMessage' in request.args:
        chatMessage = request.args['chatMessage']
        modelType = request.args['modelType']
        chatModel = request.args['chatModel']
        result = predictionController.predict(chatMessage, modelType, chatModel)
        return jsonify({"result": result})
        
    else:
        # ERROR MESSAGE
        return jsonify({"error": "Something went wrong"})
        # print()

@app.route('/model/build')
def build_model():
    if 'firstModel' in request.args and 'secondModel' in request.args:
        firstModel = request.args['firstModel']
        secondModel = request.args['secondModel']
        timeStats = {}
        startTimeFirstModel = time.time()
        buildController.build(firstModel)
        endTimeFirstModel = time.time()
        timeStats["firstModelStartTime"] = endTimeFirstModel - startTimeFirstModel
        startTimeSecondModel = time.time()
        buildController.build(secondModel)
        endTimeSecondModel = time.time()
        timeStats["firstModelStartTime"] = endTimeSecondModel - startTimeSecondModel
        jsond = json.dumps(timeStats)
        f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + firstModel + "build-time-stats.json", "w")
        f.write(jsond)
        f.close()
        print("Finished building all models")
# app.add_url_rule('/models', 'hello_world', hello_world, methods=['GET'])

if __name__ == '__main__':
       app.run()

# for root, dirs, files in os.walk(".\models"):
#     if (root == ".\\models"):
#         for filename in dirs:
#             print(filename)