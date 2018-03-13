import json,os,sys,re
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from flask import Flask
from flask import request, jsonify
app = Flask(__name__)

log_file = None

@app.route("/api/diseaseclassifier", methods=['GET'])
def post_logfile():
   if request.method == 'GET':
        log_file = request.args['symptom']
        print (log_file)
        diseaseclassifier = Trainer(tokenizer) #STARTS CLASIFIERS
        with open("Dataset.csv", "r") as file: #OPENS DATASET
            for i in file: #FOR EACH LINE
                lines = file.next().split(",") #PARSE CSV <DISEASE> <SYMPTOM>
                diseaseclassifier.train(lines[1],  lines[0]) #TRAINING
        diseaseclassifier = Classifier(diseaseclassifier.data, tokenizer)
        classification = diseaseclassifier.classify(log_file) #CLASIFY INPUT
        print classification
        
        return json.dumps(dict(classification))
    

if __name__ == "__main__":
    app.run(port=5000)
    
