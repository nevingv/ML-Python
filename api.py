import json,os,sys,re
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)



log_file = None

class CustomType:
    def __init__(self, title, text):
        self.disease = title
        self.probability = text

    def toJSON(self):
        '''
        Serialize the object custom object
        '''
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

@app.route("/api/diseaseclassifier", methods=['POST'])
def post_logfile():
   if request.method == 'POST':
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
        result = []
        for item in classification:
            obj = CustomType(item[0],item[1])
            result.append(json.loads(obj.toJSON()))
        # return json.dumps(OrderedDict(classification))
        return json.dumps(result,indent=4)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
