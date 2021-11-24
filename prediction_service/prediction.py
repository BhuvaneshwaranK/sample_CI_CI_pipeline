import yaml
import os
import json
import joblib
import numpy as np


params_path = "params.yaml"

target_names = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)

    vect_dir_path = config["webapp_vect_dir"]
    vectorizer = joblib.load(vect_dir_path)

    data_vect = vectorizer.transform(data)
    prediction = model.predict(data_vect).tolist()[0]

    prob = model.predict_proba(data_vect).tolist()[0]
    
    print("prob", max(prob))
    print(prediction)
    print(target_names[prediction])

    return (target_names[prediction], max(prob))


def form_response(dict_request):
    data = dict_request.values()
    # data = [list(map(float, data))]
    response, confidence_score = predict(data)
    response = {"response": response, "confidence_score": confidence_score}
    return response

def api_response(dict_request):
    try:
        print("coming in api response", dict_request)
        data = dict_request.values()
        response, confidence_score = predict(data)
        response = {"response": response, "confidence_score": confidence_score}
        return response
    except Exception as e:
        response = {"response": str(e) }
        return response



# new_data = ['I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.']

# predict(new_data)