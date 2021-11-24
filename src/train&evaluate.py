# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from urllib.parse import urlparse
import argparse
import joblib
import json
import yaml

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def eval_metrics(actual, pred):
    # rmse = np.sqrt(mean_squared_error(actual, pred))
    # mae = mean_absolute_error(actual, pred)
    # r2 = r2_score(actual, pred)
    f1_score = metrics.f1_score(actual, pred, average='macro')
    # return rmse, mae, r2, f1_score
    return f1_score

def train_and_evaluate(config_path):
    config = read_params(config_path)
    # test_data_path = config["split_data"]["test_path"]
    # train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["MultinomialNB"]["params"]["alpha"]
    # l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    # target = [config["base"]["target_col"]]

    # train = pd.read_csv(train_data_path, sep=",")
    # test = pd.read_csv(test_data_path, sep=",")

    # train_y = train[target]
    # test_y = test[target]

    # train_x = train.drop(target, axis=1)
    # test_x = test.drop(target, axis=1)

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    print("data loaded")

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups_train.data)

    vectors_test = vectorizer.transform(newsgroups_test.data)

    print("vectorization done")

    clf = MultinomialNB(alpha=alpha)
    clf.fit(vectors, newsgroups_train.target)

    predicted = clf.predict(vectors_test)

    f1_score = eval_metrics(newsgroups_test.target, predicted)
    # (rmse, mae, r2, f1_score) = eval_metrics(vectors_test, predicted)

    print("MultinomialNB model (alpha=%f):" % (alpha))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)
    print("  F1: %s" % f1_score)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            # "rmse": rmse,
            # "mae": mae,
            # "r2": r2,
            "f1-score": f1_score
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
        }
        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    vector_path = os.path.join(model_dir, "vectorizer.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vector_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="../params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)