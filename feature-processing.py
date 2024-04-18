from __future__ import print_function

import argparse
import csv
import json
import os
import boto3
import shutil
import sys
import time
from io import StringIO
from ast import literal_eval

import joblib
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, StandardScaler


label_column = "Critical Risk"
bucket = 'sagemaker-us-east-1-827930657850'
s3key_1 = 'sentencetransformer/input/train.csv'
s3key_2 = 'sentencetransformer/input/incident-batch.jsonl.out'

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    return parser.parse_known_args()

if __name__ == "__main__":

    args, _ = _parse_args()
    
    # process similar reports
    similar_reports = []
    with open(os.path.join(args.train, 'incident-batch.jsonl.out')) as f:
        for line in f:
            # converts jsonline array to normal array
            line = "[" + line.replace("[","").replace("]",",") + "]"
            similar_reports = literal_eval(line)
    batch_results = similar_reports[0].split('\"result\"')[1:]   
    similar_incident_features = np.empty([3, 3], dtype="S10")
    i=0
    for item in batch_results:
        top1 = item.split(", \"genre")[0]
        top1_risk = top1.split("critical-risk\": ")[1]
        top2 = item.split(", \"genre")[1]
        top2_risk = top2.split("critical-risk\": ")[1]
        top3 = item.split(", \"genre")[2]
        top3_risk = top3.split("critical-risk\": ")[1]
        print([top1_risk, top2_risk, top3_risk])
        similar_incident_features[i] = [top1_risk, top2_risk, top3_risk]
        i += 1
    
    
    # Load data
    df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    df_test = df.head(3)
    df_test = df_test[['Industry Sector', 'Genre', 'Critical Risk']]
    df_test['top1_risk'] = similar_incident_features[0,:]
    df_test['top2_risk'] = similar_incident_features[1,:]
    df_test['top3_risk'] = similar_incident_features[2,:]

    print("## Processing complete. Saving model...")


    categorical_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ['Industry Sector', 'Genre', 'Critical Risk']),
        ]
    )

    preprocessor.fit(df_test)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")

       
def input_fn(input_data, content_type):
    print("************** input_fn *******************")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    response = s3_client.get_object(Bucket=bucket, Key=s3key_1) #original features
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    df_test = pd.DataFrame()
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        file = pd.read_csv(response.get("Body"), index_col=0)
        df_test = file.head(3)
        df_test = df_test[['Industry Sector', 'Genre', 'Critical Risk']]
    
    response = s3_client.get_object(Bucket=bucket, Key=s3key_2) #similar reports
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        similar_reports = []
        line = response.get("Body").read().decode('utf-8')
        line = "[" + line.replace("[","").replace("]",",") + "]"
        similar_reports = literal_eval(line)
        batch_results = similar_reports[0].split('\"result\"')[1:]

        similar_incident_features = np.empty([3, 3], dtype="S10")
        i=0
        for item in batch_results:
            top1 = item.split(", \"genre")[0]
            top1_risk = top1.split("critical-risk\": ")[1]
            top2 = item.split(", \"genre")[1]
            top2_risk = top2.split("critical-risk\": ")[1]
            top3 = item.split(", \"genre")[2]
            top3_risk = top3.split("critical-risk\": ")[1]
            print([top1_risk, top2_risk, top3_risk])
            similar_incident_features[i] = [top1_risk, top2_risk, top3_risk]
            i += 1
                   
    df_test['top1_risk'] = similar_incident_features[0,:]
    df_test['top2_risk'] = similar_incident_features[1,:]
    df_test['top3_risk'] = similar_incident_features[2,:]
    print(df_test)
    return df_test


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)
    return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor