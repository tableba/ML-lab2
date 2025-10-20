import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
matplotlib.use("Qt5Agg")

def clean_data() :
    df = pd.read_csv("data.csv")

    # take a fraction of the dataset
    df = df.sample(frac=0.05, random_state=42)

    # drop the SystemAccessRate column bc too many missing data
    df = df.drop(columns=["SystemAccessRate"]) 

    # drop columns without a ResponseTime
    df.dropna(subset=["ResponseTime", "SessionIntegrityCheck", "ResourceUtilizationFlag"], inplace=True) 

    df.replace(['NA', -99, -1], np.nan, inplace=True)
    df.dropna(inplace=True)

    # turn bool into number
    df['SessionIntegrityCheck'] = df['SessionIntegrityCheck'].astype(int)
    df['ResourceUtilizationFlag'] = df['ResourceUtilizationFlag'].astype(int)

    # df = pd.get_dummies(df, columns=['AlertCategory', 'NetworkEventType', 'NetworkInteractionType'], drop_first=True)

    return df

def pipeline_build(train_x, train_y, model=None):

    if model is None:
        pass

    num_features = [
        "DataTransferVolume_IN",
        "DataTransferVolume_OUT",
        "TransactionsPerSession",
        "SessionIntegrityCheck",
        "ResourceUtilizationFlag",
        "NetworkAccessFrequency",
        "UserActivityLevel",
        "ResponseTime"
    ]

    cat_features = [
        "AlertCategory",
        "NetworkEventType",
        "NetworkInteractionType",
        "SecurityRiskLevel",
    ]
    
    preprocessing = ColumnTransformer(
        transformers=[
            ("num_preproc", StandardScaler(), num_features),
            ("cat_preproc", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
        ])

    pipeline = Pipeline(steps=[
                        ("preprocessing", preprocessing),
                        ("model", model)])
    
    pipeline.fit(train_x, train_y)

    return pipeline

if __name__ == "__main__" :
    df = clean_data()
    TARGET_COL = "Classification"

    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    train_x, test_x, train_y, test_y  = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline_build(train_x, train_y, LogisticRegression())
