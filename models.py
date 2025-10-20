import pandas as pd
from data_functions import clean_data, pipeline_build
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

TARGET_COL = "Classification"

df = clean_data()
x = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

def linear_regression_model() :
    pipeline = pipeline_build(train_x, train_y, LogisticRegression())
    pred_y = pipeline.predict(test_x)

    print("LINEAR REGRESSION EVALUATION")
    print(f"F1 Score: {f1_score(test_y, pred_y, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_y, pred_y))
    print("\nClassification Report:")
    print(classification_report(test_y, pred_y))

def naive_bayes_model():
    pipeline = pipeline_build(train_x, train_y, GaussianNB())
    pred_y = pipeline.predict(test_x)

    print("NAIVE BAYES EVALUTION:")
    print(f"F1 Score: {f1_score(test_y, pred_y, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_y, pred_y))
    print("\nClassification Report:")
    print(classification_report(test_y, pred_y))


def check_scaler(pipeline) :
    preprocessor = pipeline.named_steps['preprocessing']

# Transform numeric features only
    num_features = preprocessor.transformers_[0][2]
    scaled_values = preprocessor.named_transformers_['num_preproc'].transform(train_x[num_features])

# Wrap into a DataFrame for readability
    scaled_df = pd.DataFrame(scaled_values, columns=num_features)
    print("\nScaled numeric feature summary:")
    print(scaled_df.head())

def check_data_types(df) :
    print("Data shape:", df.shape)
    print("Missing values\n", df.isna().sum())
    print("Data types:\n", df.dtypes)

if __name__ == "__main__":
    linear_regression_model()
    naive_bayes_model()
    
