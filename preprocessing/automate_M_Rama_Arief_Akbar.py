import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    df = df.drop(columns=['student_id', 'gender', 'school_type', 'parent_education', 'travel_time', 'extra_activities', 'overall_score'])

    X = df.drop(columns=["final_score"])
    y = df["final_score"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    X[categorical_features] = encoder.fit_transform(X[categorical_features])

    numeric_features_for_outlier = X.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_features_for_outlier:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, "X.csv"), index=False)
    pd.DataFrame(y, columns=["final_score"]).to_csv(os.path.join(output_path, "y.csv"), index=False)

    return X, y

if __name__ == "__main__ ":
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  INPUT_PATH = os.path.join(BASE_DIR, "Student_Performance_raw.csv")
  OUTPUT_PATH = os.path.join(BASE_DIR, "Student_Performance_Preprocessing")
  preprocess_data(INPUT_PATH, OUTPUT_PATH)