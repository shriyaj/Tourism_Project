import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate session
login(HF_TOKEN)

# Initialize remote connections and ingest data
hf_api = HfApi()
DATASET_URL = f"hf://datasets/{HF_USERNAME}/tourism-dataset/tourism.csv"
raw_df = pd.read_csv(DATASET_URL)

print("Dataset loaded successfully.")
print(f"Dataset shape: {raw_df.shape}")
print(f"Columns: {raw_df.columns.tolist()}")

# Prune structurally invalid or empty structural columns
unnamed_features = [col for col in raw_df.columns if 'Unnamed' in col or col.strip() == '']
for feature in unnamed_features:
    raw_df.drop(columns=[feature], inplace=True)
    print(f"Dropped column: {feature}")

# Prune unique ID-like columns to prevent model overfit
high_cardinality_features = [col for col in raw_df.columns if raw_df[col].nunique(dropna=False) == len(raw_df)]
for feature in high_cardinality_features:
    raw_df.drop(columns=[feature], inplace=True)
    print(f"Dropped unique column: {feature}")

# Rectify missing data profiles across data types
print("\nImputing missing values\n")

# Median imputation for continuous metrics
numeric_features = raw_df.select_dtypes(include=[np.number]).columns
for feature in numeric_features:
    if raw_df[feature].isnull().sum() > 0:
        calculated_median = raw_df[feature].median()
        raw_df[feature].fillna(calculated_median, inplace=True)
        print(f"Filled missing values in numerical column '{feature}' with median {calculated_median}")

# Mode imputation for category classifications
low_cardinality_features = raw_df.select_dtypes(include=['object']).columns
for feature in low_cardinality_features:
    if raw_df[feature].isnull().sum() > 0:
        calculated_mode = raw_df[feature].mode()[0]
        raw_df[feature].fillna(calculated_mode, inplace=True)
        print(f"Filled missing values in categorical column '{feature}' with mode {calculated_mode}")

# Inspect distinct category bounds for object-type features
low_cardinality_features = []
for feature in raw_df.columns:
    if raw_df[feature].nunique() <= 5 and raw_df[feature].dtype == 'object':
        low_cardinality_features.append(feature)
        print(f"Column '{feature}' has unique values: {raw_df[feature].unique()}")

# Uniformly format value variances in the gender feature
if 'Gender' in raw_df.columns:
    raw_df['Gender'] = raw_df['Gender'].apply(lambda entry: entry.replace(" ", "").title())
print(f"Column 'Gender' has unique values: {raw_df['Gender'].unique()}")

# Map textual categories to discrete target integers
print("\nEncoding categorical variables")
encoder_instance = LabelEncoder()

target_categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
    'MaritalStatus', 'Designation'
]

for feature in target_categorical_features:
    if feature in raw_df.columns:
        raw_df[feature] = encoder_instance.fit_transform(raw_df[feature].astype(str))

# Isolate independent features from the dependent objective target
TARGET_VARIABLE = 'ProdTaken'
feature_matrix = raw_df.drop(columns=[TARGET_VARIABLE])
target_vector = raw_df[TARGET_VARIABLE]

print(f"\nFeatures shape: {feature_matrix.shape}")
print(f"Target shape: {target_vector.shape}")
print(f"Target distribution:\n{target_vector.value_counts()}")

# Partition data using stratifying distribution holds
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42, stratify=target_vector
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Serialize partitions locally
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("\nDatasets saved locally.")

# Export clean partitions to cloud repository storage
output_artifacts = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for artifact_path in output_artifacts:
    hf_api.upload_file(
        path_or_fileobj=artifact_path,
        path_in_repo=artifact_path.split("/")[-1],
        repo_id=f"{HF_USERNAME}/tourism-dataset",
        repo_type="dataset",
    )
    print(f"Uploaded {artifact_path} to Hugging Face")
    os.remove(artifact_path)
    

print("\nData preparation completed successfully!")
