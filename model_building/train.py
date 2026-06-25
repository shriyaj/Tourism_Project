import os
import sys
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Import models requested by the rubric
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import multiprocessing
from hf_credentials import HF_USERNAME
os.environ["PYTHONWARNINGS"] = "ignore"  
# Initialize MLflow tracking server configurations
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("tourism-package-prediction")

hf_api = HfApi()

# Set up endpoints for partition data ingestion
TRAIN_FEATURES_URL = f"hf://datasets/{HF_USERNAME}/tourism-dataset/Xtrain.csv"
TEST_FEATURES_URL = f"hf://datasets/{HF_USERNAME}/tourism-dataset/Xtest.csv"
TRAIN_TARGET_URL = f"hf://datasets/{HF_USERNAME}/tourism-dataset/ytrain.csv"
TEST_TARGET_URL = f"hf://datasets/{HF_USERNAME}/tourism-dataset/ytest.csv"

print("CRITICAL: Ingesting segmented partitions from Hugging Face...")
X_train = pd.read_csv(TRAIN_FEATURES_URL)
X_test = pd.read_csv(TEST_FEATURES_URL)
y_train = pd.read_csv(TRAIN_TARGET_URL).values.ravel()
y_test = pd.read_csv(TEST_TARGET_URL).values.ravel()

print(f"STATUS: Feature training matrix shape: {X_train.shape}")
print(f"STATUS: Feature evaluation matrix shape: {X_test.shape}")

# Target structural features for standardization
continuous_features = X_train.columns.tolist()

# Define structural column transformer for Z-score normalization
data_preprocessor = make_column_transformer(
    (StandardScaler(), continuous_features),
    remainder='passthrough'
)

# Define candidates and their hyperparameter spaces
models_to_evaluate = {
    "Random_Forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "grid": {
            "randomforestclassifier__n_estimators": [100,200,300],
            "randomforestclassifier__max_depth": [10,20,None],
            "randomforestclassifier__class_weight": ["balanced",None]
        }
    },
    "Gradient_Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "grid": {
            "gradientboostingclassifier__n_estimators": [100,150,200],
            "gradientboostingclassifier__max_depth": [3,4,5],
            "gradientboostingclassifier__learning_rate": [0.05,0.1]
        }
    },
    "XGBoost": {
        "estimator": xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False),
        "grid": {
            "xgbclassifier__n_estimators": [100,200,300],
            "xgbclassifier__max_depth": [3,5,7],
            "xgbclassifier__learning_rate": [0.05,0.1],
            "xgbclassifier__scale_pos_weight": [1,2,3]
        }
    }
}

# Variables to track the overall championship model across architectures
global_best_score = -1.0
global_best_pipeline = None
global_best_model_name = ""

print("\nEXECUTION: Initiating primary MLflow parent run scope...")
with mlflow.start_run(run_name="Model_Comparison_Suite"):
    
    # Loop over each defined algorithm candidate
    for model_name, model_config in models_to_evaluate.items():
        print("\n" + "="*60)
        print(f"STARTING EXPERIMENT: {model_name}")
        print("="*60)
        
        # Build independent pipeline for the current algorithm
        execution_pipeline = make_pipeline(data_preprocessor, model_config["estimator"])
        
        print(f"EXECUTION: Running cross-validated grid search for {model_name}...")
        optimization_search = GridSearchCV(
            execution_pipeline, 
            model_config["grid"], 
            cv=3, 
            n_jobs=-1, 
            scoring='roc_auc',
            verbose=1
        )
        optimization_search.fit(X_train, y_train)

        # Print and log the optimum hyperparameters for this candidate
        best_params = optimization_search.best_params_
        best_n_estimators = next((v for k, v in best_params.items() if k.endswith("n_estimators")), None)
        best_max_depth = next((v for k, v in best_params.items() if k.endswith("max_depth")), None)
        print(f"Best params for {model_name}: n_estimators={best_n_estimators}, max_depth={best_max_depth}")

        # Log parameters evaluated within this candidate's space to MLflow
        tuning_metrics = optimization_search.cv_results_
        for configuration_index in range(len(tuning_metrics['params'])):
            target_parameters = tuning_metrics['params'][configuration_index]
            validated_score = tuning_metrics['mean_test_score'][configuration_index]

            with mlflow.start_run(run_name=f"{model_name}_param_sweep", nested=True):
                mlflow.log_params(target_parameters)
                mlflow.log_metric(f"{model_name}_mean_roc_auc", validated_score)

        # Isolate candidate's top configuration
        candidate_best_pipeline = optimization_search.best_estimator_
        
        # Generate predictions
        train_predictions = candidate_best_pipeline.predict(X_train)
        test_predictions = candidate_best_pipeline.predict(X_test)
        train_probabilities = candidate_best_pipeline.predict_proba(X_train)[:, 1]
        test_probabilities = candidate_best_pipeline.predict_proba(X_test)[:, 1]

        # Calculate evaluation performance benchmarks
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        train_precision = precision_score(y_train, train_predictions, zero_division=0)
        test_precision = precision_score(y_test, test_predictions, zero_division=0)
        train_recall = recall_score(y_train, train_predictions, zero_division=0)
        test_recall = recall_score(y_test, test_predictions, zero_division=0)
        train_f1 = f1_score(y_train, train_predictions, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, zero_division=0)
        train_roc_auc = roc_auc_score(y_train, train_probabilities)
        test_roc_auc = roc_auc_score(y_test, test_probabilities)

        # Save individual candidate logs inside nested MLflow tags
        with mlflow.start_run(run_name=f"{model_name}_Best_Evaluation", nested=True):
            mlflow.log_params(optimization_search.best_params_)
            mlflow.log_metrics({
                f"{model_name}_test_accuracy": test_accuracy,
                f"{model_name}_test_precision": test_precision,
                f"{model_name}_test_recall": test_recall,
                f"{model_name}_test_f1_score": test_f1,
                f"{model_name}_test_roc_auc": test_roc_auc
            })

        # Print model evaluations
        print("\n" + "#"*60)
        print(f"                {model_name.upper()} METRIC REPORT                ")
        print("#"*60)
        print(f"Metric: Accuracy   | Train: {train_accuracy:.4f} | Evaluation: {test_accuracy:.4f}")
        print(f"Metric: Precision  | Train: {train_precision:.4f} | Evaluation: {test_precision:.4f}")
        print(f"Metric: Recall     | Train: {train_recall:.4f} | Evaluation: {test_recall:.4f}")
        print(f"Metric: F1-Score   | Train: {train_f1:.4f} | Evaluation: {test_f1:.4f}")
        print(f"Metric: ROC-AUC    | Train: {train_roc_auc:.4f} | Evaluation: {test_roc_auc:.4f}")
        print("#"*60)

        # Evaluate if this specific model beats all previous architectures
        if test_roc_auc > global_best_score:
            global_best_score = test_roc_auc
            global_best_pipeline = candidate_best_pipeline
            global_best_model_name = model_name

    # -------------------------------------------------------------
    # Post-Comparison: Register and Save the Overall Champ
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print(f"🏆 CHAMPION MODEL FOUND: {global_best_model_name} (Test ROC-AUC: {global_best_score:.4f})")
    print("="*60)

    # Log overall championship indicators to parent MLflow scope
    mlflow.log_param("winning_algorithm_type", global_best_model_name)
    mlflow.log_metric("championship_test_roc_auc", global_best_score)

    # Generate final definitive classification reports for the winner
    final_test_preds = global_best_pipeline.predict(X_test)
    print("\nFinal Champion Classification Report:")
    print(classification_report(y_test, final_test_preds, target_names=['No Purchase', 'Purchase']))
    print("\nFinal Champion Confusion Matrix:")
    print(confusion_matrix(y_test, final_test_preds))

    # Serialize champion binary state locally
    LOCAL_MODEL_PATH = "best_tourism_model_v1.joblib"
    joblib.dump(global_best_pipeline, LOCAL_MODEL_PATH)
    print(f"\nSTATUS: Serialized champion binary saved locally to: {LOCAL_MODEL_PATH}")

    # Track artifact state inside active MLflow run instance
    mlflow.log_artifact(LOCAL_MODEL_PATH, artifact_path="model")
    print("STATUS: Dispatched localized model file tracking states to MLflow server.")

    # Hub deployment variables
    TARGET_HF_REPO = f"{HF_USERNAME}/tourism-prediction-model"
    HUB_REPO_TYPE = "model"

    # Verify remote tracking target availability
    try:
        hf_api.repo_info(repo_id=TARGET_HF_REPO, repo_type=HUB_REPO_TYPE)
        print(f"\nNOTICE: Target model storage endpoint '{TARGET_HF_REPO}' confirmed active.")
    except RepositoryNotFoundError:
        print(f"\nALERT: Target endpoint '{TARGET_HF_REPO}' unresolved. Provisions initialized...")
        create_repo(repo_id=TARGET_HF_REPO, repo_type=HUB_REPO_TYPE, private=False)
        print(f"STATUS: Created model repository target workspace '{TARGET_HF_REPO}'.")

    # Sync localized workspace pipeline binary to Hub platform
    hf_api.upload_file(
        path_or_fileobj=LOCAL_MODEL_PATH,
        path_in_repo=LOCAL_MODEL_PATH,
        repo_id=TARGET_HF_REPO,
        repo_type=HUB_REPO_TYPE,
    )

print(f"SUCCESS: Synchronized binary state to Hugging Face: {TARGET_HF_REPO}")
print("\n" + "#"*60)
print("             PROCESS SUCCESS: PIPELINE FINISHED              ")
print("#"*60)

multiprocessing.active_children()  # Collect and synchronize process trackers
for process in multiprocessing.active_children():
    process.terminate()  # Forces a clean shutdown signal
    process.join()       