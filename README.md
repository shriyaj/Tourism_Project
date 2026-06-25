## 🗺️ Wellness Tourism Package Prediction
## An End-to-End MLOps Pipeline
------------------------------
## 📋 Project Overview
This project implements an end-to-end MLOps pipeline for predicting whether customers will purchase the **Wellness Tourism Package** from "Visit with Us" travel company.

## 🔗 Live Demonstration

* Interactive Web App: View Live on Hugging Face Spaces

------------------------------
## ⚡ Quick Start

# Clone the repository
```bash
git clone https://github.com/shriyaj/Tourism_Project.git
cd Tourism_Project
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
# Install main workflow dependencies
pip install -r tourism_project/requirements.txt
# Run the frontend application locally
cd tourism_project/deployment
streamlit run app.py
```
⚠️ Note: Running the application locally requires a pre-trained model artifact. See Deploying via GitHub Actions or try the Live Demo directly.

------------------------------
## 🗂️ Project Architecture
```
tourism_project/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # CI/CD automated deployment workflow
├── data/
│   └── tourism.csv               # Raw historical client dataset
├── deployment/
│   ├── app.py                    # Streamlit responsive user interface
│   ├── Dockerfile                # Container deployment blueprint
│   └── requirements.txt          # App-specific cloud dependencies
├── hosting/
│   └── hosting.py                # Automated Hub push & sync script
├── model_building/
│   ├── data_register.py          # Dataset registration & versioning
│   ├── prep.py                   # Feature engineering & preprocessing
│   └── train.py                  # XGBoost training & hyperparameter tuning
└── requirements.txt              # Complete pipeline workflow dependencies
```
------------------------------
## 🎯 Production Core Features
## 1. Data Versioning & Feature Engineering

* Automated Data Lifecycle: Seamless data ingestion and programmatic dataset registration directly to the Hugging Face Hub.
* Robust Preprocessing: Built-in pipelines for clean imputation of missing variables and programmatic handling of layout anomalies.
* Feature Encoding: Strict categorical transformation via targeted label encoders.
* Scientific Evaluation Splits: Implements a strict stratified 80/20 train-test split to protect against target shift and ensure high training stability.

------------------------------
## 2. 🤖 Model Training & Experimentation

* Data Streaming Ingestion: Programmatically streams segmented partitioned data matrix files directly from the Hugging Face Hub dataset repository.
* Pre-processing Pipelines: Implements automated feature scaling using a robust StandardScaler inside a make_column_transformer layout.
* Multi-Architecture Evaluation: Evaluates multiple ensemble architectures to satisfy strict project rubric parameters:
* Random Forest Classifier: Balanced vs. unbalanced weight distributions.
   * Gradient Boosting Classifier: Advanced boosting structures with deep feature interaction depths.
   * XGBoost Classifier: Advanced extreme gradient boosting with programmatic scaling for class imbalances (scale_pos_weight).
* Hyperparameter Optimization Suite: Employs a parallelized GridSearchCV with 3-Fold Cross-Validation optimized directly to maximize the ROC-AUC evaluation curve space.
* Hierarchical Experiment Tracking: Integrates native MLflow hooks to chart and track your complete modeling landscape:
* Parent Run Scope: Model_Comparison_Suite encapsulates the global execution experiment.
   * Nested Run Architecture: Tracks hyperparameter sweeps (param_sweep) independently for full transparency.
   * Deep Structural Metrics: Logs complete testing signatures across multiple dimensions, including:
   * Accuracy & Precision stability matrices (handling zero-division anomalies safely)
      * Recall metrics
      * F1-Score diagnostics
      * ROC-AUC scoring
   * Championship Model Identification: Automatically compares real-time evaluation benchmarks, isolating the single highest-performing pipeline package configuration to serialize for the deployment interface.

------------------------------
## 3. 🌐 Deployment Strategy

* Production UI: Built an interactive user interface via Streamlit configured with relative metrics mapping to protect layout scaling.
* Hermetic Containerisation: Configured an optimized Dockerfile image to standardize production builds independent of local environments.
* High-Availability Hosting: Deploys automatically to Hugging Face Spaces inside a managed cloud container environment.

## 4. 🚀 CI/CD Automation Matrix (GitHub Actions)
The infrastructure utilizes a modular GitHub Actions automation workflow split into four discrete, sequential, and state-tracked jobs:

graph TD
    A[Job 1: Data Registration] --> B[Job 2: Data Preparation]
    B --> C[Job 3: Model Training]
    C --> D[Job 4: Automated Space Deployment]


   1. Dataset Registration: Intercepts the raw data payload and securely versions the asset partition within the Hugging Face Hub dataset ecosystem.
   2. Data Preparation: Programmatically triggers outlier remediation, feature scaling transformers, and stratified dataset splitting.
   3. Model Training: Mounts the hyperparameter search matrix to cross-evaluate Random Forest, Gradient Boosting, and XGBoost variants while simultaneously piping parameter logs to a dedicated MLflow server.
   4. Automated Space Deployment: Builds the final container blueprint and pushes the complete application to Hugging Face Spaces using encrypted API endpoints.

------------------------------
## 🛠️ Infrastructure Prerequisites
To replicate this production matrix, execute the following setup sequence:
## 1. Account Token Provisioning

   1. Sign up or log into the Hugging Face Platform.
   2. Navigate directly to Settings → Access Tokens.
   3. Generate a new token. Set the scope permissions strictly to Write access. Copy this string securely.

## 2. Encrypted GitHub Secrets Configuration
To empower the automated runner pipelines without exposing plaintext keys, bind your tokens directly to your GitHub repository context:

   1. Open your project repository on GitHub.
   2. Navigate to: Settings → Secrets and Variables → Actions.
   3. Select New repository secret and provision the following key-value layout:
   * Name: HF_TOKEN
      * Value: [Paste your copied Hugging Face write token here]
   
## 3. Remote Cloud Space Configuration
Create an ingestion endpoint on Hugging Face to host your code artifacts:

* Target Space Name: wellness-tourism-prediction
* Target SDK Workspace Engine: Docker (Select the Streamlit blueprint configuration template)

------------------------------
## ⚙️ Project Configuration & Setup
## 🪪 Required Repository Reference Updates
```
⚠️ CRITICAL STEP: To ensure the data streams to your personal account instead of a placeholder path, replace <---repo id----> with your actual Hugging Face Username across these files before your initial push:


* tourism_project/model_building/data_register.py
repo_id = "YOUR_HF_USERNAME/tourism-dataset"
* tourism_project/model_building/prep.py
DATASET_PATH = "hf://datasets/YOUR_HF_USERNAME/tourism-dataset/tourism.csv"
repo_id = "YOUR_HF_USERNAME/tourism-dataset"
* tourism_project/model_building/train.py
TRAIN_FEATURES_URL = "hf://datasets/YOUR_HF_USERNAME/tourism-dataset/Xtrain.csv" (Update all 4 data URL string targets)
repo_id = "YOUR_HF_USERNAME/tourism-prediction-model"
* tourism_project/deployment/app.py
repo_id = "YOUR_HF_USERNAME/tourism-prediction-model"
* tourism_project/hosting/hosting.py
repo_id = "YOUR_HF_USERNAME/wellness-tourism-prediction"
```
------------------------------
## 💻 Local Workspace Execution
## 1. Project Initialization
```bash
# Clone the repository workspace
git clone https://github.com/shriyaj/Tourism_Project.git
cd Tourism_Project
# Install pipeline dependency packages
pip install -r tourism_project/requirements.txt

## 2. Sequential Step-by-Step Execution
Run each module sequentially to populate your remote endpoints and track metrics:

# Step 1: Export your token and register the baseline raw data partition
export HF_TOKEN="your_hugging_face_token"  # Windows Command Prompt: set HF_TOKEN="your_token"
python tourism_project/model_building/data_register.py
# Step 2: Run outlier analysis and structural transformations
python tourism_project/model_building/prep.py
# Step 3: Boot your tracking server and initiate grid search optimization
mlflow ui --host 0.0.0.0 --port 5050 &
python tourism_project/model_building/train.py
# Step 4: Programmatically push artifacts to your active Space container
python tourism_project/hosting/hosting.py

## 3. Alternative: Launch UI Locally
If you already have your model serialization arrays saved locally and want to test the interactive dashboard interface directly, run:

cd tourism_project/deployment
streamlit run app.py 
or
streamlit run app.py --theme.backgroundColor="#FFFFFF" --theme.secondaryBackgroundColor="#F0F2F6" --theme.textColor="#31333E" --theme.primaryColor="#FF4B4B" # to run in dark mode
```
------------------------------
## 🚀 Cloud Deployment Via GitHub Actions
## 1. Pre-Flight Repository Checklist
Before triggering the automated cloud runner, ensure your workspace is verified:

* All <---repo id----> references are updated to your actual Hugging Face Username.
* Your HF_TOKEN write key is safely mounted within your GitHub Repository Secrets.

## 2. Triggering the Automation Matrix
Commit your workspace code to your remote source branch to initialize the container orchestration runner:

# Stage, package, and snapshot local assets
git add .
git commit -m "feat: complete automated mlops infrastructure pipeline"
# Stream local commits to the primary deployment branch
git push origin main

## 3. Monitoring Infrastructure & Access

   1. Open your repository workspace directly on GitHub.
   2. Click the Actions tab to trace the live, sequential, four-job execution output logs.
   3. Once the build returns a green status, launch your provisioned live instance endpoint at:
   https://huggingface.co

------------------------------
## 📊 Operational Metric Signatures
The core execution environment uses an optimization layer to continuously contrast multiple modeling paradigms.
## 1. Matrix Feature Architecture

* Total Ingested Features: 17 structured matrices (including buyer demographic vectors, customer journey logs, and historic financial metrics).
* Target Classification Objective: Binary Variant Mapping (1 = Client Purchases Wellness Package, 0 = Offer Denied).

## 2. Runtime Evaluation Architecture
The production script evaluates cross-validated models (Random Forest, Gradient Boosting, and XGBoost) inside an optimization loop, calculating and streaming the following evaluation arrays to MLflow:

* Accuracy Score: Checks strict target alignment across test partitions.
* Precision Profile: Gauges predictive stability to minimize false positives in targeted outreach.
* Recall / Sensitivity Matrix: Protects business margins by identifying all true potential buyers.
* F1-Score Diagnostic: Harmonic balance indicator across skewed customer segments.
* ROC-AUC Space: The final sorting benchmark used to programmatically crown the production champion.

------------------------------
## 🎨 Interactive Frontend Interface Capabilities
The Streamlit UI application engine exposes a professional dashboard interface with the following production capabilities:

* Asymmetric Grid Structuring: Form layouts split across a clean, responsive two-column entry space.
* Client Demographic Field Arrays: Interactive components capturing customer profiles (e.g., age, occupation profile, marital matrix, and income parameters).
* Engagement Interaction Signatures: Controls to monitor pitch durations, historic follow-up frequencies, and product tier interests.
* Real-Time Prediction Engine: Uses backend serialisation streams to return predictive metrics alongside a live probability confidence gauge.
* Dynamic Business Logic Rules: Automatically evaluates model inference outputs to print context-aware customer conversion steps.

------------------------------
## 🔬 Multi-Model Experiment Tracking Layout
The automated python search core leverages an embedded MLflow infrastructure to securely isolate performance metrics and manage structural model provenance:

* Hyperparameter Matrix Profiling: Automatically tracks dense hyperparameter arrays across nested runs (including tree counts, maximum depths, objective weights, and learning rate variations).
* Comprehensive Metrics History: Maps real-time evaluation performance dimensions across multiple test validations directly to a centralized tracking dashboard.
* Hermetic Artifact Storage: Automatically isolates the best-performing serialization objects and metadata configurations to streamline production deployment routines.
* Architecture Comparison Framework: Exposes visual comparison capabilities inside the MLflow UI to analyze model variances and monitor over-fitting risks across your parameter sweeps.

------------------------------
## 📦 Ingested Dataset Matrix
The production pipeline reads customer journey arrays mapped against a binary target outcome to predict business conversion probabilities:
## 1. Model Features Breakdown

* Client Demographic Vectors: Captures core background data arrays including client age profiles, gender variables, and occupational industry matrices.
* Travel Preference Footprints: Monitors historical client interest points, including target tier ratings (PropertyStar) and historical booking counts (NumberOfTrips).
* Interaction Engagement Signatures: Traces conversion touchpoints, specifically salesperson engagement times (PitchDuration), follow-up touchpoint metrics, and post-pitch survey outputs (SatisfactionScore).
* Financial Position Benchmarks: Tracks disposable income indices (MonthlyIncome) paired with structural lifestyle indicator vectors (OwnCar).

## 2. Analytical Target Objective

* Target Array Vector (ProdTaken): Evaluates a binary classification label context where 0 denotes an offer denial and 1 establishes a successful purchase of the specialized Wellness Tourism Package.

------------------------------
## 🛠️ Unified System Technology Stack
The engineering pipeline combines performance-focused open-source core libraries with cloud-native automation frameworks:

* Runtime Core: Python 3.12 (Optimized for modern virtual environment structures).
* Statistical Inference: XGBoost & scikit-learn (Ensemble gradient tree architectures paired with scalable validation workflows).
* Experiment Management: MLflow (Hierarchical multi-run tracing configurations).
* Interface Presenter: Streamlit (Responsive multi-column parameter inputs).
* Isolation Layer: Docker (Immutable baseline runtime container images).
* Orchestration Runner: GitHub Actions (Autonomous CI/CD code evaluation pipelines).
* Cloud Storage Endpoints: Hugging Face Hub (Secure, version-controlled remote storage systems for data matrices, model weights, and container apps).

## 📝 Assignment Submission Checklist

- [x] Complete folder structure created
- [x] Data registration script (`data_register.py`)
- [x] Data preparation script (`prep.py`)
- [x] Model training script with MLflow (`train.py`)
- [x] Streamlit application (`app.py`)
- [x] Dockerfile for deployment
- [x] Deployment requirements.txt
- [x] Hosting script (`hosting.py`)
- [x] GitHub Actions workflow (`pipeline.yml`)
- [x] Workflow requirements.txt
- [x] Jupyter notebook with all code cells filled
- [x] GitHub repository created
- [x] HF_TOKEN added to GitHub Secrets
- [x] Hugging Face Space created
- [] Pipeline executed successfully
- [ ] Screenshots of:
  - GitHub repository structure
  - GitHub Actions workflow execution
  - Deployed Streamlit app on Hugging Face

## 📸 Output Requirements

### 1. GitHub Repository
- Screenshot showing folder structure
- Screenshot showing successful workflow execution

### 2. Hugging Face Space
- Link to deployed application
- Screenshot of the Streamlit app in action

## 🔍 Troubleshooting

### Common Issues:

1. **HF_TOKEN not found**:
   - Ensure the token is added to GitHub Secrets
   - Verify the secret name is exactly `HF_TOKEN`

2. **Import errors**:
   - Check all dependencies are in requirements.txt
   - Verify correct versions are specified

3. **Model not loading in Streamlit**:
   - Ensure model is uploaded to correct HF repository
   - Check repo_id matches across files

4. **GitHub Actions failing**:
   - Check workflow logs for specific errors
   - Verify all file paths are correct
   - Ensure requirements.txt includes all dependencies

## 👨‍💻 Author

This MLOps pipeline was developed as part of the Advanced Machine Learning and MLOps course assignment by Shriya Jain.

## 📄 License

This project is licensed under the [MIT License](LICENSE).


