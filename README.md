# 🏦 BOB AI Binary Classification — AML / Fraud

A Streamlit-based **binary classification platform** designed for AML, fraud detection, and other tabular binary-classification use cases.

The application allows users to:

* Upload CSV or Excel datasets
* Select any binary target column
* Automatically preprocess categorical and numerical features
* Train a small feed-forward neural network using TensorFlow/Keras
* Handle class imbalance using class weights
* Optimize the prediction threshold using F1-score
* Evaluate the trained model using multiple classification metrics
* Visualize training performance
* Run predictions on new datasets
* Persist the trained model and preprocessing artifacts to a **private GitHub Gist**
* Restore the trained model after Streamlit Community Cloud restarts or redeployments

---

# 📌 Overview

The application provides an end-to-end workflow:

```text
                 🏦 BOB AI Binary Classification
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
        🏋️ TRAIN                         🔮 PREDICT
             │                                 │
      Upload Dataset                     Upload New Data
             │                                 │
      Select Target Column                     │
             │                                 │
      Data Sampling                            │
             │                                 │
      Feature Encoding                         │
             │                                 │
      Feature Scaling                          │
             │                                 │
      Train Neural Network                     │
             │                                 │
      Early Stopping                           │
             │                                 │
      Best Threshold                           │
             │                                 │
      Model Evaluation ────────────────► Predictions
             │                                 │
             └──────────────┬──────────────────┘
                            │
                            ▼
                   ☁️ GitHub Gist
                    Model Persistence
```

The application is particularly suited to **proof-of-concept AI/ML workflows in banking, AML, fraud detection, and other structured-data classification problems**.

---

# ✨ Key Features

## 🏋️ 1. Model Training

Upload a labeled dataset in:

* CSV
* Excel (`.xlsx`)

The user selects the target column, which must contain exactly two unique classes.

The application then:

1. Identifies feature columns
2. Separates the target
3. Encodes categorical features
4. Encodes the target
5. Standardizes numerical/encoded features
6. Creates stratified training and test sets
7. Calculates class weights
8. Builds a feed-forward neural network
9. Trains using Keras
10. Restores the best validation F1-score weights

---

# 🎯 Binary Target Requirement

The selected target column must contain exactly **two unique values**.

For example:

```text
Fraud
-----
0
1
```

or:

```text
Transaction_Status
------------------
Normal
Fraud
```

If the selected target contains more or fewer than two unique values, training is stopped.

---

# 📂 Supported Input Formats

The training and prediction interfaces support:

```text
.csv
.xlsx
```

CSV files are loaded using:

```python
pd.read_csv()
```

Excel files are loaded using:

```python
pd.read_excel()
```

---

# 🔢 Data Sampling

The application provides a **Data Sample Percentage** slider ranging from:

```text
1% → 100%
```

When less than 100% is selected, the dataset is sampled using stratified sampling based on the **target column selected by the user**.

```python
train_test_split(
    df,
    train_size=portion / 100.0,
    random_state=42,
    stratify=df[target_column]
)
```

This helps preserve the target-class distribution in the sampled dataset.

---

# 🔤 Feature Preprocessing

## Categorical Features

Object and category columns are automatically identified.

Each categorical feature receives its own `LabelEncoder`.

```python
LabelEncoder()
```

The fitted encoders are retained and saved with the trained model.

This is important because inference must use the **same category-to-number mapping that was learned during training**.

---

# 📏 Feature Scaling

After categorical encoding, the feature matrix is standardized using:

```python
StandardScaler()
```

The fitted scaler is saved with the trained model.

During prediction, the same scaler is reused:

```python
scaler.transform(...)
```

This ensures that training and inference use the same feature transformation.

---

# 🎯 Target Encoding

The target column is also encoded using:

```python
LabelEncoder()
```

The fitted target encoder is persisted alongside the model.

During inference, predicted numeric classes are converted back into the original target labels.

---

# ⚖️ Class Imbalance Handling

AML and fraud datasets frequently contain imbalanced classes.

The application calculates balanced class weights using:

```python
compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
```

The resulting weights are supplied to Keras during training.

Conceptually:

```text
Majority Class ────────────────► Lower Weight
Minority Class ────────────────► Higher Weight
```

This helps prevent the neural network from simply favoring the majority class.

---

# 🧠 Neural Network Architecture

The application uses a small feed-forward neural network built with TensorFlow/Keras.

The architecture is:

```text
Input Features
      │
      ▼
Dense Layer 1
      │
   ReLU
      │
 Optional Dropout
      │
      ▼
Dense Layer 2
      │
   ReLU
      │
 Optional Dropout
      │
      ▼
Output Layer
   Sigmoid
      │
      ▼
Binary Prediction
```

The default configuration is:

```text
Layer 1: 64 units
Layer 2: 32 units
Dropout: 0.0
Learning Rate: 0.0001
Maximum Epochs: 200
Early-Stopping Patience: 10
```

These parameters can be modified from the sidebar.

---

# ⚙️ Model Settings

The sidebar provides controls for:

| Setting                 | Range / Default      |
| ----------------------- | -------------------- |
| Layer 1 units           | 4–512, default 64    |
| Layer 2 units           | 4–512, default 32    |
| Dropout rate            | 0.0–0.7, default 0.0 |
| Learning rate           | `1e-5` to `1e-3`     |
| Maximum epochs          | 10–2000, default 200 |
| Early-stopping patience | 3–100, default 10    |

---

# 🚀 Optimizer

The model uses Keras `AdamW`:

```python
AdamW(
    learning_rate=lr,
    weight_decay=1e-6
)
```

The loss function is:

```text
Binary Cross-Entropy
```

The model tracks:

```text
Accuracy
F1-score
```

---

# 🛑 Early Stopping

Training uses Keras' native `EarlyStopping` callback.

The validation F1-score is monitored:

```python
EarlyStopping(
    monitor="val_f1_metric",
    mode="max",
    patience=patience,
    restore_best_weights=True
)
```

This means that when training finishes, the model automatically restores the weights corresponding to the best validation F1-score.

This avoids retaining a later epoch whose performance may have degraded.

---

# 📊 Training Progress

The application displays a lightweight progress indicator during training.

For each epoch, the status includes:

```text
Epoch
Loss
Accuracy
F1
Validation Loss
Validation Accuracy
Validation F1
```

The application intentionally does **not** regenerate the full set of charts after every epoch.

This significantly reduces Streamlit rendering overhead during model training.

---

# 📈 Model Evaluation

After training, the application evaluates the model on the held-out test set.

The primary metrics displayed are:

### Accuracy

Measures the overall proportion of correctly classified observations.

### Loss

Reports the binary cross-entropy loss.

### F1-score

Provides a combined measure of Precision and Recall.

The application also provides visual evaluation outputs.

---

# 📊 Training Curves

The training dashboard displays:

```text
Accuracy over epochs
F1 over epochs
Loss over epochs
```

Each chart shows:

* Training metric
* Validation metric
* Best epoch

The best epoch is marked using a vertical reference line.

---

# 🔍 Prediction Threshold Optimization

The default binary threshold is:

```text
0.50
```

However, the application searches for the threshold that maximizes F1-score on the held-out test set.

It uses:

```python
precision_recall_curve()
```

instead of repeatedly evaluating hundreds of manually selected thresholds.

Conceptually:

```text
Model Probability
       │
       ▼
Threshold Search
       │
       ▼
Maximum F1
       │
       ▼
Recommended Threshold
```

The user can then adjust the prediction threshold using a slider:

```text
0.00 → 1.00
```

---

# 📌 Confusion Matrix

The application generates a confusion matrix showing:

```text
                 Predicted
              0          1
Actual  0    TN         FP
        1    FN         TP
```

This is particularly useful for AML/fraud use cases because false positives and false negatives can have very different operational implications.

---

# 📋 Classification Report

The application generates a classification report containing metrics such as:

* Precision
* Recall
* F1-score
* Support

The report is displayed visually using a heatmap.

---

# 📈 ROC Curve

The application also generates a Receiver Operating Characteristic (ROC) curve.

The chart displays:

```text
True Positive Rate
        │
        │       ╭────
        │     ╭─
        │   ╭─
        │ ╭─
        └──────────────────
          False Positive Rate
```

The area under the ROC curve is displayed as:

```text
AUC = ...
```

---

# 🔮 Prediction

The **Predict** tab allows users to upload new CSV or Excel data.

The prediction dataset must contain all feature columns that were used during training.

The application verifies the required columns before performing inference.

---

# 🔄 Consistent Inference Preprocessing

A major feature of this implementation is that inference reuses the exact preprocessing artifacts learned during training.

The following are reused:

```text
Feature LabelEncoders
Target LabelEncoder
StandardScaler
Feature Column Order
Prediction Threshold
```

This avoids a common machine-learning error where categorical encoders are fitted again on inference data.

---

# ⚠️ Unseen Categories

If an inference dataset contains a categorical value that was not observed during training, the application detects it.

For example:

```text
Training:
Gold
Silver
Bronze

Inference:
Gold
Silver
Platinum
```

`Platinum` was not present during training.

The application maps unseen categories to a safe placeholder based on the first known encoder class instead of allowing `LabelEncoder.transform()` to fail.

The application also displays a warning indicating the affected feature and number of rows.

---

# 📊 Prediction Output

The result contains the original input data plus:

```text
Prediction
Prediction_Probability
```

Example:

| Transaction | Prediction | Prediction_Probability |
| ----------- | ---------- | ---------------------: |
| TX001       | Normal     |                  0.083 |
| TX002       | Fraud      |                  0.921 |
| TX003       | Normal     |                  0.274 |

The prediction labels are converted back to the original target labels using the saved target encoder.

---

# ⬇️ Download Predictions

Prediction results can be downloaded as:

```text
predictions.csv
```

using the Streamlit download button.

---

# 🧪 Prediction Dataset Evaluation

If the prediction file also contains the original target column, the application automatically calculates:

* Accuracy
* F1-score
* Confusion Matrix
* Classification Report

This allows labeled datasets to be used for post-training validation.

---

# ☁️ GitHub Gist Model Persistence

One of the key features of this application is persistent model storage.

Streamlit Community Cloud uses an ephemeral filesystem. A trained model stored only on the local filesystem may disappear when the application restarts or is redeployed.

To solve this, the application stores the trained artifacts in a **private GitHub Gist**.

The following artifacts are persisted:

```text
model.b64
scaler.pkl.b64
feature_encoders.pkl.b64
target_encoder.pkl.b64
meta.json
```

---

# 💾 Persisted Model Components

## Model

The Keras model is saved in `.keras` format and then Base64 encoded.

```text
model.b64
```

## Feature Scaler

The fitted `StandardScaler` is serialized using pickle:

```text
scaler.pkl.b64
```

## Feature Encoders

All training-time categorical `LabelEncoder` objects are stored:

```text
feature_encoders.pkl.b64
```

## Target Encoder

The target `LabelEncoder` is stored separately:

```text
target_encoder.pkl.b64
```

## Metadata

The application stores:

```json
{
    "feature_columns": [],
    "target_column": "...",
    "threshold": 0.5,
    "saved_at_utc": "..."
}
```

This allows the application to reconstruct the required inference configuration.

---

# 🔐 GitHub Gist Setup

A GitHub Personal Access Token with **Gist access** is required.

The application expects:

```text
GITHUB_TOKEN
GIST_ID
```

---

# 🛠️ Local Secrets Configuration

Create:

```text
.streamlit/secrets.toml
```

Example:

```toml
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
GIST_ID = ""
```

On the first save, the application can create a new private Gist.

After saving, copy the displayed Gist ID into:

```toml
GIST_ID = "your-gist-id"
```

---

# ☁️ Streamlit Cloud Configuration

For Streamlit Community Cloud:

1. Open the application's settings.
2. Open **Secrets**.
3. Add:

```toml
GITHUB_TOKEN = "your-github-token"
GIST_ID = "your-gist-id"
```

4. Save the secrets.
5. Restart/redeploy the application if required.

The token is entered as a password field in the sidebar when the application is running.

---

# 🔒 Security Considerations

The application creates/updates the Gist as:

```text
public = False
```

However, a secret/private Gist should **not** be treated as equivalent to enterprise-grade private storage.

Do not store sensitive production data in the Gist.

Particular caution should be exercised for:

* Customer data
* Personally identifiable information
* Transaction records
* Account information
* KYC information
* Confidential banking data
* Production AML investigation data
* Production fraud case information

For production environments, an approved enterprise storage mechanism should be considered.

---

# 🏦 AML / Fraud Use Case

The application is designed to support binary classification scenarios such as:

```text
Normal Transaction
        vs.
Fraudulent Transaction
```

or:

```text
Legitimate Activity
        vs.
Suspicious Activity
```

Potential feature inputs could include:

```text
Transaction Amount
Transaction Frequency
Customer Age
Account Tenure
Transaction Type
Channel
Geography
Device Information
Historical Activity
Risk Indicators
```

The exact features depend on the dataset and use case.

---

# 🧠 Why F1-score Matters

In highly imbalanced fraud or AML datasets, accuracy alone can be misleading.

For example, if only a small percentage of transactions are fraudulent, a model could achieve high accuracy simply by predicting the majority class.

F1-score provides a combined view of:

```text
Precision
    +
Recall
    ↓
F1-score
```

The application therefore monitors F1 during training and uses validation F1 for early stopping.

---

# ⚖️ Threshold Selection

The prediction threshold directly affects classification behavior.

For example:

```text
Probability ≥ Threshold
        │
        ▼
Positive Class
```

Changing the threshold can alter the balance between:

* False positives
* False negatives
* Precision
* Recall
* F1-score

The application therefore provides an adjustable threshold rather than permanently forcing `0.50`.

---

# 🧩 Application Tabs

The user interface contains three main tabs.

## 🏋️ Train

Used for:

* Dataset upload
* Target selection
* Sampling
* Model training
* Model evaluation
* Threshold optimization
* Training visualizations

---

## 🔮 Predict

Used for:

* Uploading new data
* Running inference
* Viewing predictions
* Viewing prediction probabilities
* Downloading predictions
* Evaluating predictions when true labels are available

---

## ℹ️ About

Provides information about:

* Application workflow
* Preprocessing
* Model persistence
* GitHub Gist configuration

---

# 🎨 Bank of Baroda UI

The application uses a Bank of Baroda-inspired color palette.

| Color       | Hex       |
| ----------- | --------- |
| BOB Orange  | `#F7941D` |
| Deep Orange | `#E8531B` |
| BOB Maroon  | `#8E1B3A` |
| BOB Navy    | `#12284C` |
| Light Navy  | `#1E3E73` |
| Cream       | `#FFF8F1` |
| Grey        | `#5B6675` |

The interface includes a branded header:

```text
🔍 Binary Classification — AML / Fraud
Upload labeled data to train a model, then score new records with it.
```

---

# 🏗️ Application Architecture

```text
                       Streamlit Application
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
            Train           Predict           About
              │                │
              ▼                ▼
       Uploaded Dataset    New Dataset
              │                │
              ▼                │
      Target Selection        │
              │                │
              ▼                │
      Categorical Encoding    │
              │                │
              ▼                │
       Standard Scaling       │
              │                │
              ▼                │
      Train/Test Split        │
              │                │
              ▼                │
       Class Weighting        │
              │                │
              ▼                │
       Keras Neural Net       │
              │                │
              ▼                │
       Early Stopping         │
              │                │
              ▼                │
       Threshold Search       │
              │                │
              ▼                │
       Model + Artifacts ──────┤
              │                │
              ▼                ▼
        GitHub Gist       Preprocessing
                           + Inference
```

---

# 📁 Recommended Project Structure

```text
BOB-AI-Binary-Classifier/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── secrets.toml
```

---

# 📦 Requirements

A typical `requirements.txt` can contain:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
requests
pytz
openpyxl
```

`openpyxl` is required when Excel `.xlsx` files are uploaded.

---

# 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 Example Workflow

## Step 1 — Upload Training Data

Open:

```text
🏋️ Train
```

Upload:

```text
transactions.csv
```

---

## Step 2 — Select Target

Select the binary target:

```text
Fraud
```

The application verifies that the column contains exactly two classes.

---

## Step 3 — Select Data Percentage

Choose how much of the dataset should be used.

For example:

```text
Data sample percentage: 80%
```

Sampling is stratified using the selected target column.

---

## Step 4 — Configure the Model

Example:

```text
Layer 1 units: 64
Layer 2 units: 32
Dropout: 0.10
Learning rate: 0.0001
Max epochs: 200
Patience: 10
```

---

## Step 5 — Train

Click:

```text
🚀 Train model
```

The application trains the neural network and restores the best validation-F1 weights.

---

## Step 6 — Review Performance

Review:

```text
Accuracy
Loss
F1-score
```

along with:

* Training curves
* Confusion matrix
* Classification report
* ROC curve
* ROC-AUC

---

## Step 7 — Select Prediction Threshold

The application automatically identifies an F1-maximizing threshold on the held-out test data.

The threshold can then be adjusted manually.

---

## Step 8 — Save the Model

From the sidebar:

```text
☁️ Model persistence (GitHub Gist)
```

Click:

```text
💾 Save to Gist
```

---

## Step 9 — Restore Later

After a Streamlit Cloud restart/redeployment:

```text
📥 Load from Gist
```

The application restores:

```text
Keras Model
StandardScaler
Feature Encoders
Target Encoder
Feature Columns
Target Column
Prediction Threshold
```

---

## Step 10 — Predict New Data

Open:

```text
🔮 Predict
```

Upload the new CSV/Excel dataset.

The application applies the original training preprocessing and generates predictions.

---

# 🔧 Improvements Over the Original Implementation

This implementation includes several important corrections and performance improvements.

## 1. Consistent Label Encoding

### Previous behavior

Inference could fit a new encoder on inference data.

This could silently change the numerical representation of categories.

### Current behavior

The exact training-time encoders are saved and reused.

```text
Training Encoder
       │
       ▼
Persisted
       │
       ▼
Inference
```

This maintains consistent feature representation.

---

## 2. Correct Stratified Sampling

The sampling process now uses the target column selected by the user:

```python
stratify=df[target_column]
```

rather than implicitly relying on another column.

---

## 3. Correct Best-Epoch Restoration

The application uses:

```python
restore_best_weights=True
```

with Keras `EarlyStopping`.

This ensures that the model retained after training corresponds to the best monitored validation F1-score.

---

## 4. More Efficient Training

The previous implementation repeatedly called:

```text
model.fit(epochs=1)
```

and regenerated plots and evaluation outputs after each epoch.

The current implementation uses one:

```python
model.fit(...)
```

call with callbacks.

This significantly reduces unnecessary rendering and evaluation overhead.

---

## 5. Vectorized Threshold Search

The threshold search uses:

```python
precision_recall_curve()
```

instead of manually evaluating 101 thresholds.

This provides a more efficient method for identifying the F1-maximizing threshold.

---

# 🛡️ Production Considerations

This application is primarily suitable for **PoC, experimentation, and analytical workflows**.

For production AML/fraud deployment, additional controls may be required, including:

* Model governance
* Data lineage
* Feature validation
* Model versioning
* Explainability
* Drift monitoring
* Bias/fairness assessment
* Audit logging
* Access control
* Secure model storage
* Encryption
* Threshold governance
* Human review workflows
* Regulatory controls
* Production-grade monitoring

The neural network's predictions should not be treated as an autonomous decision without appropriate business and governance controls.

---

# ⚠️ Limitations

The current implementation is intentionally lightweight.

Potential limitations include:

* LabelEncoder is ordinal rather than target-mean or one-hot encoding.
* Unseen categorical values are mapped to a placeholder category.
* The model is a relatively small feed-forward neural network.
* No automated hyperparameter optimization is included.
* No feature-selection framework is included.
* No model explainability framework is included.
* No drift monitoring is included.
* GitHub Gist is used as a PoC persistence mechanism rather than an enterprise model registry.
* Prediction performance depends heavily on the quality and representativeness of the supplied dataset.

---

# 📈 Potential Future Enhancements

Possible extensions include:

### Model Enhancements

* XGBoost
* LightGBM
* Random Forest
* Logistic Regression
* TabNet
* Ensemble models
* Hyperparameter optimization

### Explainability

* SHAP
* LIME
* Feature importance
* Individual prediction explanations

### AML/Fraud Features

* Transaction velocity analysis
* Customer risk profiling
* Device fingerprinting
* Geographic anomalies
* Behavioral features
* Network/graph-based fraud detection
* Sequence-based transaction modeling

### Monitoring

* Data drift
* Prediction drift
* Concept drift
* Threshold monitoring
* Model performance monitoring

### Model Management

* Model versioning
* Model registry
* Training metadata
* Experiment tracking
* Audit trails

---

# 🏦 BOB AI Use-Case Positioning

The application can serve as a reusable binary-classification component within a broader banking AI platform.

```text
                         🏦 BOB AI
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
        AML              Fraud             Risk
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                Binary Classification
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
          Training       Evaluation      Inference
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     AI Decision Support
```

The same application architecture can therefore be adapted to different binary classification problems by changing the uploaded dataset and target variable.

---

# 📄 License

Add the appropriate project license here.

For example:

```text
MIT License
```

Third-party libraries retain their respective licenses.

---

# ⚠️ Disclaimer

This application is intended for **research, proof-of-concept, experimentation, and analytical purposes**.

It does not by itself constitute a production AML/fraud detection system.

Model predictions are probabilistic outputs and should be evaluated within the relevant business, regulatory, risk-management, and model-governance framework before being used in production.

---

# 🏦 Summary

**BOB AI Binary Classification — AML / Fraud** provides an end-to-end Streamlit workflow for:

```text
📂 Upload Data
      ↓
🎯 Select Binary Target
      ↓
🔤 Encode Features
      ↓
📏 Scale Features
      ↓
⚖️ Handle Class Imbalance
      ↓
🧠 Train Keras Model
      ↓
🛑 Early Stop on Validation F1
      ↓
🎯 Optimize Prediction Threshold
      ↓
📊 Evaluate Model
      ↓
💾 Persist Model + Preprocessing
      ↓
☁️ Private GitHub Gist
      ↓
🔮 Restore & Predict
      ↓
⬇️ Download Results
```

The application combines **TensorFlow/Keras, scikit-learn, Streamlit, GitHub Gist persistence, and Bank of Baroda-inspired UI** into a reusable binary-classification framework for AML, fraud, and related banking AI proof-of-concept use cases.
