"""
Binary Classification App (AML / Fraud use-cases)
--------------------------------------------------
- Train a small feed-forward Keras model on any binary-target tabular dataset.
- Run inference on new data using the trained model.
- Persist trained model + preprocessing artifacts to a private GitHub Gist so
  the app can reload them after Streamlit Community Cloud restarts/redeploys
  (the local filesystem there is ephemeral and is wiped on every reboot).

Fixes vs. the original version:
  * Inference now reuses the SAME LabelEncoders fit during training instead of
    re-fitting fresh encoders on the inference data (which silently produced
    wrong / inconsistent codes).
  * Stratified sampling of the "data percentage" slider now stratifies on the
    column the user actually picked as the target, not just the last column.
  * Best-epoch weights are restored via Keras' own EarlyStopping
    (restore_best_weights=True) instead of a hand-rolled tracker that never
    actually reloaded the checkpoint file.
  * Training runs as a single model.fit(..., callbacks=[...]) call instead of
    a Python loop that called model.fit(epochs=1) and then re-rendered three
    full plots + confusion matrix + classification report + ROC curve after
    EVERY epoch. That was the main reason training felt extremely slow.
  * Threshold search uses sklearn's precision_recall_curve (vectorized)
    instead of a 101-iteration Python loop repeated at every epoch.
"""

import base64
import io
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import AdamW

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BOB_CREAM};
    }}
    #bob-banner {{
        background: radial-gradient(circle at 15% 50%, {BOB_ORANGE} 0%, {BOB_ORANGE_DEEP} 45%, {BOB_MAROON} 100%);
        padding: 22px 30px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }}
    #bob-banner h1 {{
        color: white;
        margin: 0;
        font-size: 1.9em;
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    #bob-banner p {{
        color: #FFEFE0;
        margin: 4px 0 0 0;
        font-size: 0.95em;
    }}
    h1, h2, h3 {{ color: {BOB_NAVY}; }}
    /* BOB AI Portal Card Border */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1.5px solid {BOB_ORANGE_DEEP} !important;
        border-radius: 14px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Binary Classifier — AML / Fraud", layout="wide", page_icon="🔍")

IST = pytz.timezone("Asia/Kolkata")
GITHUB_API = "https://api.github.com"


# --------------------------------------------------------------------------- #
# Custom metric (must be available at load-time for restored models too)
# --------------------------------------------------------------------------- #
def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred > 0.5, "float32")
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * precision * recall / (precision + recall + K.epsilon())


CUSTOM_OBJECTS = {"f1_metric": f1_metric}


# --------------------------------------------------------------------------- #
# Session state defaults
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "model": None,
    "scaler": None,
    "feature_columns": None,
    "feature_encoders": None,   # dict: column -> fitted LabelEncoder (training-time)
    "label_encoder": None,      # LabelEncoder for the target
    "target_column": None,
    "threshold": 0.5,
    "history": None,
    "best_epoch": None,
    "train_flag": False,
    "prev_file": None,
    "gist_id": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# --------------------------------------------------------------------------- #
# GitHub Gist persistence helpers
# --------------------------------------------------------------------------- #
def _gh_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def save_artifacts_to_gist(token: str, gist_id: str) -> str:
    """Serialize the current model + preprocessing objects and push them to a
    private Gist. Returns the gist id (creates a new gist if none was given)."""
    tmp_path = "temp_model_save.keras"
    st.session_state.model.save(tmp_path)
    with open(tmp_path, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode()
    os.remove(tmp_path)

    files = {
        "model.b64": {"content": model_b64},
        "scaler.pkl.b64": {"content": base64.b64encode(pickle.dumps(st.session_state.scaler)).decode()},
        "feature_encoders.pkl.b64": {
            "content": base64.b64encode(pickle.dumps(st.session_state.feature_encoders)).decode()
        },
        "target_encoder.pkl.b64": {
            "content": base64.b64encode(pickle.dumps(st.session_state.label_encoder)).decode()
        },
        "meta.json": {
            "content": json.dumps(
                {
                    "feature_columns": st.session_state.feature_columns,
                    "target_column": st.session_state.target_column,
                    "threshold": st.session_state.threshold,
                    "saved_at_utc": datetime.utcnow().isoformat(),
                }
            )
        },
    }

    if gist_id:
        resp = requests.patch(f"{GITHUB_API}/gists/{gist_id}", headers=_gh_headers(token), json={"files": files})
    else:
        resp = requests.post(
            f"{GITHUB_API}/gists",
            headers=_gh_headers(token),
            json={"description": "Binary classifier artifacts (auto-saved)", "public": False, "files": files},
        )
    resp.raise_for_status()
    return resp.json()["id"]


def load_artifacts_from_gist(token: str, gist_id: str):
    resp = requests.get(f"{GITHUB_API}/gists/{gist_id}", headers=_gh_headers(token))
    resp.raise_for_status()
    files = resp.json()["files"]

    model_bytes = base64.b64decode(files["model.b64"]["content"])
    tmp_path = "temp_model_load.keras"
    with open(tmp_path, "wb") as f:
        f.write(model_bytes)
    model = load_model(tmp_path, custom_objects=CUSTOM_OBJECTS)
    os.remove(tmp_path)

    scaler = pickle.loads(base64.b64decode(files["scaler.pkl.b64"]["content"]))
    feature_encoders = pickle.loads(base64.b64decode(files["feature_encoders.pkl.b64"]["content"]))
    target_encoder = pickle.loads(base64.b64decode(files["target_encoder.pkl.b64"]["content"]))
    meta = json.loads(files["meta.json"]["content"])
    return model, scaler, feature_encoders, target_encoder, meta


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def status_metric(label, value, good, bad, higher_is_better=True):
    if higher_is_better:
        ok, warn = value >= good, value < bad
    else:
        ok, warn = value <= good, value > bad
    if ok:
        st.success(f"✅ {label}: {value:.4f}")
    elif warn:
        st.error(f"❌ {label}: {value:.4f}")
    else:
        st.warning(f"⚠️ {label}: {value:.4f}")


def plot_training_curves(history_dict, best_epoch):
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    panels = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("f1_metric", "val_f1_metric", "F1"),
        ("loss", "val_loss", "Loss"),
    ]
    for i, (train_k, val_k, title) in enumerate(panels):
        ax[i].plot(epochs, history_dict[train_k], label=f"Train {title}")
        ax[i].plot(epochs, history_dict[val_k], label=f"Val {title}")
        ax[i].axvline(x=best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
        ax[i].set_title(f"{title} over epochs")
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(title)
        ax[i].legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_and_report(y_true, y_pred):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader("📋 Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)


def plot_roc(y_true, y_prob):
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)


def best_threshold_by_f1(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0  # last P/R point has no threshold
    if len(thresholds) == 0:
        return 0.5, 0.0
    return float(thresholds[best_idx]), float(f1s[best_idx])


class StreamlitProgressCallback(Callback):
    """Cheap per-epoch UI update — no re-plotting, no full evaluation."""

    def __init__(self, progress_bar, status_text, max_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.progress_bar.progress(min((epoch + 1) / self.max_epochs, 1.0))
        self.status_text.text(
            f"Epoch {epoch + 1}/{self.max_epochs} — "
            f"loss {logs.get('loss', 0):.4f} | acc {logs.get('accuracy', 0):.4f} | "
            f"f1 {logs.get('f1_metric', 0):.4f}  ||  "
            f"val_loss {logs.get('val_loss', 0):.4f} | val_acc {logs.get('val_accuracy', 0):.4f} | "
            f"val_f1 {logs.get('val_f1_metric', 0):.4f}"
        )


# --------------------------------------------------------------------------- #
# Sidebar: hyperparameters + Gist persistence
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🧠 Model settings")
    layer1 = st.number_input("Layer 1 units", 4, 512, 64, step=4)
    layer2 = st.number_input("Layer 2 units", 4, 512, 32, step=4)
    dropout = st.slider("Dropout rate", 0.0, 0.7, 0.0, step=0.05)
    lr = st.select_slider("Learning rate", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
    max_epochs = st.number_input("Max epochs", 10, 2000, 200, step=10)
    patience = st.number_input("Early-stopping patience", 3, 100, 10, step=1)

    st.divider()
    st.header("☁️ Model persistence (GitHub Gist)")
    st.caption(
        "Streamlit Community Cloud wipes local files on every reboot/redeploy. "
        "Save the trained model to a private Gist so it survives restarts."
    )
    default_token = st.secrets.get("GITHUB_TOKEN", "") if hasattr(st, "secrets") else ""
    default_gist = st.secrets.get("GIST_ID", "") if hasattr(st, "secrets") else ""
    github_token = st.text_input("GitHub token (gist scope)", value=default_token, type="password", label_visibility="collapsed")
    gist_id_input = st.text_input("Gist ID (leave blank to create one on first save)", value=st.session_state.gist_id or default_gist, label_visibility="collapsed")
    st.session_state.gist_id = gist_id_input

    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("💾 Save to Gist", disabled=st.session_state.model is None):
            if not github_token:
                st.error("Provide a GitHub token with `gist` scope first.")
            else:
                try:
                    with st.spinner("Uploading artifacts to Gist..."):
                        new_id = save_artifacts_to_gist(github_token, st.session_state.gist_id)
                    st.session_state.gist_id = new_id
                    st.success(f"Saved. Gist ID: `{new_id}` — add it to secrets as GIST_ID for auto-reuse.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
    with col_load:
        if st.button("📥 Load from Gist", disabled=not gist_id_input):
            if not github_token:
                st.error("Provide a GitHub token with `gist` scope first.")
            else:
                try:
                    with st.spinner("Loading artifacts from Gist..."):
                        model, scaler, feature_encoders, target_encoder, meta = load_artifacts_from_gist(
                            github_token, gist_id_input
                        )
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_encoders = feature_encoders
                    st.session_state.label_encoder = target_encoder
                    st.session_state.feature_columns = meta["feature_columns"]
                    st.session_state.target_column = meta["target_column"]
                    st.session_state.threshold = meta.get("threshold", 0.5)
                    st.session_state.train_flag = True
                    st.success(f"Loaded model saved at {meta.get('saved_at_utc', 'unknown time')} (UTC).")
                except Exception as e:
                    st.error(f"Load failed: {e}")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
# st.title("🔍 Binary Classification — AML / Fraud")
# st.caption("Upload labeled data to train a model, then score new records with it.")
st.markdown(
    """
    <div id="bob-banner">
        <h1>🔍 Binary Classification — AML / Fraud</h1>
        <p>Upload labeled data to train a model, then score new records with it.s</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_train, tab_predict, tab_about = st.tabs(["🏋️ Train", "🔮 Predict", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# TRAIN TAB
# --------------------------------------------------------------------------- #
with tab_train:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file for training", type=["csv", "xlsx"])
    if st.session_state.prev_file != uploaded_file:
        st.session_state.train_flag = False
        st.session_state.prev_file = uploaded_file

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        m1, m2 = st.columns(2)
        m1.metric("Rows", len(df))
        m2.metric("Columns", len(df.columns))

        st.subheader("📄 Preview")
        st.dataframe(df.head(), use_container_width=True)

        target_column = st.selectbox("🎯 Target column (must be binary)", df.columns, index=len(df.columns) - 1)

        if df[target_column].nunique() != 2:
            st.error("❌ Target column must be binary (exactly 2 unique values).")
        else:
            portion = st.slider("🔢 Data sample percentage", 1, 100, 100, step=1)
            if portion < 100:
                df, _ = train_test_split(
                    df, train_size=portion / 100.0, random_state=42, stratify=df[target_column]
                )
                df = df.reset_index(drop=True)
                st.caption(f"Using {len(df)} sampled rows (stratified on `{target_column}`).")

            feature_columns = [c for c in df.columns if c != target_column]
            X = df[feature_columns].copy()
            y = df[target_column].copy()

            feature_encoders = {}
            for col in X.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                feature_encoders[col] = le

            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))

            if st.button("🚀 Train model", type="primary"):
                model = Sequential()
                model.add(Dense(layer1, activation="relu", input_shape=(X_train.shape[1],)))
                if dropout > 0:
                    model.add(Dropout(dropout))
                model.add(Dense(layer2, activation="relu"))
                if dropout > 0:
                    model.add(Dropout(dropout))
                model.add(Dense(1, activation="sigmoid"))
                model.compile(
                    optimizer=AdamW(learning_rate=lr, weight_decay=1e-6),
                    loss="binary_crossentropy",
                    metrics=["accuracy", f1_metric],
                )

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                early_stop = EarlyStopping(
                    monitor="val_f1_metric", mode="max", patience=patience,
                    restore_best_weights=True, verbose=0,
                )
                progress_cb = StreamlitProgressCallback(progress_bar, status_text, max_epochs)

                with st.spinner("Training..."):
                    hist = model.fit(
                        X_train, y_train,
                        epochs=max_epochs,
                        batch_size=32,
                        validation_split=0.2,
                        class_weight=class_weight_dict,
                        callbacks=[early_stop, progress_cb],
                        verbose=0,
                    )

                best_epoch = int(np.argmax(hist.history["val_f1_metric"])) + 1
                now_ist = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p")
                st.toast(f"✅ Trained. Best epoch {best_epoch}. {now_ist} IST")
                st.success(f"✅ Model trained — restored weights from best epoch ({best_epoch}).")

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_columns = feature_columns
                st.session_state.feature_encoders = feature_encoders
                st.session_state.label_encoder = target_encoder
                st.session_state.target_column = target_column
                st.session_state.history = hist.history
                st.session_state.best_epoch = best_epoch
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.train_flag = True

            if st.session_state.train_flag and st.session_state.model is not None:
                model = st.session_state.model
                X_test, y_test = st.session_state.X_test, st.session_state.y_test
                history, best_epoch = st.session_state.history, st.session_state.best_epoch

                loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose=0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    status_metric("Accuracy", accuracy, good=0.75, bad=0.50)
                with c2:
                    status_metric("Loss", loss, good=0.25, bad=0.75, higher_is_better=False)
                with c3:
                    status_metric("F1-score", f1, good=0.75, bad=0.50)

                st.subheader("📊 Training curves")
                plot_training_curves(history, best_epoch)

                y_prob = model.predict(X_test, verbose=0).flatten()
                best_thr, best_thr_f1 = best_threshold_by_f1(y_test, y_prob)
                st.info(f"Best F1-maximizing threshold on held-out test set: **{best_thr:.2f}** (F1={best_thr_f1:.4f})")

                threshold = st.slider("🎯 Prediction threshold", 0.0, 1.0, float(best_thr), 0.01)
                st.session_state.threshold = threshold
                y_pred = (y_prob > threshold).astype(int)

                c1, c2 = st.columns(2)
                with c1:
                    status_metric("Accuracy @ threshold", accuracy_score(y_test, y_pred), good=0.75, bad=0.50)
                with c2:
                    status_metric("F1 @ threshold", f1_score(y_test, y_pred), good=0.75, bad=0.50)

                plot_confusion_and_report(y_test, y_pred)
                plot_roc(y_test, y_prob)
    else:
        st.session_state.train_flag = False
        st.info("Upload a training file to get started.")


# --------------------------------------------------------------------------- #
# PREDICT TAB
# --------------------------------------------------------------------------- #
with tab_predict:
    if not st.session_state.train_flag or st.session_state.model is None:
        st.warning("⚠️ No trained model available yet — train one in the **Train** tab, or load one from a Gist in the sidebar.")
    else:
        st.caption(
            f"Active model expects target `{st.session_state.target_column}` and "
            f"{len(st.session_state.feature_columns)} feature columns."
        )
        infer_file = st.file_uploader("📂 Upload CSV/Excel for prediction", type=["csv", "xlsx"], key="infer")

        if infer_file:
            infer_df = pd.read_csv(infer_file) if infer_file.name.endswith(".csv") else pd.read_excel(infer_file)

            missing = [c for c in st.session_state.feature_columns if c not in infer_df.columns]
            if missing:
                st.error(f"❌ Uploaded file is missing required feature column(s): {missing}")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Rows", len(infer_df))
                m2.metric("Columns", len(infer_df.columns))
                st.subheader("📄 Preview")
                st.dataframe(infer_df.head(), use_container_width=True)

                try:
                    work_df = infer_df.copy()
                    unseen_counts = {}
                    for col, le in st.session_state.feature_encoders.items():
                        if col in work_df.columns:
                            str_vals = work_df[col].astype(str)
                            known = set(le.classes_)
                            unseen_mask = ~str_vals.isin(known)
                            unseen_counts[col] = int(unseen_mask.sum())
                            # unseen categories -> a sentinel value so transform never raises
                            safe_vals = str_vals.where(~unseen_mask, le.classes_[0])
                            work_df[col] = le.transform(safe_vals)

                    if any(v > 0 for v in unseen_counts.values()):
                        details = ", ".join(f"{c}: {v}" for c, v in unseen_counts.items() if v > 0)
                        st.warning(f"⚠️ Rows with categories unseen during training were mapped to a placeholder ({details}).")

                    threshold = st.session_state.threshold
                    infer_X = st.session_state.scaler.transform(work_df[st.session_state.feature_columns])
                    infer_probs = st.session_state.model.predict(infer_X, verbose=0).flatten()
                    infer_preds = (infer_probs > threshold).astype(int)
                    infer_labels = st.session_state.label_encoder.inverse_transform(infer_preds)

                    result_df = infer_df.copy()
                    result_df["Prediction"] = infer_labels
                    result_df["Prediction_Probability"] = infer_probs

                    st.subheader("📈 Predictions")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=result_df.to_csv(index=False).encode(),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    target_col = st.session_state.target_column
                    if target_col in infer_df.columns:
                        true_labels = st.session_state.label_encoder.transform(infer_df[target_col].astype(str))
                        c1, c2 = st.columns(2)
                        with c1:
                            status_metric("Accuracy", accuracy_score(true_labels, infer_preds), good=0.75, bad=0.50)
                        with c2:
                            status_metric("F1-score", f1_score(true_labels, infer_preds), good=0.75, bad=0.50)
                        plot_confusion_and_report(true_labels, infer_preds)

                except Exception as e:
                    st.error(f"❌ Error during inference: {e}")


# --------------------------------------------------------------------------- #
# ABOUT TAB
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
### How it works
1. **Train** — upload a labeled dataset, pick the binary target column, and train a small
   feed-forward network. Categorical features are label-encoded and numeric features are
   standardized; both transforms are saved alongside the model so inference stays consistent.
2. **Predict** — upload new data (same feature columns) and the app scores it with the
   currently active model — either the one you just trained, or one loaded from a Gist.
3. **Persistence** — Streamlit Community Cloud's filesystem is ephemeral, so anything trained
   in one session disappears on the next reboot/redeploy. Use the sidebar to push the trained
   model, scaler and encoders to a private GitHub Gist, and to pull them back later.

### One-time setup for Gist persistence
1. Create a GitHub [personal access token](https://github.com/settings/tokens) with the
   **gist** scope only.
2. Add it to your app's secrets (`.streamlit/secrets.toml` locally, or the "Secrets" panel on
   Streamlit Community Cloud):
   ```toml
   GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
   GIST_ID = ""   # leave blank until you've saved once, then paste the id shown
   ```
3. Train a model, click **💾 Save to Gist**, copy the returned Gist ID into `GIST_ID` in your
   secrets so future sessions auto-fill it. Click **📥 Load from Gist** any time to restore.

Never make the token or Gist public if your data or model could reveal sensitive information —
the app always creates/updates the Gist as **secret**, but secret gists are still not the same
as private-in-an-org storage.
        """
    )
