import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from datetime import datetime
import pytz
import math
import random
import string

from keras import backend as K

def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')  # Threshold instead of round

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1



st.set_page_config(page_title="Binary Classifier", layout="wide")
col1, col_divider, col2 = st.columns([0.8, 1.5, 0.8])
with col_divider:
    st.title("🔍 Binary Classification (for AML / Fraud)")

st.write("")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'train_flag' not in st.session_state:
    st.session_state.train_flag = False
if "prev_file" not in st.session_state:
    st.session_state.prev_file = None
if "history" not in st.session_state:
    st.session_state.history = None
if "best_epoch" not in st.session_state:
    st.session_state.best_epoch = None


# print(f"Session Train Flag: {st.session_state.train_flag}")

# Add this near the top where layout begins

col1, col_divider, col2 = st.columns([1, 0.05, 1])


with col1:
    # Upload training data
    st.subheader("🏋️ Upload Data for Training")
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file for Training", type=["csv", "xlsx"])
    if st.session_state.prev_file != uploaded_file:
        st.session_state.train_flag = False
        st.session_state.prev_file = uploaded_file

    if uploaded_file:
        # st.session_state.train_flag = False
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.markdown(
            f"""
            <div style="
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: #333;
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
                ">
                <span style='color:#aaaa88'>Training Data:</span>
                Rows: <span style='color:#2E8B57'>{len(df)}</span>, 
                Columns: <span style='color:#1E90FF'>{len(df.columns)}</span>
            </div><br/>
            """,
            unsafe_allow_html=True
        )
        portion = st.slider("🔢 Choose Data Sample Percentage:", 1, 100, value=100, step=1)
        portion_decimal = portion / 100.0
        if portion_decimal < 1:
            temp_df, _ = train_test_split(df, test_size=(1 - portion_decimal), random_state=42, stratify=df[df.columns[-1]])
            df = temp_df.reset_index(drop=True)
            temp_df = None

        st.markdown(
            f"""
            <div style="
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: #333;
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
                ">
                <span style='color:#aa88aa'>Final Training Data:</span>
                Final Rows: <span style='color:#2E8B57'>{len(df)}</span>, 
                Final Columns: <span style='color:#1E90FF'>{len(df.columns)}</span>
            </div><br/>
            """,
            unsafe_allow_html=True
        )
        st.subheader("📄 Preview of Uploaded Data")
        df.index += 1
        st.dataframe(df.head())

        # Select target column
        target_column = st.selectbox("🎯 Select Target Column", df.columns, index=len(df.columns)-1)

        if target_column:
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].copy()
            y = df[target_column].copy()

            if len(np.unique(y)) != 2:
                st.error("❌ Target column must be binary (contain exactly 2 unique values).")
            else:
                # Label encode non-numeric features
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

                # Label encode target
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                st.session_state.label_encoder = le_target

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler
                st.session_state.feature_columns = feature_columns

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                # Compute class weights
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = dict(enumerate(class_weights))

                

                if st.button("🚀 Train Model"):

                    st.session_state.best_epoch = None

                    # Build model
                    model = Sequential([
                        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-6), loss='binary_crossentropy', metrics=['accuracy', f1_metric])

                    ck = ModelCheckpoint("best_model.keras", monitor="val_f1_metric", save_best_only=True, verbose=0, mode="max")

                    es = 10

                    stop_epoch = 0

                    with st.spinner("🚀 Training in progress..."):

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        int_placeholder = st.empty()

                        best_epoch = 0
                        best_f1 = float("-inf")

                        history = {"loss": [], "accuracy": [], "f1_metric": [],  "val_loss": [], "val_accuracy": [], "val_f1_metric": []}
                        epochs = 10000
                        for epoch in range(epochs):
                            hist = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=0, class_weight=class_weight_dict, callbacks=[ck])
                            
                            # Collect metrics
                            history["loss"].append(hist.history["loss"][0])
                            history["accuracy"].append(hist.history["accuracy"][0])
                            history["f1_metric"].append(hist.history["f1_metric"][0])
                            history["val_loss"].append(hist.history["val_loss"][0])
                            history["val_accuracy"].append(hist.history["val_accuracy"][0])
                            history["val_f1_metric"].append(hist.history["val_f1_metric"][0])
                            
                            if best_f1 < hist.history["val_f1_metric"][0]:
                                best_f1 = hist.history["val_f1_metric"][0]
                                best_epoch = epoch + 1
                                es = 10
                            else:
                                es -= 1

                            curr_epoch = (math.ceil((epoch + 1) / 10)) * 10

                            # Update progress bar and status
                            progress_bar.progress((epoch + 1) / curr_epoch)
                            status_text.text(f"Epoch {epoch + 1}/{curr_epoch} - Loss: {hist.history['loss'][0]:.4f}, Accuracy: {hist.history['accuracy'][0]:.4f}, F1: {hist.history['f1_metric'][0]:.4f}\nVal Loss: {hist.history['val_loss'][0]:.4f}, Val Accuracy: {hist.history['val_accuracy'][0]:.4f}, Val F1: {hist.history['val_f1_metric'][0]:.4f}\nBest Epoch: {best_epoch}, Validation F1: {best_f1:.4f}")

                            

                            with int_placeholder.container():
                                # Evaluation
                                int_loss, int_accuracy, int_f1 = model.evaluate(X_test, y_test, verbose=0)

                                if int_accuracy >= 0.75:
                                    st.success(f"✅ Intermediate Validation Accuracy: {int_accuracy:.4f}")
                                elif int_accuracy < 0.50:
                                    st.error(f"❌ Intermediate Validation Accuracy: {int_accuracy:.4f}")
                                else:
                                    st.warning(f"⚠️ Intermediate Validation Accuracy: {int_accuracy:.4f}")

                                if int_loss <= 0.25:
                                    st.success(f"✅ Intermediate Validation Loss: {int_loss:.4f}")
                                elif int_loss > 0.75:
                                    st.error(f"❌ Intermediate Validation Loss: {int_loss:.4f}")
                                else:
                                    st.warning(f"⚠️ Intermediate Validation Loss: {int_loss:.4f}")

                                if int_f1 >= 0.75:
                                    st.success(f"✅ Intermediate Validation F1-Score: {int_f1:.4f}")
                                elif int_f1 < 0.50:
                                    st.error(f"❌ Intermediate Validation F1-Score: {int_f1:.4f}")
                                else:
                                    st.warning(f"⚠️ Intermediate Validation F1-Score: {int_f1:.4f}")

                                # Training Curves
                                st.subheader(f"📊 Intermediate Training Curves (Epoch: {epoch + 1})")
                                int_epochs = range(1, len(history["accuracy"]) + 1)
                                int_fig, int_ax = plt.subplots(1, 3, figsize=(16, 4))

                                # Compute difference between last train and val accuracy
                                diff_acc = - history["accuracy"][-1] + history["val_accuracy"][-1]
                                diff_loss = history["loss"][-1] - history["val_loss"][-1]
                                diff_f1 = - history["f1_metric"][-1] + history["val_f1_metric"][-1]

                                # Accuracy plot
                                int_ax[0].plot(int_epochs, history["accuracy"], label='Train Accuracy')
                                int_ax[0].plot(int_epochs, history["val_accuracy"], label='Validation Accuracy')
                                int_ax[0].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                                int_ax[0].axhline(y=max(history["accuracy"]), linestyle='dotted', color='teal')
                                int_ax[0].axhline(y=max(history["val_accuracy"]), linestyle='dotted', color='orange')
                                int_ax[0].set_title('Intermediate Accuracy over Epochs')
                                int_ax[0].set_xlabel('Intermediate Epoch')
                                int_ax[0].set_ylabel('Intermediate Accuracy')
                                # int_ax[0].legend()
                                int_ax[0].legend(title=f"Acc Diff: {diff_acc:.4f}")


                                # F1 plot
                                int_ax[1].plot(int_epochs, history["f1_metric"], label='Train F1')
                                int_ax[1].plot(int_epochs, history["val_f1_metric"], label='Validation F1')
                                int_ax[1].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                                int_ax[1].axhline(y=max(history["f1_metric"]), linestyle='dotted', color='teal')
                                int_ax[1].axhline(y=max(history["val_f1_metric"]), linestyle='dotted', color='orange')
                                int_ax[1].set_title('Intermediate F1 over Epochs')
                                int_ax[1].set_xlabel('Intermediate Epoch')
                                int_ax[1].set_ylabel('Intermediate F1')
                                # int_ax[1].legend()
                                int_ax[1].legend(title=f"F1 Diff: {diff_f1:.4f}")

                                # Loss plot
                                int_ax[2].plot(int_epochs, history["loss"], label='Train Loss')
                                int_ax[2].plot(int_epochs, history["val_loss"], label='Validation Loss')
                                int_ax[2].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                                int_ax[2].axhline(y=min(history["loss"]), linestyle='dotted', color='teal')
                                int_ax[2].axhline(y=min(history["val_loss"]), linestyle='dotted', color='orange')
                                int_ax[2].set_title('Intermediate Loss over Epochs')
                                int_ax[2].set_xlabel('Intermediate Epoch')
                                int_ax[2].set_ylabel('Intermediate Loss')
                                # int_ax[2].legend()
                                int_ax[2].legend(title=f"Loss Diff: {diff_loss:.4f}")

                                st.pyplot(int_fig)

                                # Threshold Optimization
                                int_y_prob = model.predict(X_test).flatten()
                                int_thres = 0
                                int_best_f1 = 0
                                int_progress_bar = st.progress(0)
                                with st.spinner("🚀 Intermediate Optimization in progress..."):
                                    for int_epoch in range(101):
                                        int_y_pred = (int_y_prob > (int_epoch / 100.0)).astype("int32")
                                        int_f1 = f1_score(y_test, int_y_pred)
                                        if int_f1 > int_best_f1:
                                            int_best_f1 = int_f1
                                            int_thres = int_epoch / 100.0
                                        int_progress_bar.progress(int_epoch / 100)
                                int_progress_bar.empty()

                                def int_generate_code(length=16):
                                    characters = string.ascii_uppercase + string.digits + string.ascii_lowercase
                                    return ''.join(random.choices(characters, k=length))

                                st.warning(f"⚠️ Intermediate Best Threshold: {int_thres:.2f}")
                                int_threshold = st.slider("🎯 Intermediate Prediction Threshold", min_value=0.0, max_value=1.0, value=int_thres, step=0.01, key=int_generate_code())
                                st.session_state.threshold = int_threshold
                                int_y_pred = (int_y_prob > int_threshold).astype("int32")

                                int_acc = accuracy_score(y_test, int_y_pred)
                                int_f1 = f1_score(y_test, int_y_pred)

                                if int_acc >= 0.75:
                                    st.success(f"✅ Intermediate Accuracy: {int_acc:.4f}")
                                elif int_acc < 0.50:
                                    st.error(f"❌ Intermediate Accuracy: {int_acc:.4f}")
                                else:
                                    st.warning(f"⚠️ Intermediate Accuracy: {int_acc:.4f}")

                                if int_f1 >= 0.75:
                                    st.success(f"✅ Intermediate F1-Score: {int_f1:.4f}")
                                elif int_f1 < 0.50:
                                    st.error(f"❌ Intermediate F1-Score: {int_f1:.4f}")
                                else:
                                    st.warning(f"⚠️ Intermediate F1-Score: {int_f1:.4f}")

                                # Confusion Matrix
                                st.subheader("📌 Intermediate Confusion Matrix")
                                int_cm = confusion_matrix(y_test, int_y_pred)
                                int_fig_cm, int_ax_cm = plt.subplots()
                                sns.heatmap(int_cm, annot=True, fmt='d', cmap='Blues', ax=int_ax_cm)
                                int_ax_cm.set_xlabel('Intermediate Predicted')
                                int_ax_cm.set_ylabel('Intermediate Actual')
                                st.pyplot(int_fig_cm)

                                # Classification Report
                                st.subheader("📋 Intermediate Classification Report")
                                int_report = classification_report(y_test, int_y_pred, output_dict=True)
                                int_report_df = pd.DataFrame(int_report).transpose()
                                int_fig_cr, int_ax_cr = plt.subplots(figsize=(10, 4))
                                sns.heatmap(int_report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt=".4f", ax=int_ax_cr)
                                int_ax_cr.set_title('Intermediate Classification Report')
                                st.pyplot(int_fig_cr)

                                # ROC Curve
                                st.subheader("📈 Intermediate ROC Curve")
                                int_fpr, int_tpr, _ = roc_curve(y_test, int_y_prob)
                                int_roc_auc = auc(int_fpr, int_tpr)
                                int_fig_roc, int_ax_roc = plt.subplots()
                                int_ax_roc.plot(int_fpr, int_tpr, label=f'AUC = {int_roc_auc:.2f}')
                                int_ax_roc.plot([0, 1], [0, 1], 'k--')
                                int_ax_roc.set_xlabel('Intermediate False Positive Rate')
                                int_ax_roc.set_ylabel('Intermediate True Positive Rate')
                                int_ax_roc.set_title('Intermediate ROC Curve')
                                int_ax_roc.legend(loc='lower right')
                                st.pyplot(int_fig_roc)


                            if es == 0:
                                st.error(f"🔴 Early stopping triggered at epoch {epoch + 1}")
                                break
                                
                        st.success("✅ Model trained successfully!")

                        int_placeholder.empty()

                        st.success(f"🟢 Restoring best weights from epoch {best_epoch}")

                        st.session_state.best_epoch = best_epoch

                        # Define IST timezone
                        ist = pytz.timezone('Asia/Kolkata')

                        # Get current time in IST
                        current_time_ist = datetime.now(ist)

                        # Print formatted time
                        st.toast(f"✅ Model trained successfully at: {current_time_ist.strftime('%Y-%m-%d %I:%M:%S %p')}")

                        st.session_state.model = model

                        st.session_state.train_flag = True

                        st.session_state.history = history
        
                    


                    
                # Prediction section (only if model is trained)
                if ("model" in st.session_state) and (st.session_state.train_flag == True):
                    model = st.session_state.model
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test

                    history = st.session_state.history

                    best_epoch = st.session_state.best_epoch

                    # Evaluation
                    loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose=0)
                    if accuracy >= 0.75:
                        st.success(f"✅ Validation Accuracy: {accuracy:.4f}")
                    elif accuracy < 0.50:
                        st.error(f"❌ Validation Accuracy: {accuracy:.4f}")
                    else:
                        st.warning(f"⚠️ Validation Accuracy: {accuracy:.4f}")
                    if loss <= 0.25:
                        st.success(f"✅ Validation Loss: {loss:.4f}")
                    elif loss > 0.75:
                        st.error(f"❌ Validation Loss: {loss:.4f}")
                    else:
                        st.warning(f"⚠️ Validation Loss: {loss:.4f}")
                    if f1 >= 0.75:
                        st.success(f"✅ Validation F1-Score: {f1:.4f}")
                    elif f1 < 0.50:
                        st.error(f"❌ Validation F1-Score: {f1:.4f}")
                    else:
                        st.warning(f"⚠️ Validation F1-Score: {f1:.4f}")


                    # Generate epoch range starting from 1
                    epochs = range(1, len(history["accuracy"]) + 1)

                    # Streamlit section
                    st.subheader("📊 Training Curves")
                    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

                    # Accuracy plot
                    ax[0].plot(epochs, history["accuracy"], label='Train Accuracy')
                    ax[0].plot(epochs, history["val_accuracy"], label='Validation Accuracy')
                    ax[0].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                    ax[0].axhline(y=max(history["accuracy"]), linestyle='dotted', color='teal')
                    ax[0].axhline(y=max(history["val_accuracy"]), linestyle='dotted', color='orange')
                    ax[0].set_title('Accuracy over Epochs')
                    ax[0].set_xlabel('Epoch')
                    ax[0].set_ylabel('Accuracy')
                    ax[0].legend()

                    # F1 plot
                    ax[1].plot(epochs, history["f1_metric"], label='Train F1')
                    ax[1].plot(epochs, history["val_f1_metric"], label='Validation F1')
                    ax[1].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                    ax[1].axhline(y=max(history["f1_metric"]), linestyle='dotted', color='teal')
                    ax[1].axhline(y=max(history["val_f1_metric"]), linestyle='dotted', color='orange')
                    ax[1].set_title('F1 over Epochs')
                    ax[1].set_xlabel('Epoch')
                    ax[1].set_ylabel('F1')
                    ax[1].legend()

                    # Loss plot
                    ax[2].plot(epochs, history["loss"], label='Train Loss')
                    ax[2].plot(epochs, history["val_loss"], label='Validation Loss')
                    ax[2].axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch: {best_epoch}')
                    ax[2].axhline(y=min(history["loss"]), linestyle='dotted', color='teal')
                    ax[2].axhline(y=min(history["val_loss"]), linestyle='dotted', color='orange')
                    ax[2].set_title('Loss over Epochs')
                    ax[2].set_xlabel('Epoch')
                    ax[2].set_ylabel('Loss')
                    ax[2].legend()

                    # Show plot in Streamlit
                    st.pyplot(fig)

                    # plt.close()


                    y_prob = model.predict(X_test).flatten()
                    
                    thres = 0
                    best_f1 = 0
                    progress_bar = st.progress(0)
                    with st.spinner("🚀 Optimization in progress..."):            
                        status_text = st.empty()
                        for epoch in range(101):
                            y_pred = (y_prob > (epoch/100.0)).astype("int32")
                            f1 = f1_score(y_test, y_pred)
                            if f1 > best_f1:
                                best_f1 = f1
                                thres = epoch/100.0
                            progress_bar.progress((epoch) / 100)

                    progress_bar.empty()

                    st.warning(f"⚠️ Best Threshold: {thres:.2f}")
                            
                    threshold = st.slider("🎯 Prediction Threshold", min_value=0.0, max_value=1.0, value=thres, step=0.01)
                    st.session_state.threshold = threshold
                    y_pred = (y_prob > threshold).astype("int32")

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    if acc >= 0.75:
                        st.success(f"✅ Accuracy: {acc:.4f}")
                    elif acc < 0.50:
                        st.error(f"❌ Accuracy: {acc:.4f}")
                    else:
                        st.warning(f"⚠️ Accuracy: {acc:.4f}")
                    if f1 >= 0.75:
                        st.success(f"✅ F1-Score: {f1:.4f}")
                    elif f1 < 0.50:
                        st.error(f"❌ F1-Score: {f1:.4f}")
                    else:
                        st.warning(f"⚠️ F1-Score: {f1:.4f}")

                    # Confusion Matrix
                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    st.pyplot(fig_cm)

                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    fig_cr, ax_cr = plt.subplots(figsize=(10, 4))
                    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt=".4f", ax=ax_cr)
                    ax_cr.set_title('Classification Report')
                    st.pyplot(fig_cr)

                    # ROC Curve
                    st.subheader("📈 ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                    ax_roc.plot([0, 1], [0, 1], 'k--')
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('ROC Curve')
                    ax_roc.legend(loc='lower right')
                    st.pyplot(fig_roc)

    else:
        st.session_state.train_flag = False


with col_divider:
    border_choices = ["solid", "dashed", "dotted"]
    st.markdown(f"<div style='border: 1px {np.random.choice(border_choices)} #000088; height: 4000px; width: 1px'></div>", unsafe_allow_html=True)


with col2:

    infer_file = None

    if st.session_state.train_flag == True:
        # Inference section
        st.subheader("🔍 Upload New Data for Inference")
        infer_file = st.file_uploader("📂 Upload CSV/Excel for Prediction", type=["csv", "xlsx"], key="infer")

    if infer_file and uploaded_file:
        if "model" not in st.session_state or st.session_state.train_flag == False:
            st.warning("⚠️ Please train the model first before uploading inference data.")
        else:
            infer_df = pd.read_csv(infer_file) if infer_file.name.endswith(".csv") else pd.read_excel(infer_file)
            st.markdown(
                f"""
                <div style="
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    text-align: center;
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                    box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
                    ">
                    <span style='color:#aa8888'>Inference Data: </span>
                    Rows: <span style='color:#2E8B57'>{len(infer_df)}</span>, 
                    Columns: <span style='color:#1E90FF'>{len(infer_df.columns)}</span>
                </div><br/>
                """,
                unsafe_allow_html=True
            )
            st.subheader("📄 Preview of Inference Data")
            infer_df.index += 1
            st.dataframe(infer_df.head())

            try:
                # Reuse training encoders
                label_encoders = {}

                for col in infer_df.select_dtypes(include=['object', 'category']).columns:
                    if col in st.session_state.feature_columns:
                        le = LabelEncoder()
                        infer_df[col] = le.fit_transform(infer_df[col].astype(str))
                        label_encoders[col] = le  # Save encoder for decoding


                # Ensure threshold is set
                threshold = st.session_state.get("threshold", 0.5)

                # Prepare features
                infer_X = st.session_state.scaler.transform(infer_df[st.session_state.feature_columns])
                infer_probs = st.session_state.model.predict(infer_X).flatten()
                infer_preds = (infer_probs > threshold).astype("int32")
                infer_labels = st.session_state.label_encoder.inverse_transform(infer_preds)

                infer_df["Prediction"] = infer_labels
                st.subheader("📈 Predictions")

                decoded_df = infer_df.copy()

                for col, le in label_encoders.items():
                    decoded_df[col] = le.inverse_transform(decoded_df[col])

                # decoded_df.index += 1
                st.dataframe(decoded_df)


                # Evaluation (only if target column exists)
                if target_column in infer_df.columns:
                    true_labels = st.session_state.label_encoder.transform(infer_df[target_column])
                    acc = accuracy_score(true_labels, infer_preds)
                    f1 = f1_score(true_labels, infer_preds)
                    if acc >= 0.75:
                        st.success(f"✅ Accuracy: {acc:.4f}")
                    elif acc < 0.50:
                        st.error(f"❌ Accuracy: {acc:.4f}")
                    else:
                        st.warning(f"⚠️ Accuracy: {acc:.4f}")
                    if f1 >= 0.75:
                        st.success(f"✅ F1-Score: {f1:.4f}")
                    elif f1 < 0.50:
                        st.error(f"❌ F1-Score: {f1:.4f}")
                    else:
                        st.warning(f"⚠️ F1-Score: {f1:.4f}")

                

                # Evaluation (only if target column exists)
                if target_column in infer_df.columns:
                    # Confusion Matrix
                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(true_labels, infer_preds)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    st.pyplot(fig_cm)

                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(true_labels, infer_preds, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    fig_cr, ax_cr = plt.subplots(figsize=(10, 4))
                    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt=".4f", ax=ax_cr)
                    ax_cr.set_title('Classification Report')
                    st.pyplot(fig_cr)

            except Exception as e:
                st.error(f"❌ Error during inference: {e}")

    elif infer_file:
        st.warning("⚠️ Please train the model first before uploading inference data.")
