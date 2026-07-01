import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

st.title(":material/analytics: Evaluasi Model")
st.caption("Performa model Random Forest multi-output — Kategori Pemahaman & Tingkat ASD")

model_ready = st.session_state.get("model_ready", False)
model_ai = st.session_state.get("model", None)

DATA_PATH = "dataset/dataset.csv"

@st.cache_data
def load_dataset(_mtime):
    df = pd.read_csv(DATA_PATH, sep=";")
    df.fillna("Tidak Diketahui", inplace=True)
    return df

@st.cache_resource
def build_pipeline():
    fitur_teks = 'Ujaran Bersih'
    fitur_kategorikal = ['Echolalia', 'Struktur Sintaksis', 'Kompleksitas Kalimat', 'Intensi Komunikasi']
    fitur_numerik = ['MLU']

    preprocessor = ColumnTransformer(transformers=[
        ('teks', TfidfVectorizer(max_features=100), fitur_teks),
        ('kategori', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal),
        ('angka', StandardScaler(), fitur_numerik)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    return pipeline

@st.cache_data
def prepare_data(_mtime):
    df = load_dataset(_mtime)
    fitur_teks = 'Ujaran Bersih'
    fitur_kategorikal = ['Echolalia', 'Struktur Sintaksis', 'Kompleksitas Kalimat', 'Intensi Komunikasi']
    fitur_numerik = ['MLU']
    X = df[[fitur_teks] + fitur_kategorikal + fitur_numerik]
    y = df[['Kategori Pemahaman', 'ASD']]
    groups = df['ID Anak'].values
    y_combined = df['Kategori Pemahaman'].astype(str) + '_' + df['ASD'].astype(str)
    return X, y, groups, y_combined

if not model_ready:
    st.warning("Model tidak ditemukan. Evaluasi tidak dapat dilakukan.")
    st.stop()

mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0
pipeline = build_pipeline()
X, y, groups, y_combined = prepare_data(mtime)

st.info(f"**Model:** Random Forest (100 estimators, multi-output)  |  **Target:** Kategori Pemahaman + ASD  |  **Dataset:** {len(X)} sampel")

y_pemahaman = y['Kategori Pemahaman']
y_asd = y['ASD']

Y_PEMAHAMAN_LABEL = "Kategori Pemahaman"
Y_ASD_LABEL = "Tingkat ASD"

tab_overview, tab_pemahaman, tab_asd, tab_cv, tab_feat = st.tabs([
    "Ringkasan",
    "Conf. Matrix Pemahaman",
    "Conf. Matrix ASD",
    "Cross-Validation",
    "Feature Importance"
])

target_names_p = sorted(y_pemahaman.unique())
target_names_a = sorted(y_asd.unique())

# Full dataset predictions (for CM and report)
with st.spinner("Menjalankan prediksi pada seluruh dataset..."):
    y_pred_full = model_ai.predict(X)
y_pred_p = y_pred_full[:, 0]
y_pred_a = y_pred_full[:, 1]

# Cross-validation predictions
with st.spinner("Menjalankan 5-Fold StratifiedGroupKFold..."):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv_full = np.empty((len(X), 2), dtype=object)
    acc_per_fold_p = []
    acc_per_fold_a = []

    for train_idx, test_idx in cv.split(X, y_combined, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        model = clone(pipeline)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        y_pred_cv_full[test_idx] = preds
        acc_per_fold_p.append(accuracy_score(y.iloc[test_idx, 0], preds[:, 0]))
        acc_per_fold_a.append(accuracy_score(y.iloc[test_idx, 1], preds[:, 1]))

    y_pred_cv_p = y_pred_cv_full[:, 0]
    y_pred_cv_a = y_pred_cv_full[:, 1]

with tab_overview:
    st.subheader("Ringkasan Performa (5-Fold StratifiedGroupKFold)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi Pemahaman", f"{np.mean(acc_per_fold_p)*100:.2f}%")
    col2.metric("F1 Pemahaman", f"{f1_score(y_pemahaman, y_pred_cv_p, average='macro')*100:.2f}%")
    col3.metric("Akurasi ASD", f"{np.mean(acc_per_fold_a)*100:.2f}%")
    col4.metric("F1 ASD", f"{f1_score(y_asd, y_pred_cv_a, average='macro')*100:.2f}%")

    st.markdown("""
    **Catatan:**
    - Training menggunakan **5-Fold StratifiedGroupKFold** — ujaran dari anak yang sama tidak terpisah antar fold
    - **ASD tidak digunakan sebagai fitur** — model belajar murni dari pola linguistik
    - Multi-output: 1 model memprediksi 2 target sekaligus
    """)

def plot_confusion_matrix(y_true, y_pred, target_names, title, ax):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=target_names,
        cmap='Blues', ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(title, pad=20, fontsize=14)

with tab_pemahaman:
    st.subheader("Confusion Matrix — Kategori Pemahaman (Full Dataset)")

    acc_full_p = accuracy_score(y_pemahaman, y_pred_p)
    st.metric("Akurasi (Full Dataset)", f"{acc_full_p:.2%}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(y_pemahaman, y_pred_p, target_names_p,
                          "Confusion Matrix — Kategori Pemahaman", ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Classification Report — Kategori Pemahaman**")
    report_p = classification_report(y_pemahaman, y_pred_p, target_names=target_names_p, output_dict=True)
    st.dataframe(
        pd.DataFrame(report_p).transpose(),
        column_config={
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1-score": st.column_config.NumberColumn("F1-Score", format="%.3f"),
            "support": st.column_config.NumberColumn("Support", format="%d"),
        },
    )

with tab_asd:
    st.subheader("Confusion Matrix — Tingkat ASD (Full Dataset)")

    acc_full_a = accuracy_score(y_asd, y_pred_a)
    st.metric("Akurasi (Full Dataset)", f"{acc_full_a:.2%}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(y_asd, y_pred_a, target_names_a,
                          "Confusion Matrix — Tingkat ASD", ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Classification Report — Tingkat ASD**")
    report_a = classification_report(y_asd, y_pred_a, target_names=target_names_a, output_dict=True)
    st.dataframe(
        pd.DataFrame(report_a).transpose(),
        column_config={
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1-score": st.column_config.NumberColumn("F1-Score", format="%.3f"),
            "support": st.column_config.NumberColumn("Support", format="%d"),
        },
    )

with tab_cv:
    st.subheader("5-Fold StratifiedGroupKFold")

    col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
    col_cv1.metric("Rata2 Akurasi Pemahaman", f"{np.mean(acc_per_fold_p)*100:.2f}%")
    col_cv2.metric("Std Dev Pemahaman", f"{np.std(acc_per_fold_p)*100:.2f}%")
    col_cv3.metric("Rata2 Akurasi ASD", f"{np.mean(acc_per_fold_a)*100:.2f}%")
    col_cv4.metric("Std Dev ASD", f"{np.std(acc_per_fold_a)*100:.2f}%")

    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Akurasi Pemahaman": [f"{v:.0%}" for v in acc_per_fold_p],
        "Akurasi ASD": [f"{v:.0%}" for v in acc_per_fold_a],
    })
    st.dataframe(cv_df, hide_index=True)

    st.markdown("**Confusion Matrix — Pemahaman (Cross-Validation)**")
    fig_cv_p, ax_cv_p = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(y_pemahaman, y_pred_cv_p, target_names_p,
                          "Confusion Matrix — Pemahaman (5-Fold CV)", ax_cv_p)
    plt.tight_layout()
    st.pyplot(fig_cv_p)
    plt.close()

    st.markdown("**Confusion Matrix — ASD (Cross-Validation)**")
    fig_cv_a, ax_cv_a = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(y_asd, y_pred_cv_a, target_names_a,
                          "Confusion Matrix — ASD (5-Fold CV)", ax_cv_a)
    plt.tight_layout()
    st.pyplot(fig_cv_a)
    plt.close()

with tab_feat:
    st.subheader("Feature Importance (Random Forest)")

    with st.spinner("Mengekstrak feature importance..."):
        try:
            importances = model_ai.named_steps['classifier'].feature_importances_
            feature_names = model_ai.named_steps['preprocessor'].get_feature_names_out()

            feat_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(20)

            st.markdown("**Top 20 Fitur Paling Penting**")
            st.dataframe(
                feat_df,
                hide_index=True,
                column_config={
                    "Fitur": st.column_config.TextColumn("Fitur", width="medium"),
                    "Importance": st.column_config.NumberColumn("Importance", format="%.4f"),
                },
            )

            fig_f, ax_f = plt.subplots(figsize=(10, 6))
            ax_f.barh(range(len(feat_df)), feat_df["Importance"].values[::-1])
            ax_f.set_yticks(range(len(feat_df)))
            ax_f.set_yticklabels(feat_df["Fitur"].values[::-1])
            ax_f.set_xlabel("Importance")
            ax_f.set_title("Top 20 Feature Importance", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_f)
            plt.close()
        except Exception as e:
            st.error(f"Tidak dapat mengekstrak feature importance: {e}")
            st.info("Feature importance hanya tersedia untuk model berbasis pohon (Random Forest).")
