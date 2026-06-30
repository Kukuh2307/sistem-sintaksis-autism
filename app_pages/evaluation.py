import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

st.title(":material/analytics: Evaluasi Model")
st.caption("Performa model Random Forest pada klasifikasi Kategori Pemahaman")

model_ready = st.session_state.get("model_ready", False)
model_ai = st.session_state.get("model", None)

DATA_PATH = "dataset/dataset.csv"

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.fillna("Tidak Diketahui", inplace=True)
    return df

@st.cache_resource
def train_pipeline():
    df = load_dataset()
    fitur_teks = 'Ujaran Bersih'
    fitur_kategorikal = ['ASD', 'Echolalia', 'Struktur Sintaksis', 'Kompleksitas Kalimat', 'Intensi Komunikasi']
    fitur_numerik = ['MLU']
    X = df[[fitur_teks] + fitur_kategorikal + fitur_numerik]
    y = df['Kategori Pemahaman']
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    preprocessor = ColumnTransformer(transformers=[
        ('teks', TfidfVectorizer(max_features=100), fitur_teks),
        ('kategori', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal),
        ('angka', StandardScaler(), fitur_numerik)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    return pipeline, X, y

if not model_ready:
    st.warning("Model tidak ditemukan. Evaluasi tidak dapat dilakukan.")
    st.stop()

st.info(f"**Model:** Random Forest (100 estimators)  |  **Target:** Kategori Pemahaman (3 kelas)  |  **Dataset:** 132 sampel")

tab_cm, tab_cv, tab_report, tab_feat = st.tabs([
    "Confusion Matrix",
    "Cross-Validation",
    "Classification Report",
    "Feature Importance"
])

pipeline, X, y = train_pipeline()

with tab_cm:
    st.subheader("Confusion Matrix — Prediksi vs Aktual")

    with st.spinner("Menjalankan prediksi pada seluruh dataset..."):
        y_pred = model_ai.predict(X)

    accuracy = accuracy_score(y, y_pred)
    st.metric("Akurasi (Full Dataset)", f"{accuracy:.2%}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        cmap='Blues',
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix — Model Random Forest", pad=20, fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Baris = Label Aktual, Kolom = Label Prediksi. Diagonal menunjukkan prediksi yang benar.")

with tab_cv:
    st.subheader("5-Fold Stratified Cross-Validation")

    with st.spinner("Menjalankan Cross-Validation (5 Fold)..."):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)

    acc_per_fold = np.round(cv_results['test_accuracy'], 2)
    f1_per_fold = np.round(cv_results['test_f1_macro'], 2)

    col_cv1, col_cv2, col_cv3 = st.columns(3)
    col_cv1.metric("Rata-rata Akurasi", f"{acc_per_fold.mean() * 100:.2f}%")
    col_cv2.metric("Rata-rata F1-Score", f"{f1_per_fold.mean() * 100:.2f}%")
    col_cv3.metric("Standar Deviasi", f"{acc_per_fold.std() * 100:.2f}%")

    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Akurasi": [f"{v:.0%}" for v in acc_per_fold],
        "F1-Score": [f"{v:.0%}" for v in f1_per_fold],
    })
    st.dataframe(cv_df, hide_index=True)

    st.markdown("**Confusion Matrix (Hasil Cross-Validation)**")
    fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred_cv,
        cmap='Blues',
        ax=ax_cv,
        xticks_rotation=45,
        colorbar=False,
    )
    ax_cv.set_title("Confusion Matrix — Cross-Validation (5 Folds)", pad=20, fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_cv)
    plt.close()

with tab_report:
    st.subheader("Classification Report per Kelas")

    target_names = sorted(y.unique())
    report_dict = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(
        report_df,
        column_config={
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1-score": st.column_config.NumberColumn("F1-Score", format="%.3f"),
            "support": st.column_config.NumberColumn("Support", format="%d"),
        },
    )

    st.markdown("**Interpretasi:**")
    st.markdown("""
    - **Precision:** Proporsi prediksi positif yang benar per kelas
    - **Recall:** Proporsi data aktual positif yang berhasil diprediksi benar
    - **F1-Score:** Rata-rata harmonik precision dan recall
    """)

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
