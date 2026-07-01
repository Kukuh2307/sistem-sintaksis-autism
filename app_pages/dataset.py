import os
import streamlit as st
import pandas as pd

st.title(":material/table: Dataset")
st.caption("Eksplorasi seluruh data ujaran anak dengan ASD")

DATA_PATH = "dataset/dataset.csv"

@st.cache_data
def load_dataset(_mtime):
    df = pd.read_csv(DATA_PATH, sep=";")
    df.fillna("Tidak Diketahui", inplace=True)
    return df

mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0
df = load_dataset(mtime)

total_samples = len(df)
total_children = df["ID Anak"].nunique()
asd_counts = df["ASD"].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sampel Ujaran", total_samples)
col2.metric("Total Anak", total_children)
col3.metric("Kelas ASD", f"{len(asd_counts)} level")
col4.metric("Total Fitur", len(df.columns))

st.divider()

st.subheader(":material/table_rows: Tabel Dataset Lengkap")
st.caption("Dataset referensi dari `dataset/dataset.csv` — 14 kolom fitur klinis & linguistik")

st.dataframe(
    df,
    hide_index=True,
    column_config={
        "ID Anak": st.column_config.TextColumn("ID Anak", width="small"),
        "JK": st.column_config.TextColumn("JK", width="small"),
        "ASD": st.column_config.TextColumn("ASD", width="small"),
        "Usia": st.column_config.TextColumn("Usia", width="small"),
        "Konteks": st.column_config.TextColumn("Konteks", width="small"),
        "Ujaran Anak": st.column_config.TextColumn("Ujaran Anak", width="medium"),
        "Struktur Sintaksis": st.column_config.TextColumn("Struktur", width="small"),
        "MLU": st.column_config.NumberColumn("MLU", width="small"),
        "Echolalia": st.column_config.TextColumn("Echolalia", width="small"),
        "Ujaran Bersih": st.column_config.TextColumn("Ujaran Bersih", width="medium"),
        "Token": st.column_config.ListColumn("Token", width="medium"),
        "Kategori Pemahaman": st.column_config.TextColumn("Kategori", width="small"),
        "Kompleksitas Kalimat": st.column_config.TextColumn("Kompleksitas", width="small"),
        "Intensi Komunikasi": st.column_config.TextColumn("Intensi", width="small"),
    },
)

st.divider()

st.subheader(":material/bar_chart: Visualisasi Distribusi Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribusi ASD & Demografi",
    "Pemahaman & Kompleksitas",
    "Intensi & Echolalia",
    "Analisis MLU"
])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribusi Tingkat ASD**")
        asd_chart = df["ASD"].value_counts().reset_index()
        asd_chart.columns = ["ASD", "Jumlah"]
        st.bar_chart(asd_chart, x="ASD", y="Jumlah", color="ASD")

    with col_b:
        st.markdown("**Distribusi Jenis Kelamin**")
        jk_chart = df["JK"].value_counts().reset_index()
        jk_chart.columns = ["JK", "Jumlah"]
        st.bar_chart(jk_chart, x="JK", y="Jumlah", color="JK")

    st.markdown("**Distribusi Konteks Interaksi**")
    konteks_chart = df["Konteks"].value_counts().reset_index()
    konteks_chart.columns = ["Konteks", "Jumlah"]
    st.bar_chart(konteks_chart, x="Konteks", y="Jumlah", color="Konteks")

    st.markdown("**Cross-tab: ASD vs Jenis Kelamin**")
    crosstab_jk = pd.crosstab(df["ASD"], df["JK"])
    st.dataframe(crosstab_jk, use_container_width=True)

with tab2:
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Distribusi Kategori Pemahaman**")
        kat_chart = df["Kategori Pemahaman"].value_counts().reset_index()
        kat_chart.columns = ["Kategori Pemahaman", "Jumlah"]
        st.bar_chart(kat_chart, x="Kategori Pemahaman", y="Jumlah", color="Kategori Pemahaman")

    with col_d:
        st.markdown("**Distribusi Kompleksitas Kalimat**")
        komp_chart = df["Kompleksitas Kalimat"].value_counts().reset_index()
        komp_chart.columns = ["Kompleksitas Kalimat", "Jumlah"]
        st.bar_chart(komp_chart, x="Kompleksitas Kalimat", y="Jumlah", color="Kompleksitas Kalimat")

    st.markdown("**Cross-tab: ASD vs Kategori Pemahaman**")
    crosstab_asd_kat = pd.crosstab(df["ASD"], df["Kategori Pemahaman"])
    st.dataframe(crosstab_asd_kat, use_container_width=True)
    st.caption("Baris = Tingkat ASD, Kolom = Kategori Pemahaman")

    st.markdown("**Cross-tab: ASD vs Kompleksitas Kalimat**")
    crosstab_asd_komp = pd.crosstab(df["ASD"], df["Kompleksitas Kalimat"])
    st.dataframe(crosstab_asd_komp, use_container_width=True)

with tab3:
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("**Distribusi Intensi Komunikasi**")
        int_chart = df["Intensi Komunikasi"].value_counts().reset_index()
        int_chart.columns = ["Intensi Komunikasi", "Jumlah"]
        st.bar_chart(int_chart, x="Intensi Komunikasi", y="Jumlah", color="Intensi Komunikasi")

    with col_f:
        st.markdown("**Distribusi Echolalia**")
        echo_chart = df["Echolalia"].value_counts().reset_index()
        echo_chart.columns = ["Echolalia", "Jumlah"]
        st.bar_chart(echo_chart, x="Echolalia", y="Jumlah", color="Echolalia")

    st.markdown("**Cross-tab: Echolalia vs ASD**")
    crosstab_echo = pd.crosstab(df["Echolalia"], df["ASD"])
    st.dataframe(crosstab_echo, use_container_width=True)

    st.markdown("**Cross-tab: Intensi vs Kategori Pemahaman**")
    crosstab_int = pd.crosstab(df["Intensi Komunikasi"], df["Kategori Pemahaman"])
    st.dataframe(crosstab_int, use_container_width=True)

with tab4:
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown("**Distribusi MLU**")
        mlu_dist = df["MLU"].value_counts().sort_index().reset_index()
        mlu_dist.columns = ["MLU", "Jumlah"]
        st.bar_chart(mlu_dist, x="MLU", y="Jumlah")

    with col_h:
        st.markdown("**Rata-rata MLU per ASD**")
        mlu_asd = df.groupby("ASD")["MLU"].mean().reset_index()
        mlu_asd.columns = ["ASD", "Rata-rata MLU"]
        st.bar_chart(mlu_asd, x="ASD", y="Rata-rata MLU", color="ASD")

    st.markdown("**Rata-rata MLU per Kategori Pemahaman**")
    mlu_kat = df.groupby("Kategori Pemahaman")["MLU"].mean().reset_index()
    mlu_kat.columns = ["Kategori Pemahaman", "Rata-rata MLU"]
    st.bar_chart(mlu_kat, x="Kategori Pemahaman", y="Rata-rata MLU", color="Kategori Pemahaman")

    st.markdown("**Statistik Deskriptif MLU**")
    mlu_stats = df["MLU"].describe().reset_index()
    mlu_stats.columns = ["Statistik", "Nilai"]
    st.dataframe(mlu_stats, hide_index=True)
