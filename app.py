import streamlit as st
import pandas as pd
import re
import joblib
import os

# ==========================================
# CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Autism Syntax Assistant (ASA)",
    page_icon="�",
    layout="wide"
)

# ==========================================
# 1. TAHAP PREPROCESSING
# ==========================================
def proses_teks_sintaksis(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

# ==========================================
# 2. TAHAP PARSER OTOMATIS (SIMULASI AI TAHAP 2)
# ==========================================
# ==========================================
# 2. TAHAP PARSER DINAMIS (SIMULASI AI TAHAP 2)
# ==========================================
def auto_parse_sintaksis(ujaran_bersih, echolalia):
    """
    Fungsi ini membedah pola SPOK sesuai urutan kata asli (Kiri ke Kanan).
    Menggunakan kamus leksikon mini dan logika status posisi kata.
    """
    # Bypass khusus untuk gejala klinis Echolalia/Repetisi
    token = ujaran_bersih.split()
    if echolalia == "Ya": 
        return "Echolalia"
    if len(token) >= 2 and len(set(token)) == 1:
        return "Repetisi"

    # Kamus Leksikon Mini (Berdasarkan data klinis anak)
    kata_negasi = ['tidak', 'bukan', 'jangan', 'belum']
    kata_ket = ['kemarin', 'besok', 'tadi', 'sekarang', 'sore', 'pagi', 'sangat', 'cepat', 'di', 'ke', 'dari', 'sini', 'sana', 'epat']
    kata_predikat = ['mau', 'ingin', 'minta', 'makan', 'minum', 'lari', 'main', 'duduk', 'lihat', 'putar', 'tidur', 'mandi', 'siram', 'baca', 'pergi']
    kata_nomina = ['aku', 'saya', 'kamu', 'dia', 'mama', 'papa', 'anak', 'mobil', 'bunga', 'sepeda', 'kucing', 'susu', 'air', 'buku', 'kebun', 'binatang', 'ini', 'itu']

    pola_kasar = []
    has_predikat = False
    is_ket_phrase = False # Penanda jika masuk ke frasa keterangan (misal: "ke kebun")

    # Membaca kata dari kiri ke kanan
    for kata in token:
        # Jika sebelumnya adalah preposisi (di/ke/dari), maka kata berikutnya pasti Keterangan
        if is_ket_phrase:
            pola_kasar.append("Ket")
            continue

        # Klasifikasi berdasarkan Kamus
        if kata in kata_negasi:
            pola_kasar.append("Negasi")
        elif kata in kata_ket:
            pola_kasar.append("Ket")
            if kata in ['di', 'ke', 'dari']:  # Memicu frasa keterangan bersambung
                is_ket_phrase = True
        elif kata in kata_predikat:
            pola_kasar.append("P")
            has_predikat = True
        elif kata in kata_nomina:
            # Logika Cerdas: Nomina sebelum Predikat = Subjek. Setelah Predikat = Objek.
            if not has_predikat:
                pola_kasar.append("S")
            else:
                pola_kasar.append("O")
        else:
            # Fallback untuk kata yang tidak ada di kamus
            if not has_predikat and "S" not in pola_kasar:
                pola_kasar.append("S")
            elif has_predikat:
                pola_kasar.append("O")
            else:
                pola_kasar.append("Ket")

    # PENGGABUNGAN UNSUR (Reduksi Duplikasi)
    # Mengubah ['Ket', 'S', 'Ket', 'Ket', 'Ket'] menjadi ['Ket', 'S', 'Ket']
    pola_final = []
    for p in pola_kasar:
        if not pola_final or pola_final[-1] != p:
            pola_final.append(p)

    # Mengembalikan hasil dengan pemisah "+"
    return "+".join(pola_final) if pola_final else "Tidak Teridentifikasi"

# ==========================================
# 3. TAHAP EKSTRAKSI FITUR (KOMPLEKSITAS & INTENSI)
# ==========================================
def tentukan_kompleksitas(struktur_sintaksis):
    struktur = str(struktur_sintaksis).strip().lower()
    if 'tunggal' in struktur or 'repetisi' in struktur or 'echolalia' in struktur or 'negasi' == struktur:
        return 'Kata Tunggal (K1)'
    elif 'frasa' in struktur or struktur in ['p+o', 'p+ket', 'negasi+p', 's+p']:
        return 'Frasa (K2)'
    elif struktur.count('+') >= 3:
        return 'Kalimat Majemuk (K4)'
    else:
        return 'Kalimat Sederhana (K3)'

def tentukan_intensi(ujaran, konteks):
    ujaran = str(ujaran).strip().lower()
    konteks = str(konteks).strip().lower()
    if any(kata in ujaran for kata in ['tidak', 'bukan', 'jangan', 'takut', 'maaf']):
        return 'Protesting'
    elif any(kata in ujaran for kata in ['mau', 'ingin', 'minta', 'ambil', 'buka', 'lagi', 'tolong']):
        return 'Requesting'
    elif konteks == 'instruksi' or any(kata in ujaran for kata in ['sudah', 'iya', 'belum']):
        return 'Answering'
    else:
        return 'Commenting'

# ==========================================
# 4. TAHAP KALKULATOR MEDIS (IKLA DSM-5)
# ==========================================
def hitung_ikla(mlu, kompleksitas, intensi, echolalia, konteks, token_len):
    skor_sintaksis = 27 if (kompleksitas == 'Kalimat Majemuk (K4)' or (kompleksitas == 'Kalimat Sederhana (K3)' and mlu >= 3)) else (18 if kompleksitas == 'Frasa (K2)' or mlu == 2 else 9)
    skor_leksikal = 18 if token_len > 3 else (12 if token_len >= 2 else 6)
    skor_pragmatik = 22.5 if intensi in ['Commenting', 'Answering'] else (15.0 if intensi == 'Requesting' else 10.0)
    skor_echo = 13.5 if echolalia == 'Tidak' else 3.0
    skor_inisiasi = 9.0 if konteks in ['Bermain', 'Bercerita'] else (6.0 if konteks == 'Percakapan' else 3.0)
        
    total_skor = skor_sintaksis + skor_leksikal + skor_pragmatik + skor_echo + skor_inisiasi
    
    if total_skor <= 30:
        level_asd = "Autisme Berat (DSM-5 Level 3)"
    elif total_skor <= 60:
        level_asd = "Autisme Sedang (DSM-5 Level 2)"
    else:
        level_asd = "Autisme Ringan (DSM-5 Level 1)"
    return round(total_skor, 2), level_asd

# KAMUS REKOMENDASI TERAPIS
KAMUS_REKOMENDASI = {
    'Belum Berkembang': {
        'Protesting': "Gunakan PECS (Picture Exchange Communication System) untuk membantu ekspresi penolakan. Latih kepatuhan 1 kata.",
        'Requesting': "Teknik Manding (Permintaan). Stimulasi anak memproduksi vokal dasar sebelum memberikan objek.",
        'Commenting': "Tingkatkan joint attention menggunakan mainan sensorik. Latih imitasi bunyi verbal.",
        'Answering': "Berikan stimulasi imitasi vokal 'Ya/Tidak' disertai gestur."
    },
    'Berkembang Sedang': {
        'Protesting': "Ajarkan frasa penolakan 'tidak mau'. Berikan jadwal visual untuk transisi.",
        'Requesting': "Latih perluasan kalimat dari kata tunggal menjadi frasa 2 kata.",
        'Commenting': "Pancing anak melengkapi bagian kalimat yang hilang pada gambar.",
        'Answering': "Latih menjawab pertanyaan 'Apa' dan 'Siapa' menggunakan visual."
    },
    'Sudah Mahir': {
        'Protesting': "Latih ekspresi alasan penolakan secara logis (karena/sebab).",
        'Requesting': "Tingkatkan kemampuan meminta ke struktur lengkap dengan sopan santun.",
        'Commenting': "Terapkan 'Story Retelling' untuk melatih klausa bertingkat.",
        'Answering': "Stimulasi pertanyaan tingkat tinggi 'Mengapa' dan 'Bagaimana'."
    }
}

# ==========================================
# APP UI & FRONTEND
# ==========================================
st.title("Autism Syntax Assistant (ASA)")
st.subheader("Sistem Hibrida Komputasional Linguistik & Pendukung Keputusan")
st.markdown("---")

# Load ML Model
model_path = 'model_sintaksis_real.pkl'
model_ready = os.path.exists(model_path)
if model_ready:
    model_ai = joblib.load(model_path)
else:
    st.warning("⚠️ Berkas model (.pkl) tidak ditemukan. Berjalan pada Mode Simulasi.")

col_input, col_output = st.columns([1, 1.5])

with col_input:
    st.header("Form Analisis Cepat")
    st.caption("Input disederhanakan hanya pada parameter klinis yang wajib.")
    
    with st.container(border=True):
        # UI KINI SANGAT BERSIH! HANYA 4 INPUTAN.
        tingkat_asd_input = st.selectbox("Baseline Tingkat ASD Klien", ["ASD-1", "ASD-2", "ASD-3"])
        konteks = st.selectbox("Konteks Interaksi", ["Bermain", "Percakapan", "Bercerita", "Deskripsi Gambar", "Instruksi"])
        echolalia = st.radio("Terdeteksi Echolalia / Repetisi?", ["Tidak", "Ya"], horizontal=True)
        
        ujaran_anak = st.text_area("Teks Ujaran Anak", placeholder="Ketikkan ujaran anak di sini...", height=100)
        
        tombol_analisis = st.button("Auto-Parse & Analisis", use_container_width=True)

with col_output:
    st.header("Dashboard Diagnosis")
    
    if tombol_analisis and ujaran_anak:
        with st.spinner("Mesin NLP sedang membedah kalimat..."):
            # 1. Preprocessing & Auto-Parsing
            ujaran_bersih = proses_teks_sintaksis(ujaran_anak)
            token_list = ujaran_bersih.split()
            mlu_hitung = len(token_list)
            
            # PARSER OTOMATIS BEKERJA DI SINI
            struktur_sintaksis_otomatis = auto_parse_sintaksis(ujaran_bersih, echolalia)
            
            kompleksitas_kalimat = tentukan_kompleksitas(struktur_sintaksis_otomatis)
            intensi_komunikasi = tentukan_intensi(ujaran_bersih, konteks)
            
            # 2. Prediksi ML
            if model_ready:
                data_input_ml = pd.DataFrame([{
                    'Ujaran Bersih': ujaran_bersih,
                    'ASD': tingkat_asd_input,
                    'Echolalia': echolalia,
                    'Struktur Sintaksis': struktur_sintaksis_otomatis,
                    'Kompleksitas Kalimat': kompleksitas_kalimat,
                    'Intensi Komunikasi': intensi_komunikasi,
                    'MLU': mlu_hitung
                }])
                prediksi_pemahaman = model_ai.predict(data_input_ml)[0]
            else:
                prediksi_pemahaman = "Belum Berkembang" if mlu_hitung <= 1 else "Sudah Mahir"
                
            # 3. Kalkulator IKLA & Rekomendasi
            skor_ikla, label_diagnosis = hitung_ikla(mlu_hitung, kompleksitas_kalimat, intensi_komunikasi, echolalia, konteks, mlu_hitung)
            rekomendasi_final = KAMUS_REKOMENDASI.get(prediksi_pemahaman, {}).get(intensi_komunikasi, "Lanjutkan stimulasi wicara dasar.")
            
            # Menampilkan Output
            # st.success("Analisis Selesai!")
            st.toast("Analisis Selesai!", icon="✅")

            
            # 1. Output Syntactic Parsing (Otomatis)
            st.subheader("1. Pembedahan Sintaksis (Auto-Parser)")
            st.code(f"Pola Terdeteksi: {struktur_sintaksis_otomatis}\nKompleksitas : {kompleksitas_kalimat}\nMLU Aktual   : {mlu_hitung} kata", language="markdown")
            
            # 2. Output Pemahaman & Pragmatik
            st.subheader("2. Prediksi Kognitif & Pragmatik")
            c1, c2 = st.columns(2)
            c1.info(f"**Pemahaman:** {prediksi_pemahaman}")
            c2.warning(f"**Intensi:** {intensi_komunikasi}")
            
            # 3. Output Klinis IKLA
            with st.expander("Skor Indeks Keparahan Linguistik (IKLA DSM-5)", expanded=True):
                st.metric(label="Total Skor IKLA", value=f"{skor_ikla} / 90")
                st.markdown(f"**Indikasi Klinis:** {label_diagnosis}")

            # 4. Rekomendasi
            st.subheader("3. Rekomendasi Sistem")
            st.write(f"{rekomendasi_final}")
            if echolalia == "Ya":
                st.error("⚠️ Catatan: Terapkan intervensi potong-rantai untuk mengatasi repetisi Echolalia.")
                
    elif tombol_analisis and not ujaran_anak:
        st.error("Ujaran anak tidak boleh kosong!")
    else:
        st.info("Form siap. Masukkan ujaran dan klik tombol di sebelah kiri.")