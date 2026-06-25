import streamlit as st
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sintaksis Autisme", layout="wide", page_icon="�")

# Menambahkan CSS kustom sederhana untuk merapikan 'Card' (Panel)
st.markdown("""
    <style>
    div.st-emotion-cache-1v0mbdj > img {margin-bottom: 20px;}
    .css-1v0mbdj {border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Header Aplikasi
st.title("Prototype Aplikasi Analisis Sintaksis")
st.markdown("Model Komputasional Bootstrapping Sintaksis pada Pemerolehan Kalimat Kompleks Anak Autisme")
st.markdown("---")

# ==========================================
# 2. MEMUAT MODEL AI (OTAK PREDIKSI)
# ==========================================
# Menggunakan @st.cache_resource agar model hanya dimuat sekali di awal (menghemat memori)
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_sintaksis.pkl')
    except Exception as e:
        st.error("Gagal memuat model. Pastikan file 'model_sintaksis.pkl' berada di folder yang sama.")
        return None

model = load_model()

# ==========================================
# 3. FUNGSI MESIN REKOMENDASI (RULE-BASED)
# ==========================================
def dapatkan_rekomendasi(prediksi):
    if prediksi == "Belum Berkembang":
        return "Fokus pada penguasaan kosakata dasar dan kalimat tunggal (S-P-O) menggunakan kartu gambar benda."
    elif prediksi == "Berkembang Sedang":
        return "Latihan kalimat subordinatif sederhana dengan bantuan visual gambar sebab-akibat."
    else: # Sudah Mahir
        return "Tingkatkan kompleksitas kalimat dengan variasi cerita beruntun dan instruksi multi-langkah."

def deteksi_relasi(teks):
    konjungsi_subordinatif = ["karena", "sehingga", "yang", "ketika", "jika", "kalau", "meskipun", "walaupun", "agar", "supaya", "bahwa", "sebab", "sejak", "setelah", "sebelum", "untuk"]
    teks_lower = teks.lower()
    if any(k in teks_lower for k in konjungsi_subordinatif):
        return "Kalimat Kompleks Subordinatif", "Sebab-Akibat / Temporal / Atributif"
    else:
        return "Kalimat Tunggal / Koordinatif", "Tidak Ada / Setara"

# ==========================================
# 4. TATA LETAK 3 KOLOM (FRONT-END)
# ==========================================
# Membagi layar menjadi 3 kolom dengan proporsi yang seimbang
col1, col2, col3 = st.columns(3)

# ----------------- KOLOM 1: INPUT -----------------
with col1:
    st.subheader("Input Data Bahasa")
    with st.container():
        input_teks = st.text_area(
            "Masukkan ujaran anak:", 
            placeholder="Masukkan ujaran anak atau kalimat yang akan dianalisis...",
            height=200
        )
        tombol_analisis = st.button("Analisis Sintaksis", use_container_width=True)

# Jika tombol ditekan dan input tidak kosong
if tombol_analisis and input_teks:
    
    # Menjalankan Prediksi (Back-End)
    prediksi_pemahaman = model.predict([input_teks])[0]
    jenis_kalimat, relasi = deteksi_relasi(input_teks)
    rekomendasi = dapatkan_rekomendasi(prediksi_pemahaman)

    # ----------------- KOLOM 2: PROSES ANALISIS -----------------
    with col2:
        st.subheader("Proses Analisis")
        
        with st.expander("1. Tokenisasi", expanded=True):
            st.write("Memecah kalimat menjadi unit kata.")
            st.info(f"Token: {input_teks.split()}")
            
        with st.expander("2. Parsing Sintaksis", expanded=True):
            st.write("Mengidentifikasi SPOK.")
            st.success("Struktur pohon sintaksis berhasil dipetakan (Simulasi Dummy).")
            
        with st.expander("3. Deteksi Relasi Klausa", expanded=True):
            st.write("Mengenali relasi sebab, waktu, dan atributif.")
            st.info(f"Relasi Ditemukan: {relasi}")
            
        with st.expander("4. Prediksi Pemerolehan", expanded=True):
            st.write("Memprediksi tingkat pemahaman sintaksis.")
            st.success("Evaluasi Machine Learning Selesai.")

    # ----------------- KOLOM 3: OUTPUT SISTEM -----------------
    with col3:
        st.subheader("Output Sistem")
        
        st.info(f"**Jenis Kalimat:**\n\n{jenis_kalimat}")
        
        st.info(f"**Relasi Sintaksis:**\n\n{relasi}")
        
        # Memberikan warna berbeda berdasarkan tingkat pemahaman
        if prediksi_pemahaman == "Belum Berkembang":
            st.error(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
        elif prediksi_pemahaman == "Berkembang Sedang":
            st.warning(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
        else:
            st.success(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
            
        st.success(f"**Rekomendasi Intervensi:**\n\n{rekomendasi}")

elif tombol_analisis and not input_teks:
    st.error("Silakan masukkan kalimat ujaran terlebih dahulu sebelum melakukan analisis.")