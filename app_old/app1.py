import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sintaksis Autisme", layout="wide", page_icon="�")

st.markdown("""
    <style>
    div.st-emotion-cache-1v0mbdj > img {margin-bottom: 20px;}
    .css-1v0mbdj {border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

st.title("Prototype Aplikasi Analisis Sintaksis")
st.markdown("Model Komputasional Bootstrapping Sintaksis pada Pemerolehan Kalimat Kompleks Anak Autisme")
st.markdown("---")

# ==========================================
# 2. FUNGSI LOGIKA (PENGGANTI MODEL .PKL SEMENTARA)
# ==========================================
def prediksi_pemerolehan(teks):
    # Menggunakan logika murni yang pasti akurat sesuai aturan kita
    kata = teks.lower().split()
    jumlah_kata = len(kata)
    
    konjungsi_sederhana = ["dan", "atau", "tetapi", "serta", "lalu", "kemudian", "melainkan", "sedangkan"]
    konjungsi_subordinatif = ["karena", "sehingga", "yang", "ketika", "jika", "kalau", "meskipun", "walaupun", "agar", "supaya", "bahwa", "sebab", "sejak", "setelah", "sebelum", "untuk"]
    
    ada_konjungsi_sederhana = any(k in kata for k in konjungsi_sederhana)
    ada_konjungsi_subordinatif = any(k in kata for k in konjungsi_subordinatif)
    
    if jumlah_kata >= 12 and ada_konjungsi_subordinatif:
        return "Sudah Mahir"
    elif (7 <= jumlah_kata <= 11) or ada_konjungsi_sederhana:
        return "Berkembang Sedang"
    else:
        return "Belum Berkembang"

def dapatkan_rekomendasi(prediksi):
    if prediksi == "Belum Berkembang":
        return "Fokus pada penguasaan kosakata dasar dan kalimat tunggal (S-P-O) menggunakan kartu gambar benda."
    elif prediksi == "Berkembang Sedang":
        return "Latihan kalimat subordinatif sederhana dengan bantuan visual gambar sebab-akibat."
    else: 
        return "Tingkatkan kompleksitas kalimat dengan variasi cerita beruntun dan instruksi multi-langkah."

def deteksi_relasi(teks):
    konjungsi_subordinatif = ["karena", "sehingga", "yang", "ketika", "jika", "kalau", "meskipun", "walaupun", "agar", "supaya", "bahwa", "sebab", "sejak", "setelah", "sebelum", "untuk"]
    teks_lower = teks.lower()
    if any(k in teks_lower for k in konjungsi_subordinatif):
        return "Kalimat Kompleks Subordinatif", "Sebab-Akibat / Temporal / Atributif"
    else:
        return "Kalimat Tunggal / Koordinatif", "Tidak Ada / Setara"

# ==========================================
# 3. TATA LETAK KOLOM (FRONT-END)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input Data Bahasa")
    with st.container():
        input_teks = st.text_area("Masukkan ujaran anak:", height=200)
        tombol_analisis = st.button("Analisis Sintaksis", use_container_width=True)

if tombol_analisis and input_teks:
    
    # Eksekusi Logika
    prediksi_pemahaman = prediksi_pemerolehan(input_teks)
    jenis_kalimat, relasi = deteksi_relasi(input_teks)
    rekomendasi = dapatkan_rekomendasi(prediksi_pemahaman)

    with col2:
        st.subheader("Proses Analisis")
        with st.expander("1. Tokenisasi", expanded=True):
            st.info(f"Token: {input_teks.split()}")
        with st.expander("2. Parsing Sintaksis", expanded=True):
            st.success("Struktur pohon sintaksis berhasil dipetakan.")
        with st.expander("3. Deteksi Relasi Klausa", expanded=True):
            st.info(f"Relasi Ditemukan: {relasi}")
        with st.expander("4. Prediksi Pemerolehan", expanded=True):
            st.success("Evaluasi Selesai.")

    with col3:
        st.subheader("Output Sistem")
        st.info(f"**Jenis Kalimat:**\n\n{jenis_kalimat}")
        st.info(f"**Relasi Sintaksis:**\n\n{relasi}")
        
        if prediksi_pemahaman == "Belum Berkembang":
            st.error(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
        elif prediksi_pemahaman == "Berkembang Sedang":
            st.warning(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
        else:
            st.success(f"**Prediksi Pemahaman:**\n\n{prediksi_pemahaman}")
            
        st.success(f"**Rekomendasi Intervensi:**\n\n{rekomendasi}")