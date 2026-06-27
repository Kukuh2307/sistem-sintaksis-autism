import pandas as pd

def berikan_label_sintetis(teks):
    # Memastikan input berupa string
    if not isinstance(teks, str):
        return "Belum Berkembang"
        
    # Memecah kalimat menjadi token kata
    kata = teks.split()
    jumlah_kata = len(kata)
    
    # Mengubah ke huruf kecil agar pencocokan konjungsi lebih akurat
    teks_lower = teks.lower()
    kata_lower = teks_lower.split()
    
    # Daftar konjungsi sesuai aturan
    konjungsi_sederhana = ["dan", "atau", "tetapi", "serta", "lalu", "kemudian", "melainkan", "sedangkan"]
    konjungsi_subordinatif = ["karena", "sehingga", "yang", "ketika", "jika", "kalau", "meskipun", "walaupun", "agar", "supaya", "bahwa", "sebab", "sejak", "setelah", "sebelum", "untuk"]
    
    # Mengecek keberadaan konjungsi dalam kalimat
    ada_konjungsi_sederhana = any(k in kata_lower for k in konjungsi_sederhana)
    ada_konjungsi_subordinatif = any(k in kata_lower for k in konjungsi_subordinatif)
    
    # Aturan 3: Sudah Mahir
    # Batas kata diturunkan dari > 15 menjadi >= 12 agar kelas ini kembali mendeteksi kalimat kompleks dengan optimal.
    if jumlah_kata >= 12 and ada_konjungsi_subordinatif:
        return "Sudah Mahir"
        
    # Aturan 2: Berkembang Sedang
    # Rentang kata dipersempit menjadi 7–11 kata agar tidak terlalu mendominasi keseluruhan data latih.
    elif (7 <= jumlah_kata <= 11) or ada_konjungsi_sederhana:
        return "Berkembang Sedang"
        
    # Aturan 1: Belum Berkembang
    # Semua kalimat pendek dengan batas kurang dari 7 kata (dan tanpa konjungsi) akan aman tertampung di sini.
    else:
        return "Belum Berkembang"

# ==========================================
# BAGIAN EKSEKUSI DATA
# ==========================================

print("Mulai membaca data...")

# 1. Membaca data
df = pd.read_csv('dataset/Corpus_Kalimat.csv')

# 2. Menyimpan nama kolom teks
nama_kolom_teks = 'Kalimat'

# 3. Menerapkan fungsi ke seluruh baris data untuk membuat kolom baru
print("Memproses pelabelan untuk 1000+ baris...")
df['Tingkat Pemahaman'] = df[nama_kolom_teks].apply(berikan_label_sintetis)

# 4. Menyimpan hasil ke file baru
df.to_csv('data_kalimat_berlabel.csv', index=False)
print("Selesai! File 'data_kalimat_berlabel.csv' berhasil dibuat.")