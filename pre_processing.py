import pandas as pd
import re

# ==========================================
# FUNGSI 1: PRA-PEMROSESAN TEKS
# ==========================================
def proses_teks_sintaksis(teks):
    """
    Fungsi ini melakukan pra-pemrosesan teks sesuai dengan 
    kebutuhan analisis sintaksis klinis.
    """
    teks = str(teks)
    
    # Case Folding
    teks = teks.lower()
    
    # Data Cleaning (Menghapus tanda baca, angka, emoji)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    
    # Tokenization
    token = teks.split()
    
    return teks, token

# ==========================================
# FUNGSI 2: PELABELAN KATEGORI PEMAHAMAN
# ==========================================
def tentukan_kategori(row):
    """
    Fungsi untuk menentukan Kategori Pemahaman berdasarkan 
    aturan klinis sintaksis anak autisme (Echolalia, Struktur, MLU).
    """
    echolalia = str(row['Echolalia']).strip().lower()
    struktur = str(row['Struktur Sintaksis']).strip().lower()
    
    try:
        mlu = int(row['MLU'])
    except ValueError:
        mlu = 0

    # Aturan 1: Belum Berkembang
    kata_kunci_fase_awal = ['kata tunggal', 'repetisi', 'neologisme', 'predikat', 'nomina tunggal']
    if echolalia == 'ya' or mlu == 1 or any(keyword in struktur for keyword in kata_kunci_fase_awal):
        return 'Belum Berkembang'

    # Aturan 3: Sudah Mahir (Tidak echolalia, MLU >= 3, dan minimal 3 unsur SPOK / 2 tanda '+')
    elif echolalia == 'tidak' and mlu >= 3 and struktur.count('+') >= 2:
        return 'Sudah Mahir'

    # Aturan 2: Berkembang Sedang (Sisa data frasa sederhana)
    else:
        return 'Berkembang Sedang'

# ==========================================
# PROSES EKSEKUSI PADA DATASET
# ==========================================
try:
    print("Membaca file data_real.csv...")
    # Membaca dataset CSV dengan pemisah titik koma (;)
    df = pd.read_csv('dataset/data_real.csv', sep=';', encoding='utf-8')

    # Memastikan semua kolom yang dibutuhkan tersedia
    kolom_wajib = ['Ujaran Anak', 'Struktur Sintaksis', 'MLU', 'Echolalia']
    if not all(kolom in df.columns for kolom in kolom_wajib):
        print(f"Error: Pastikan file CSV Anda memiliki kolom berikut: {kolom_wajib}")
    else:
        print("Tahap 1: Memulai proses Text Preprocessing...")
        # Menerapkan fungsi preprocessing
        df[['Ujaran Bersih', 'Token']] = df['Ujaran Anak'].apply(
            lambda x: pd.Series(proses_teks_sintaksis(x))
        )

        print("Tahap 2: Menerapkan aturan Pelabelan Klinis...")
        # Menerapkan fungsi pelabelan medis
        df['Kategori Pemahaman'] = df.apply(tentukan_kategori, axis=1)

        # Menyimpan hasil akhir ke file baru dengan pemisah titik koma (;) agar rapi di Excel
        nama_file_baru = 'dataset/data_real_lengkap.csv'
        df.to_csv(nama_file_baru, index=False, sep=';')
        
        print(f"\nSelesai! File berhasil disimpan sebagai '{nama_file_baru}'.\n")
        
        # Menampilkan pratinjau data
        print("Pratinjau Hasil Akhir:")
        # Menampilkan kolom-kolom inti yang baru saja diproses
        print(df[['Ujaran Anak', 'Struktur Sintaksis', 'Kategori Pemahaman', 'Ujaran Bersih', 'Token']].head())

except FileNotFoundError:
    print("Error: File 'dataset/data_real.csv' tidak ditemukan. Pastikan path dan lokasi file benar.")
except Exception as e:
    print(f"Terjadi kesalahan yang tidak terduga: {e}")