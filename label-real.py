import pandas as pd

# ==========================================
# FUNGSI 1: MENENTUKAN KOMPLEKSITAS KALIMAT
# ==========================================
def tentukan_kompleksitas(row):
    """
    Fungsi ini mengklasifikasikan kompleksitas kalimat (K1 - K4)
    berdasarkan pola Struktur Sintaksis.
    """
    struktur = str(row['Struktur Sintaksis']).strip().lower()
    
    # K1: Kata Tunggal (Hanya 1 unsur atau tidak jelas)
    if 'tunggal' in struktur or 'repetisi' in struktur or 'neologisme' in struktur or struktur == 'p' or struktur == 'negasi':
        return 'Kata Tunggal (K1)'
    
    # K2: Frasa (2 unsur tapi belum membentuk Subjek+Predikat yang utuh)
    elif 'frasa' in struktur or struktur in ['p+o', 'p+ket', 'negasi+p', 's+o', 'p+pel']:
        return 'Frasa (K2)'
    
    # K4: Kalimat Majemuk/Kompleks (Memiliki 4 unsur atau lebih, tandanya ada 3 atau lebih '+')
    elif struktur.count('+') >= 3:
        return 'Kalimat Majemuk (K4)'
    
    # K3: Kalimat Sederhana (Sisa data, umumnya S+P, S+P+O, S+P+Ket)
    else:
        return 'Kalimat Sederhana (K3)'

# ==========================================
# FUNGSI 2: MENENTUKAN INTENSI KOMUNIKASI
# ==========================================
def tentukan_intensi(row):
    """
    Fungsi ini menebak Fungsi Pragmatik (Intensi) berdasarkan 
    kata kunci dalam ujaran dan konteks interaksi.
    """
    ujaran = str(row['Ujaran Anak']).strip().lower()
    konteks = str(row['Konteks']).strip().lower()
    
    # 1. Protesting (Menolak/Menghindar)
    kata_menolak = ['tidak', 'bukan', 'jangan', 'takut', 'maaf']
    if any(kata in ujaran for kata in kata_menolak):
        return 'Protesting'
    
    # 2. Requesting (Meminta/Menuntut)
    kata_meminta = ['mau', 'ingin', 'minta', 'ambil', 'buka', 'lagi', 'tolong']
    if any(kata in ujaran for kata in kata_meminta):
        return 'Requesting'
    
    # 3. Answering (Menjawab) - Biasanya merespons instruksi atau pertanyaan
    kata_menjawab = ['sudah', 'iya', 'belum']
    if konteks == 'instruksi' or any(kata in ujaran for kata in kata_menjawab):
        return 'Answering'
    
    # 4. Commenting (Berkomentar/Mendeskripsikan) - Default untuk sisa ujaran
    # Biasanya banyak terjadi di konteks Bercerita, Bermain, atau Deskripsi Gambar
    else:
        return 'Commenting'

# ==========================================
# PROSES EKSEKUSI PADA DATASET
# ==========================================
try:
    print("Membaca file data_real_lengkap.csv...")
    df = pd.read_csv('dataset/data_real_lengkap.csv', sep=';', encoding='utf-8')

    if 'Struktur Sintaksis' not in df.columns or 'Ujaran Anak' not in df.columns:
        print("Error: Kolom 'Struktur Sintaksis' atau 'Ujaran Anak' tidak ditemukan.")
    else:
        print("Tahap 1: Mengekstrak fitur Kompleksitas Kalimat...")
        df['Kompleksitas Kalimat'] = df.apply(tentukan_kompleksitas, axis=1)

        print("Tahap 2: Mengekstrak fitur Intensi Komunikasi...")
        df['Intensi Komunikasi'] = df.apply(tentukan_intensi, axis=1)

        nama_file_final = 'dataset/data_final_siap_ml.csv'
        df.to_csv(nama_file_final, index=False, sep=';')
        
        print(f"\nSukses! File Final telah disimpan sebagai '{nama_file_final}'.")
        print("\nPratinjau Data Baru:")
        print(df[['Ujaran Anak', 'Struktur Sintaksis', 'Kompleksitas Kalimat', 'Intensi Komunikasi']].head(10))

except FileNotFoundError:
    print("Error: File 'dataset/data_real_lengkap.csv' tidak ditemukan. Pastikan Anda sudah menjalankan skrip sebelumnya.")