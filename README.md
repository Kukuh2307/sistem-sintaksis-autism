# Autism Syntax Assistant (ASA)

Sistem Hibrida Komputasional Linguistik & Pendukung Keputusan untuk Analisis Sintaksis Ujaran Anak Autisme.

---

## Gambaran Umum

ASA adalah sistem berbasis **Streamlit** yang menganalisis ujaran anak autisme secara sintaksis dan pragmatik. Sistem menggabungkan **rule-based parser SPOK** (Tahap 1–3) dengan **model Machine Learning Random Forest** (Tahap 4) serta **kalkulator klinis berbasis DSM-5** (Tahap 5) untuk menghasilkan diagnosis tingkat keparahan autisme dan rekomendasi intervensi terapi.

---

## Alur Proses (Pipeline)

```
Input Ujaran Anak
       │
       ▼
┌─────────────────────────────┐
│ Tahap 1: Text Preprocessing │
│  - Case folding             │
│  - Cleaning (hapus angka,   │
│    tanda baca, emoji)       │
│  - Whitespace normalization │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 2: Auto-Parser SPOK   │
│  - Kamus leksikon mini      │
│  - Logika posisi kata (S/O) │
│  - Deteksi Echolalia/       │
│    Repetisi                 │
│  - Reduksi duplikasi label  │
│  Output: S+P+O / Ket+S+P+O  │
│          / Echolalia / dll. │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 3: Ekstraksi Fitur    │
│  3a. Kompleksitas (K1–K4)   │
│  3b. Intensi Komunikasi     │
│      (Protesting, Requesting│
│       Answering, Commenting)│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 4: Prediksi ML        │
│  - Random Forest (100 trees)│
│  - Fitur: TF-IDF + OneHot + │
│    StandardScaler           │
│  - Akurasi: 98.49%          │
│  Output: Belum Berkembang / │
│          Berkembang Sedang /│
│          Sudah Mahir        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 5: Kalkulator IKLA    │
│  - 6 komponen (Sintaksis,   │
│    Leksikal, Pragmatik,     │
│    Echolalia, Inisiasi,     │
│    ASD Level)               │
│  - Skor total maks: 90      │
│  - Output: Level DSM-5      │
│    (Ringan/Sedang/Berat)    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 6: Sanity Check       │
│  - Validasi MLU vs level    │
│  - Validasi IKLA vs level   │
│  - Koreksi inkonsistensi    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Tahap 7: Rekomendasi        │
│  - 12 skenario rekomendasi  │
│    terapi                   │
│  - Rekomendasi tambahan per │
│    komponen skor rendah     │
└─────────────────────────────┘
```

---

## Detail Tahapan

### Tahap 1: Text Preprocessing (`app.py:19–23`)

| Langkah | Proses |
|---|---|
| Case folding | `teks.lower()` |
| Cleaning | Hapus `[^a-z\s]` (angka, tanda baca, emoji) |
| Whitespace | Normalisasi spasi berlebih |

### Tahap 2: Auto-Parser Sintaksis (`app.py:31–93`)

#### Kamus Leksikon

| Kategori | Kata Kunci |
|---|---|
| **Negasi** | `tidak`, `bukan`, `jangan`, `belum` |
| **Keterangan (Ket)** | `kemarin`, `besok`, `tadi`, `sekarang`, `sore`, `pagi`, `sangat`, `cepat`, `di`, `ke`, `dari`, `sini`, `sana`, `epat` |
| **Modal (P)** | `mau`, `ingin` (kata kerja bantu, selalu jadi P) |
| **Predikat (P/O)** | `minta`, `makan`, `minum`, `lari`, `main`, `duduk`, `lihat`, `putar`, `tidur`, `mandi`, `siram`, `baca`, `pergi` |
| **Nomina (S/O)** | `aku`, `saya`, `kamu`, `dia`, `mama`, `papa`, `anak`, `mobil`, `bunga`, `sepeda`, `kucing`, `susu`, `air`, `buku`, `kebun`, `binatang`, `ini`, `itu` |

#### Aturan Parsing

1. **Echolalia**: Jika input `echolalia == "Ya"` → label `"Echolalia"`
2. **Repetisi**: Jika token >= 2 dan `set(token)` hanya 1 unsur → label `"Repetisi"`
3. **Frasa Keterangan Bersambung**: Preposisi `di/ke/dari` → kata berikutnya otomatis `Ket`
4. **Logika S vs O**: Nomina sebelum predikat pertama = **S**; nomina setelah predikat = **O**
5. **Fallback**: Kata di luar kamus → S (jika belum ada S), O (jika sudah ada predikat), Ket (lainnya)
6. **Reduksi Duplikasi**: Deretan label identik berurutan digabung (contoh: `[Ket, Ket, Ket]` → `[Ket]`)
7. **Output**: String dipisah `+`, contoh: `"S+P+O"`, `"Echolalia"`, `"Repetisi"`

### Tahap 3a: Kompleksitas Kalimat (`app.py:98–107`)

| Aturan | Hasil |
|---|---|
| Tidak ada `+` ATAU struktur = `echolalia/repetisi/negasi/tidak teridentifikasi` | **Kata Tunggal (K1)** |
| `+` muncul 1 kali | **Frasa (K2)** |
| `+` muncul >= 3 kali | **Kalimat Majemuk (K4)** |
| Selainnya (2 tanda `+`) | **Kalimat Sederhana (K3)** |

### Tahap 3b: Intensi Komunikasi (`app.py:109–119`)

| Aturan Kata Kunci | Intensi |
|---|---|
| Mengandung `tidak`, `bukan`, `jangan`, `takut`, `maaf` | **Protesting** |
| Mengandung `mau`, `ingin`, `minta`, `ambil`, `buka`, `lagi`, `tolong` | **Requesting** |
| Konteks = instruksi ATAU mengandung `sudah`, `iya`, `belum` | **Answering** |
| Default | **Commenting** |

### Tahap 4: Prediksi Machine Learning (`app.py:344–397`, `model2.ipynb`)

**Model**: Random Forest Classifier (100 trees, `random_state=42`, `class_weight='balanced'`)

**Pipeline Preprocessing**:

| Tipe Data | Kolom | Transformasi |
|---|---|---|
| Teks | `Ujaran Bersih` | TF-IDF Vectorizer (max 100 features) |
| Kategorikal | `ASD`, `Echolalia`, `Struktur Sintaksis`, `Kompleksitas Kalimat`, `Intensi Komunikasi` | OneHotEncoder |
| Numerik | `MLU` | StandardScaler |

**Target** (3 kelas): `Belum Berkembang` (33), `Berkembang Sedang` (48), `Sudah Mahir` (51)

**Kinerja Model** (5-fold Stratified Cross-Validation):
- Akurasi: **98.49%**
- F1-score (macro): **98.34%**

**Fallback** (jika file `.pkl` tidak ditemukan):
- MLU <= 1 → `Belum Berkembang`
- MLU > 1 → `Sudah Mahir`

### Tahap 5: Kalkulator IKLA — DSM-5 (`app.py:128–152`)

Skor total maksimal: **90**, terdiri dari 6 komponen:

| Komponen | Maks | Kriteria Skor |
|---|---|---|
| **Sintaksis** | 24 | K4 / (K3 & MLU>=3) = 24; K2 / MLU=2 = 16; lainnya = 8 |
| **Leksikal** | 16 | Token >3 = 16; Token >=2 = 10; Token =1 = 5 |
| **Pragmatik** | 20 | Commenting/Answering = 20; Requesting = 13; Protesting = 9 |
| **Echolalia** | 12 | "Tidak" = 12; "Ya" = 3 |
| **Inisiasi** | 8 | Bermain/Bercerita = 8; Percakapan = 5; Deskripsi/Instruksi = 3 |
| **ASD Level** | 10 | ASD-1 = 10; ASD-2 = 6; ASD-3 = 2 |

#### Output Diagnosis DSM-5

| Total Skor | Diagnosis |
|---|---|
| <= 30 | **Autisme Berat (Level 3)** |
| 31 – 60 | **Autisme Sedang (Level 2)** |
| > 60 | **Autisme Ringan (Level 1)** |

### Tahap 6: Sanity Check (`app.py:311–335`)

#### Validasi MLU vs Level Pemahaman

| Level | Rentang MLU |
|---|---|
| Belum Berkembang | 0 – 1 |
| Berkembang Sedang | 2 – 3 |
| Sudah Mahir | >= 4 |

Jika prediksi ML tidak sesuai rentang MLU → koreksi paksa ke level yang sesuai.

#### Validasi IKLA vs Level

- "Sudah Mahir" tapi IKLA <= 60 → turunkan ke "Berkembang Sedang"
- "Berkembang Sedang" tapi IKLA <= 30 → turunkan ke "Belum Berkembang"

### Tahap 7: Rekomendasi Terapis (`app.py:154–320`)

#### Matriks Rekomendasi (3 level × 4 intensi = 12 skenario)

| Level | Protesting | Requesting | Commenting | Answering |
|---|---|---|---|---|
| **Belum Berkembang** | PECS kartu "tidak" | Teknik Manding vokal | Joint attention & imitasi | Imitasi vokal Ya/Tidak |
| **Berkembang Sedang** | Frasa "tidak mau" + jadwal visual | Ekspansi 1→2 kata | Teknik Cloze kalimat | Visual support + prompt bertingkat |
| **Sudah Mahir** | Alasan penolakan dg "karena" | Kalimat lengkap + santun | Story Retelling 3 klausa | Stimulasi "Mengapa"/"Bagaimana" |

Setiap rekomendasi mencakup: **Target**, **Metode**, **Contoh**, **Frekuensi**, **Indikator**.

#### Rekomendasi Tambahan

Jika skor komponen < 50% dari maksimal, sistem menampilkan catatan per komponen:
- **Sintaksis**: Latih perluasan 1→2→3 kata
- **Leksikal**: Ekspos 3-5 kosakata baru per sesi
- **Pragmatik**: Fokus turn-taking & kontak mata
- **Echolalia**: Intervensi potong-rantai
- **Inisiasi**: Ciptakan situasi butuh inisiatif
- **ASD Level**: Sesuaikan target intervensi dengan tingkat keparahan — fokus pada komunikasi dasar (PECS/isyarat) untuk ASD-3

Jika MLU <= 2 → rekomendasi ekspansi kalimat (+1 kata dari produksi anak).

---

## Dataset

**Sumber**: 65 anak autisme (A001–A065), 2 ujaran per anak = **132 sampel**

| Kolom | Deskripsi |
|---|---|
| `ID Anak` | Kode unik (A001–A065) |
| `JK` | Jenis kelamin (L/P) |
| `ASD` | Tingkat autisme (ASD-1/2/3) |
| `Usia` | Usia dalam format tahun;bulan |
| `Konteks` | Bermain, Percakapan, Bercerita, Deskripsi Gambar, Instruksi |
| `Ujaran Anak` | Teks asli ujaran anak |
| `Struktur Sintaksis` | Pola SPOK label manual |
| `MLU` | Mean Length of Utterance (jumlah kata) |
| `Echolalia` | Ya/Tidak |

### Pipeline Pelabelan Dataset

1. **`pre_processing.py`**: Preprocessing → `Ujaran Bersih` + `Token` + Aturan `tentukan_kategori()` → `Kategori Pemahaman`
2. **`label-real.py`**: Ekstraksi fitur `Kompleksitas Kalimat` (K1–K4) + `Intensi Komunikasi`

---

## Struktur File

```
├── app.py                          # Aplikasi Streamlit utama
├── pre_processing.py               # Preprocessing & pelabelan dataset (tahap 1)
├── label-real.py                   # Ekstraksi fitur dataset (tahap 2)
├── model2.ipynb                    # Notebook training model Random Forest
├── model_sintaksis_real.pkl        # Model ML terlatih
├── version.py                      # Versi pustaka
├── requirements.txt                # Dependensi
├── dataset/
│   ├── data_real.csv               # Dataset mentah (132 sampel)
│   ├── data_real_lengkap.csv       # Setelah preprocessing + kategori
│   └── data_final_siap_ml.csv      # Setelah ekstraksi fitur (siap ML)
├── app_old/
│   ├── app1.py                     # Prototype awal (rule-based sederhana)
│   ├── label.py                    # Labeling versi lama
│   ├── model_sintaksis.pkl         # Model versi lama
│   └── model1.ipynb                # Notebook versi lama
└── README.md
```

---

## Teknologi

| Komponen | Tools |
|---|---|
| **Framework** | Streamlit 1.58.0 |
| **Machine Learning** | scikit-learn 1.6.1 (Random Forest) |
| **Data Processing** | pandas 3.0.2 |
| **Model Serialization** | joblib 1.4.2 |
| **Text Processing** | regex (re) |
| **Lingkungan** | Python 3.12 |

---

## Cara Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

---

## Catatan Klinis

Sistem ini dirancang sebagai **alat bantu keputusan** (decision support system) untuk terapis dan klinisi. Hasil analisis dan rekomendasi bersifat indikatif dan tidak menggantikan diagnosis profesional dari dokter/psikolog klinis.

---
