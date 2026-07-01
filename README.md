# ATLAS (Automated Text & Language Assessment System)

Sistem Hibrida Komputasional Linguistik & Pendukung Keputusan untuk Assessment Linguistik & Sintaksis Ujaran Anak Autisme.

---

## Gambaran Umum

ATLAS adalah sistem berbasis **Streamlit** yang menganalisis ujaran anak autisme secara sintaksis dan pragmatik. Sistem menggabungkan **hybrid parser SPOK** (rule-based expanded lexicon untuk visualisasi + rule-based mini untuk ML), **model Machine Learning multi-output Random Forest**, serta **kalkulator klinis berbasis DSM-5** untuk menghasilkan diagnosis tingkat keparahan autisme dan rekomendasi intervensi terapi.

Fitur utama:
- **Multi-output ML**: 1 model memprediksi **Kategori Pemahaman** (3 kelas) + **Tingkat ASD** (3 level) secara simultan
- **IKLA otomatis**: Skor IKLA & DSM Level menggunakan output ASD dari ML, bukan input manual
- **Sanity check**: Validasi inkonsistensi MLU → Pemahaman dan IKLA → Level

---

## Navigasi Aplikasi

Aplikasi memiliki 3 halaman yang dapat diakses melalui navigasi atas:

| Halaman | Ikon | Fungsi |
|---|---|---|
| **Analisis** | `:material/quick_reference:` | Form input ujaran → parsing SPOK → prediksi ML → skor IKLA → rekomendasi terapi |
| **Dataset** | `:material/table:` | Tabel interaktif dataset + visualisasi distribusi data (ASD, JK, kategori, MLU, dll) |
| **Evaluasi Model** | `:material/analytics:` | Confusion matrix, cross-validation, classification report, feature importance |

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
┌─────────────────────────────────────┐
│ Tahap 2: Hybrid Parsing SPOK        │
│  ┌─────────────────────────────┐    │
│  │ 2a. Rule-based Expanded     │    │
│  │     Lexicon — untuk         │    │
│  │     visualisasi anotasi     │    │
│  │  ~200+ kata (verba, nomina, │    │
│  │  keterangan, negasi, modal) │    │
│  └─────────────┬───────────────┘    │
│                ▼                    │
│  ┌─────────────────────────────┐    │
│  │ 2b. Rule-based Mini —       │    │
│  │     untuk feature ML        │    │
│  │  Kamus mini + logika posisi │    │
│  │  Output: S+P+O / Ket+S+P+O  │    │
│  └─────────────┬───────────────┘    │
└─────────────────────────────────────┘
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
┌──────────────────────────────────────┐
│ Tahap 4: Prediksi ML (Multi-Output)  │
│  - Random Forest (100 trees,         │
│    multioutput.MultiOutputClassifier)│
│  - Fitur: TF-IDF + OneHot +          │
│    StandardScaler                    │
│  - Akurasi Pemahaman: 94.16%         │
│  - Akurasi ASD: 75.91%               │
│  Output 1: Belum Berkembang /        │
│            Berkembang Sedang /       │
│            Sudah Mahir               │
│  Output 2: ASD-1 / ASD-2 / ASD-3     │
└─────────────┬────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Tahap 5: Kalkulator IKLA (Auto-ASD)  │
│  - 6 komponen (Sintaksis, Leksikal,  │
│    Pragmatik, Echolalia, Inisiasi,   │
│    ASD Level dari output ML)         │
│  - Skor total maks: 90               │
│  - Output: Level DSM-5               │
│    (Ringan/Sedang/Berat)             │
└─────────────┬────────────────────────┘
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

### Tahap 1: Text Preprocessing (`app_pages/home.py:12`)

| Langkah | Proses |
|---|---|
| Case folding | `teks.lower()` |
| Cleaning | Hapus `[^a-z\s]` (angka, tanda baca, emoji) |
| Whitespace | Normalisasi spasi berlebih |

### Tahap 2: Hybrid Parsing SPOK

#### 2a. Rule-based Expanded Lexicon — untuk Visualisasi

Parser dengan leksikon diperluas (~80 verba, ~80 nomina, ~30 keterangan, 6 negasi, 6 modal) untuk anotasi SPOK berwarna via `st-annotated-text`.

| Kategori | Jumlah | Contoh |
|---|---|---|
| **KATA_VERBA** | ~80 | `makan`, `minum`, `baca`, `cuci`, `beli`, `lari`, `mandi`, `tidur`, `tulis`, `buka`, ... |
| **KATA_NOMINA** | ~80 | `saya`, `mama`, `papa`, `kakak`, `adik`, `buku`, `sepeda`, `piring`, `baju`, `bola`, ... |
| **KATA_KET** | ~30 | `kemarin`, `tadi`, `nanti`, `di`, `ke`, `dari`, `sangat`, `cepat`, `sudah`, `sedang`, ... |
| **KATA_NEGASI** | 6 | `tidak`, `bukan`, `jangan`, `belum`, `tak`, `tiada` |
| **KATA_MODAL** | 6 | `mau`, `ingin`, `bisa`, `dapat`, `harus`, `akan` |

**Stemming dengan Sastrawi**: Sebelum lookup, setiap token di-*stem* (`stemmer.stem(token)`) untuk menangani variasi morfologis bahasa Indonesia — `memakan`, `dimakan`, `makannya` semua dikenali sebagai `makan`. Tanpa Sastrawi, afiksasi akan membuat gagal mencocokkan kamus.

**Aturan Parsing:**

1. **Echolalia/Repetisi**: Input `echolalia == "Ya"` → label `"Echolalia"`; token >= 2 dan semua sama → `"Repetisi"`
2. **Frasa Keterangan Bersambung**: Preposisi `di/ke/dari` → kata berikutnya otomatis `Ket`
3. **Negasi**: `tidak`, `bukan`, dll → `Negasi`
4. **Modal → P**: `mau`, `ingin`, `bisa`, `dapat`, `harus`, `akan` → `P` + tandai `has_predikat = True`
5. **Verba → P**: Semua verba selalu jadi `P`, tidak peduli status `has_predikat` (verba setelah modal tetap P, bukan O)
6. **Nomina**: Sebelum predikat pertama → `S`; setelah predikat → `O`
7. **Fallback**: Kata di luar kamus → `S` (jika belum ada S), `P` (jika S sudah ada dan belum ada predikat), `O` (jika sudah ada predikat)
8. **Reduksi Duplikasi**: Label identik berurutan digabung (P+P → P)

#### 2b. Rule-based Parser — untuk Feature ML

Parser berbasis kamus leksikon mini dan logika posisi kata (kiri ke kanan). Sama seperti parser 2a, setiap token di-*stem* dengan **Sastrawi** sebelum lookup kamus. Output tetap dipertahankan dalam format `S+P+O` yang konsisten dengan data training, sehingga **model ML tidak perlu retrain**.

**Kamus Leksikon:**

| Kategori | Kata Kunci |
|---|---|
| **Negasi** | `tidak`, `bukan`, `jangan`, `belum` |
| **Keterangan (Ket)** | `kemarin`, `besok`, `tadi`, `sekarang`, `sore`, `pagi`, `sangat`, `cepat`, `di`, `ke`, `dari`, `sini`, `sana`, `epat` |
| **Modal (P)** | `mau`, `ingin` (kata kerja bantu, selalu jadi P) |
| **Predikat (P/O)** | `minta`, `makan`, `minum`, `lari`, `main`, `duduk`, `lihat`, `putar`, `tidur`, `mandi`, `siram`, `baca`, `pergi` |
| **Nomina (S/O)** | `aku`, `saya`, `kamu`, `dia`, `mama`, `papa`, `anak`, `mobil`, `bunga`, `sepeda`, `kucing`, `susu`, `air`, `buku`, `kebun`, `binatang`, `ini`, `itu` |

**Aturan Parsing:**

1. **Echolalia**: Jika input `echolalia == "Ya"` → label `"Echolalia"`
2. **Repetisi**: Jika token >= 2 dan `set(token)` hanya 1 unsur → label `"Repetisi"`
3. **Frasa Keterangan Bersambung**: Preposisi `di/ke/dari` → kata berikutnya otomatis `Ket`
4. **Negasi**: `tidak`, `bukan`, `jangan`, `belum` → `Negasi`
5. **Modal → P**: `mau`, `ingin` → `P` + tandai `has_predikat = True`
6. **Verba**: Sebelum predikat → `P`; setelah predikat → `O`
7. **Logika S vs O**: Nomina sebelum predikat pertama = **S**; nomina setelah predikat = **O**
8. **Fallback**: Kata di luar kamus → S (jika belum ada S), P (jika S sudah ada dan belum ada predikat), O (jika sudah ada predikat)
9. **Reduksi Duplikasi**: Deretan label identik berurutan digabung (contoh: `[Ket, Ket, Ket]` → `[Ket]`)
10. **Output**: String dipisah `+`, contoh: `"S+P+O"`, `"Echolalia"`, `"Repetisi"`

### Tahap 3a: Kompleksitas Kalimat (`app_pages/home.py:155`)

| Aturan | Hasil |
|---|---|
| Tidak ada `+` ATAU struktur = `echolalia/repetisi/negasi/tidak teridentifikasi` | **Kata Tunggal (K1)** |
| `+` muncul 1 kali | **Frasa (K2)** |
| `+` muncul >= 3 kali | **Kalimat Majemuk (K4)** |
| Selainnya (2 tanda `+`) | **Kalimat Sederhana (K3)** |

### Tahap 3b: Intensi Komunikasi (`app_pages/home.py:166`)

| Aturan Kata Kunci | Intensi |
|---|---|
| Mengandung `tidak`, `bukan`, `jangan`, `takut`, `maaf` | **Protesting** |
| Mengandung `mau`, `ingin`, `minta`, `ambil`, `buka`, `lagi`, `tolong` | **Requesting** |
| Konteks = instruksi ATAU mengandung `sudah`, `iya`, `belum` | **Answering** |
| Default | **Commenting** |

### Tahap 4: Prediksi Machine Learning Multi-Output (`model.ipynb`, implementasi di `app_pages/home.py:528–542`)

**Model**: `MultiOutputClassifier` (Random Forest 100 trees, `random_state=42`, `class_weight='balanced'`)

**Dual Target**:
| Target | Kelas | Distribusi |
|---|---|---|
| **Kategori Pemahaman** | Belum Berkembang / Berkembang Sedang / Sudah Mahir | 33 / 48 / 56 |
| **Tingkat ASD** | ASD-1 / ASD-2 / ASD-3 | 39 / 66 / 32 |

**Pipeline Preprocessing**:

| Tipe Data | Kolom | Transformasi |
|---|---|---|
| Teks | `Ujaran Bersih` | TF-IDF Vectorizer (max 100 features) |
| Kategorikal | `Echolalia`, `Struktur Sintaksis`, `Kompleksitas Kalimat`, `Intensi Komunikasi` | OneHotEncoder |
| Numerik | `MLU` | StandardScaler |

> **Catatan**: `ASD` tidak digunakan sebagai fitur input ML untuk menghindari sirkularitas (ASD adalah target kedua).

**Kinerja Model** (5-fold StratifiedGroupKFold, group by ID Anak):

| Target | Akurasi | F1-Score (macro) |
|---|---|---|
| Kategori Pemahaman | **94.16%** | **94.02%** |
| Tingkat ASD | **75.91%** | **74.88%** |

**Fallback** (jika file `.pkl` tidak ditemukan):
- MLU <= 1 → `Belum Berkembang`, MLU > 1 → `Sudah Mahir`
- ASD selalu fallback ke ASD-2 (netral)

### Tahap 5: Kalkulator IKLA — DSM-5 (`app_pages/home.py:178`)

Skor total maksimal: **90**, terdiri dari 6 komponen:

| Komponen | Maks | Kriteria Skor |
|---|---|---|
| **Sintaksis** | 24 | K4 / (K3 & MLU>=3) = 24; K2 / MLU=2 = 16; lainnya = 8 |
| **Leksikal** | 16 | Token >3 = 16; Token >=2 = 10; Token =1 = 5 |
| **Pragmatik** | 20 | Commenting/Answering = 20; Requesting = 13; Protesting = 9 |
| **Echolalia** | 12 | "Tidak" = 12; "Ya" = 3 |
| **Inisiasi** | 8 | Bermain/Bercerita = 8; Percakapan = 5; Deskripsi/Instruksi = 3 |
| **ASD Level** | 10 | ASD-1 = 10; ASD-2 = 6; ASD-3 = 2 |

> **ASD Level diisi otomatis** dari output prediksi ML (Tingkat ASD), bukan dari input manual user.

#### Output Diagnosis DSM-5

| Total Skor | Diagnosis |
|---|---|
| <= 30 | **Autisme Berat (Level 3)** |
| 31 – 60 | **Autisme Sedang (Level 2)** |
| > 60 | **Autisme Ringan (Level 1)** |

### Tahap 6: Sanity Check Dual-Target (`app_pages/home.py:371`)

#### Validasi MLU vs Level Pemahaman

| Level | Rentang MLU |
|---|---|
| Belum Berkembang | 0 – 1 |
| Berkembang Sedang | 2 – 3 |
| Sudah Mahir | >= 4 |

Jika prediksi ML tidak sesuai rentang MLU → koreksi paksa ke level yang sesuai.

#### Validasi ASD vs Kategori Pemahaman

Memeriksa inkonsistensi antara output ASD dan Kategori Pemahaman berdasarkan anomali yang ditemukan di dataset (~21% mismatch). Koreksi diterapkan jika ditemukan kombinasi yang tidak masuk akal secara linguistik.

#### Validasi IKLA vs Level

- "Sudah Mahir" tapi IKLA <= 60 → turunkan ke "Berkembang Sedang"
- "Berkembang Sedang" tapi IKLA <= 30 → turunkan ke "Belum Berkembang"

### Tahap 7: Rekomendasi Terapis (`app_pages/home.py:203`)

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

Dataset dapat dieksplorasi secara interaktif melalui halaman **Dataset** pada aplikasi, yang menampilkan tabel lengkap dengan filtering serta visualisasi distribusi berdasarkan berbagai kategori (ASD, jenis kelamin, konteks, kompleksitas, intensi, MLU, dll).

**Sumber**: 65 anak autisme (A001–A065), 137 sampel ujaran (2–3 ujaran per anak)

Dataset final (`dataset/dataset.csv`) memiliki **14 kolom** sebagai berikut:

| # | Kolom | Deskripsi | Asal |
|---|---|---|---|
| 1 | `ID Anak` | Kode unik (A001–A065) | `dataset.csv` |
| 2 | `JK` | Jenis kelamin (L/P) | `dataset.csv` |
| 3 | `ASD` | Tingkat autisme (ASD-1/2/3) | `dataset.csv` |
| 4 | `Usia` | Usia dalam format tahun;bulan | `dataset.csv` |
| 5 | `Konteks` | Bermain, Percakapan, Bercerita, Deskripsi Gambar, Instruksi | `dataset.csv` |
| 6 | `Ujaran Anak` | Teks asli ujaran anak | `dataset.csv` |
| 7 | `Struktur Sintaksis` | Pola SPOK label manual | `dataset.csv` |
| 8 | `MLU` | Mean Length of Utterance (jumlah kata) | `dataset.csv` |
| 9 | `Echolalia` | Ya/Tidak | `dataset.csv` |
| 10 | `Ujaran Bersih` | Hasil preprocessing (case folding + cleaning) | generate script |
| 11 | `Token` | Hasil tokenisasi dalam bentuk list | generate script |
| 12 | `Kategori Pemahaman` | Belum Berkembang / Berkembang Sedang / Sudah Mahir | `dataset.csv` |
| 13 | `Kompleksitas Kalimat` | K1–K4 (rule-based dari struktur sintaksis) | generate script |
| 14 | `Intensi Komunikasi` | Protesting / Requesting / Answering / Commenting | generate script |

> **Catatan**: `dataset/dataset.csv` adalah file master (9 kolom dasar, 137 baris). `dataset/dataset.csv` di-generate secara otomatis dengan menambahkan 5 kolom derived (Ujaran Bersih, Token, Kompleksitas Kalimat, Intensi Komunikasi, label Neologisme "Belum Berkembang"). Aplikasi hanya membaca `dataset.csv`.

---

## Evaluasi Model (Dual-Target)

Halaman **Evaluasi Model** menyediakan analisis performa model **multi-output Random Forest** secara mendetail untuk kedua target:

| Fitur | Deskripsi |
|---|---|
| **Confusion Matrix (2 target)** | Heatmap per target — Kategori Pemahaman (3×3) dan Tingkat ASD (3×3) |
| **Cross-Validation** | 5-fold StratifiedGroupKFold (group by ID Anak) — akurasi per fold + rata-rata |
| **Classification Report** | Precision, recall, F1-score per kelas untuk masing-masing target |
| **Feature Importance** | Top 20 fitur paling penting dari ekstraksi Random Forest |

### Performa Model Final

| Target | Akurasi | F1 (macro) |
|---|---|---|
| Kategori Pemahaman | **94.16%** | **94.02%** |
| Tingkat ASD | **75.91%** | **74.88%** |

> Evaluasi menggunakan `StratifiedGroupKFold` agar ujaran dari anak yang sama tidak terpisah antar fold.

---

## Struktur File

```
├── app.py                          # Entry point — navigasi multipage + load model (cache via mtime)
├── app_pages/                      # Halaman-halaman aplikasi
│   ├── home.py                     # Halaman Analisis (form input + dashboard ML multi-output)
│   ├── dataset.py                  # Halaman Dataset (tabel + visualisasi data)
│   └── evaluation.py               # Halaman Evaluasi Model (dual-target CM + CV)
├── model_autism_syntax_rf.pkl      # Model ML multi-output terlatih (Random Forest)
├── model.ipynb                     # Notebook training model multi-output
├── version.py                      # Cek versi pustaka
├── requirements.txt                # Dependensi Python
├── README.md                       # Dokumentasi sistem
├── .streamlit/
│   └── config.toml                 # Tema Fluent (Light, Blue #0078D4)
└── dataset/
    ├── data_mentah.csv           # Dataset master (9 kolom dasar, 137 baris) — arsip
    └── dataset.csv                 # Dataset final (14 kolom, 137 sampel)
```

---

## Teknologi

| Komponen | Tools |
|---|---|
| **Framework** | Streamlit 1.58.0 |
| **Syntactic Parsing** | Rule-based Expanded Lexicon (~200 kata) + Sastrawi Stemmer |
| **Machine Learning** | scikit-learn 1.6.1 (MultiOutput Random Forest) |
| **Data Processing** | pandas 2.2.3 |
| **Model Serialization** | joblib >= 1.3 |
| **Text Processing** | regex (re) |
| **Anotasi Warna** | st-annotated-text |
| **Visualisasi** | matplotlib, Streamlit native charts |
| **Ikon** | Google Material Symbols |
| **Tema** | Fluent Design (Light, Blue #0078D4, Open Sans) |
| **Lingkungan** | Python 3.12 |
| **Deployment** | Streamlit Cloud |

---

## Tampilan & Tema

Aplikasi menggunakan tema **Fluent Design** (terinspirasi Windows 11) dengan:
- **Warna primer**: Biru `#0078D4`
- **Font**: Open Sans (body) + Fira Code (code)
- **Base font size**: 16px untuk keterbacaan optimal
- **Ikon**: Material Symbols untuk tombol, label, dan status

File konfigurasi tema: `.streamlit/config.toml`

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
