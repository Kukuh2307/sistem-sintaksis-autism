import streamlit as st
import pandas as pd
import re
from annotated_text import annotated_text
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_stemmer = None
def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer

PRONOUN_SUBJECT = {'aku', 'saya', 'kamu', 'dia', 'mama', 'papa'}

st.title("ATLAS (Automated Text & Language Assessment System)")
st.caption("Sistem Hibrida Komputasional Linguistik & Pendukung Keputusan")

model_ready = st.session_state.get("model_ready", False)
model_ai = st.session_state.get("model", None) if model_ready else None

def proses_teks_sintaksis(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def auto_parse_sintaksis(ujaran_bersih):
    token = ujaran_bersih.split()

    stemmer = _get_stemmer()

    kata_negasi = ['tidak', 'bukan', 'jangan', 'belum']
    kata_ket = ['kemarin', 'besok', 'tadi', 'sekarang', 'sore', 'pagi', 'sangat', 'cepat', 'di', 'ke', 'dari', 'sini', 'sana', 'epat', 'lagi', 'bersama', 'bawah', 'nanti', 'sudah', 'sama', 'selamat', 'halo', 'maaf',
                'besar', 'kecil', 'panjang', 'pendek', 'tinggi', 'berat', 'ringan', 'lambat', 'baru', 'lama',
                'banyak', 'sedikit', 'semua', 'beberapa', 'masing']
    kata_modal = ['mau', 'ingin', 'bisa', 'dapat', 'harus', 'akan']
    kata_predikat = ['minta', 'makan', 'minum', 'lari', 'main', 'duduk', 'lihat', 'putar', 'tidur', 'mandi', 'siram', 'baca', 'pergi',
                     'ajak', 'bawa', 'beli', 'buka', 'gambar', 'jalan', 'jatuh', 'ikut', 'kerja', 'masuk', 'naik', 'pakai',
                     'pulang', 'simpan', 'taruh', 'takut', 'suka', 'senang', 'ajar', 'rapi', 'sampai', 'habis', 'kasih', 'belikan']
    kata_nomina = ['aku', 'saya', 'kamu', 'dia', 'mama', 'papa', 'anak', 'mobil', 'bunga', 'sepeda', 'kucing', 'susu', 'air', 'buku', 'kebun', 'binatang', 'ini', 'itu',
                   'adik', 'ayah', 'ibu', 'bola', 'boneka', 'kereta', 'kuda', 'rumah', 'tas', 'pintu', 'tata', 'nama']
    kata_partikel = ['ayo', 'mari', 'yuk', 'ayuk']
    kata_ulang_terikat = ['kura', 'kupu', 'laba', 'ubur', 'cumi', 'anai', 'onde', 'pari']

    all_known = set(kata_negasi + kata_ket + kata_modal + kata_predikat + kata_nomina + kata_partikel + kata_ulang_terikat)

    if len(token) >= 4 and len(token) % 2 == 0 and token[:len(token)//2] == token[len(token)//2:]:
        return "Echolalia", []

    if len(token) >= 2 and len(set(token)) == 1:
        stem_base = stemmer.stem(token[0])
        if stem_base in all_known and stem_base not in kata_ulang_terikat:
            return "Repetisi", []

    stems = [stemmer.stem(t) for t in token]
    if all(s not in all_known for s in stems):
        is_bound_redup = True
        i = 0
        while i < len(token):
            if i + 1 < len(token) and stems[i] == stems[i + 1]:
                i += 2
            else:
                is_bound_redup = False
                break
        if not is_bound_redup:
            return "Neologisme", []

    pola_kasar = []
    has_predikat = False
    is_ket_phrase = False
    prev_stem = None

    for i, kata in enumerate(token):
        stem = stems[i]

        if stem == prev_stem and stem not in kata_negasi and pola_kasar:
            pola_kasar.append(pola_kasar[-1])
            continue

        prev_stem = stem
        if is_ket_phrase:
            pola_kasar.append("Ket")
            is_ket_phrase = False
            continue
        if stem in kata_negasi:
            pola_kasar.append("Negasi")
        elif stem in kata_ket:
            pola_kasar.append("Ket")
            if stem in ['di', 'ke', 'dari']:
                is_ket_phrase = True
        elif stem in kata_partikel:
            pola_kasar.append("Ket")
        elif stem in kata_modal:
            pola_kasar.append("P")
            has_predikat = True
        elif stem in kata_predikat:
            if has_predikat:
                pola_kasar.append("O")
            else:
                pola_kasar.append("P")
                has_predikat = True
        elif stem in PRONOUN_SUBJECT:
            if "S" not in pola_kasar:
                pola_kasar.append("S")
            else:
                pola_kasar.append("O")
        elif stem in kata_nomina:
            if not has_predikat:
                if "S" not in pola_kasar:
                    pola_kasar.append("S")
                else:
                    pola_kasar.append("P")
                    has_predikat = True
            else:
                pola_kasar.append("O")
        else:
            if not has_predikat:
                if "S" not in pola_kasar:
                    pola_kasar.append("S")
                else:
                    pola_kasar.append("P")
                    has_predikat = True
            else:
                pola_kasar.append("O")

    pola_final = []
    for p in pola_kasar:
        if not pola_final or pola_final[-1] != p:
            pola_final.append(p)

    return ("+".join(pola_final) if pola_final else "Tidak Teridentifikasi"), pola_kasar

KATA_VERBA = [
    'angkat', 'ambil', 'baca', 'bawa', 'beli', 'beri', 'buat', 'buka',
    'cuci', 'duduk', 'ganti', 'habis', 'hitung', 'ikut', 'jatuh', 'jemput',
    'jual', 'kasih', 'lari', 'lempar', 'lihat', 'lukis', 'main', 'makan', 'mandi',
    'masak', 'minum', 'minta', 'naik', 'pakai', 'panggil', 'pegang', 'pergi',
    'pukul', 'pulang', 'putar', 'robek', 'rusak', 'simpan', 'siram', 'taruh',
    'tendang', 'tidur', 'tulis', 'tusuk', 'tutup',
    'ajak', 'jalan', 'kerja', 'masuk', 'suka', 'senang', 'takut', 'sampai',
    'gambar', 'belikan', 'balap', 'milik', 'jumpa', 'jawab',
]

KATA_NOMINA = [
    'adik', 'air', 'aku', 'anak', 'ayah', 'baju', 'balon', 'batang',
    'benda', 'beras', 'bibi', 'binatang', 'biskuit', 'bola', 'boneka',
    'botol', 'buku', 'bunga', 'celana', 'dia', 'dompet', 'gelas',
    'handuk', 'ibu', 'ikat', 'ini', 'itu', 'jendela', 'kakak', 'kalung',
    'kamar', 'kamu', 'kebun', 'kemeja', 'kereta', 'kertas', 'kipas',
    'komik', 'kotak', 'krayon', 'kucing', 'kursi', 'lampu', 'layar',
    'lemari', 'lilin', 'mama', 'meja', 'mobil', 'motor', 'paman',
    'papa', 'papan', 'payung', 'pensil', 'piring', 'radio', 'sapu',
    'saya', 'selimut', 'semangka', 'sendok', 'sepeda', 'sepatu', 'sisir',
    'spidol', 'sumpit', 'susu', 'tali', 'tamu', 'tangga', 'tape',
    'televisi', 'tenda', 'topi',
    'apel', 'burung', 'gajah', 'ikan', 'kaki', 'kuda', 'mata',
    'mi', 'nama', 'pesawat', 'petir', 'pintu', 'roti', 'rumah',
    'sekolah', 'suara', 'tas', 'teman', 'mainan', 'kabar',
    'kodok', 'tata',
]

KATA_ULANG_TERIKAT = ['kura', 'kupu', 'laba', 'ubur', 'cumi', 'anai', 'onde', 'pari']

KATA_KET = [
    'baru', 'bersama', 'besok', 'belum', 'cepat', 'cukup', 'dekat',
    'di', 'dari', 'dulu', 'hampir', 'hanya', 'jarang', 'kadang',
    'ke', 'kemarin', 'nanti', 'pagi', 'sana', 'sangat', 'sekarang', 'selalu',
    'sedang', 'sering', 'sini', 'sore', 'sudah', 'tetap', 'tadi',
    'terlalu', 'epat', 'lagi', 'bawah', 'sama', 'selamat', 'halo', 'maaf',
    'besar', 'kecil', 'panjang', 'pendek', 'tinggi', 'berat', 'ringan', 'lambat', 'lama',
    'banyak', 'sedikit', 'semua', 'beberapa', 'masing',
]

KATA_PARTIKEL = ['ayo', 'mari', 'yuk', 'ayuk']

KATA_NEGASI = ['tidak', 'bukan', 'jangan', 'belum', 'tak', 'tiada']
KATA_MODAL = ['mau', 'ingin', 'bisa', 'dapat', 'harus', 'akan']

def parse_sintaksis_lanjutan(ujaran_bersih):
    token = ujaran_bersih.split()

    stemmer = _get_stemmer()

    all_known = set(KATA_VERBA + KATA_NOMINA + KATA_KET + KATA_NEGASI + KATA_MODAL + KATA_PARTIKEL + KATA_ULANG_TERIKAT)

    if len(token) >= 4 and len(token) % 2 == 0 and token[:len(token)//2] == token[len(token)//2:]:
        return "Echolalia", ["Echolalia"] * len(token)

    if len(token) >= 2 and len(set(token)) == 1:
        stem_base = stemmer.stem(token[0])
        if stem_base in all_known and stem_base not in KATA_ULANG_TERIKAT:
            return "Repetisi", ["Repetisi"] * len(token)

    stems = [stemmer.stem(t) for t in token]
    if all(s not in all_known for s in stems):
        is_bound_redup = True
        i = 0
        while i < len(token):
            if i + 1 < len(token) and stems[i] == stems[i + 1]:
                i += 2
            else:
                is_bound_redup = False
                break
        if not is_bound_redup:
            return "Neologisme", ["Neologisme"] * len(token)

    token_roles = []
    has_predikat = False
    is_ket_phrase = False
    prev_stem = None

    for i, kata in enumerate(token):
        stem = stems[i]

        if stem == prev_stem and stem not in KATA_NEGASI and token_roles:
            token_roles.append(token_roles[-1])
            continue

        prev_stem = stem
        if is_ket_phrase:
            token_roles.append("Ket")
            is_ket_phrase = False
            continue
        if stem in KATA_NEGASI:
            token_roles.append("Negasi")
        elif stem in KATA_KET:
            token_roles.append("Ket")
            if stem in ('di', 'ke', 'dari'):
                is_ket_phrase = True
        elif stem in KATA_PARTIKEL:
            token_roles.append("Ket")
        elif stem in KATA_MODAL:
            token_roles.append("P")
            has_predikat = True
        elif stem in KATA_VERBA:
            token_roles.append("P")
            has_predikat = True
        elif stem in PRONOUN_SUBJECT:
            if "S" not in token_roles:
                token_roles.append("S")
            else:
                token_roles.append("O")
        elif stem in KATA_NOMINA:
            if not has_predikat and ("S" not in token_roles or token_roles[-1] == "S"):
                token_roles.append("S")
            else:
                token_roles.append("O")
        else:
            if not has_predikat:
                if "S" not in token_roles:
                    token_roles.append("S")
                else:
                    token_roles.append("P")
                    has_predikat = True
            else:
                token_roles.append("O")

    pola_final = []
    for p in token_roles:
        if not pola_final or pola_final[-1] != p:
            pola_final.append(p)

    return ("+".join(pola_final) if pola_final else "Tidak Teridentifikasi"), token_roles

def tentukan_kompleksitas(struktur_sintaksis):
    struktur = str(struktur_sintaksis).strip().lower()
    if '+' not in struktur or struktur in ['echolalia', 'repetisi', 'negasi', 'tidak teridentifikasi', 'neologisme']:
        return 'Kata Tunggal (K1)'
    elif struktur.count('+') == 1:
        return 'Frasa (K2)'
    elif struktur.count('+') >= 3:
        return 'Kalimat Majemuk (K4)'
    else:
        return 'Kalimat Sederhana (K3)'

def deteksi_abnormal_order(token_roles):
    if not token_roles or len(token_roles) < 2:
        return False
    p_idx = None
    s_idx = None
    for i, r in enumerate(token_roles):
        if r == "P" and p_idx is None:
            p_idx = i
        if r == "S" and s_idx is None:
            s_idx = i
    return p_idx is not None and s_idx is not None and p_idx < s_idx

def deteksi_missing_subject(token_roles, mlu):
    if mlu < 3:
        return False
    return "S" not in token_roles and "P" in token_roles

def deteksi_interupsi_predikat(token_roles):
    for i in range(1, len(token_roles) - 1):
        if token_roles[i-1] == "P" and token_roles[i] == "Ket" and token_roles[i+1] == "P":
            return True
    return False

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

def hitung_ikla(mlu, kompleksitas, intensi, echolalia, konteks, token_len, tingkat_asd, is_neologism=False, is_abnormal_order=False, is_repetition=False, is_missing_subject=False, is_interupsi=False):
    if kompleksitas == 'Kata Tunggal (K1)':
        skor_sintaksis = 8
    elif kompleksitas == 'Frasa (K2)' or kompleksitas == 'Kalimat Sederhana (K3)' and mlu == 2:
        skor_sintaksis = 16
    elif kompleksitas == 'Kalimat Majemuk (K4)' or kompleksitas == 'Kalimat Sederhana (K3)' and mlu >= 3:
        skor_sintaksis = 24
    else:
        skor_sintaksis = 8

    if is_neologism:
        skor_sintaksis = 2
        skor_leksikal = 5
    elif is_interupsi:
        skor_sintaksis = max(2, int(skor_sintaksis / 1.5))
        skor_leksikal = 10 if token_len >= 2 else 5
    elif is_abnormal_order:
        skor_sintaksis = max(2, skor_sintaksis // 2)
        skor_leksikal = 16 if token_len > 3 else (10 if token_len >= 2 else 5)
    elif is_missing_subject:
        skor_sintaksis = max(2, int(skor_sintaksis / 1.5))
        skor_leksikal = min(10, skor_leksikal)
    else:
        skor_leksikal = 16 if token_len > 3 else (10 if token_len >= 2 else 5)

    skor_pragmatik = 20 if intensi in ['Commenting', 'Answering'] else (13 if intensi == 'Requesting' else 9)
    skor_echo = 12 if echolalia == 'Tidak' else 3
    skor_inisiasi = 8 if konteks in ['Bermain', 'Bercerita'] else (5 if konteks == 'Percakapan' else 3)
    skor_asd = 10 if tingkat_asd == 'ASD-1' else (6 if tingkat_asd == 'ASD-2' else 2)

    total_skor = skor_sintaksis + skor_leksikal + skor_pragmatik + skor_echo + skor_inisiasi + skor_asd

    if is_neologism:
        total_skor = min(total_skor, 30)
    if is_repetition and tingkat_asd == "ASD-3":
        total_skor = min(total_skor, 30)
    if is_abnormal_order:
        total_skor = min(total_skor, 60)
    if is_interupsi:
        total_skor = min(total_skor, 60)
    if is_missing_subject:
        total_skor = min(total_skor, 70)

    if total_skor <= 30:
        level_asd = "Autisme Berat (DSM-5 Level 3)"
    elif total_skor <= 60:
        level_asd = "Autisme Sedang (DSM-5 Level 2)"
    else:
        level_asd = "Autisme Ringan (DSM-5 Level 1)"
    return round(total_skor, 2), level_asd, {
        "Sintaksis": (skor_sintaksis, 24),
        "Leksikal": (skor_leksikal, 16),
        "Pragmatik": (skor_pragmatik, 20),
        "Echolalia": (skor_echo, 12),
        "Inisiasi": (skor_inisiasi, 8),
        "ASD Level": (skor_asd, 10),
    }

KAMUS_REKOMENDASI = {
    'Belum Berkembang': {
        'Protesting': {
            'target': 'Ekspresi Penolakan Fungsional',
            'metode': 'PECS (Picture Exchange Communication System) — kartu bergambar "tidak/saya tidak mau"',
            'contoh': 'Saat anak menolak, tunjukkan kartu "tidak" sambil geleng kepala. Ulangi tiap kali anak menunjukkan resistensi.',
            'frekuensi': '5-7 sesi awal, 10-15 menit/sesi, 3-4x seminggu',
            'indikator': 'Anak mampu menunjuk/memberi kartu "tidak" secara mandiri saat menolak'
        },
        'Requesting': {
            'target': 'Produksi Vokal untuk Meminta',
            'metode': 'Teknik Manding — tangkap setiap vokal yang diproduksi anak dan hubungkan dengan objek yang diinginkan',
            'contoh': 'Anak melihat biskuit → tunggu vokal (/"ah"/) → beri biskuit sambil ucapkan "mau"',
            'frekuensi': 'Setiap sesi, 10-15 peluang per sesi',
            'indikator': 'Anak memproduksi minimal 2 vokal berbeda secara konsisten untuk meminta'
        },
        'Commenting': {
            'target': 'Joint Attention & Imitasi Verbal',
            'metode': 'Mainan sensorik (gelembung, lampu, tekstur) untuk menarik perhatian bersama',
            'contoh': 'Tiup gelembung, tiru suara "bu..bu..bu" saat gelembung pecah. Tunggu anak meniru.',
            'frekuensi': '2-3 sesi/hari, 5-10 menit',
            'indikator': 'Anak menatap objek yang sama dengan terapis minimal 3 detik'
        },
        'Answering': {
            'target': 'Respon Verbal terhadap Pertanyaan Ya/Tidak',
            'metode': 'Imitasi vokal "Ya" dan "Tidak" disertai gestur angguk/geleng',
            'contoh': 'Tanya "Mau?" sambil angguk → stimulus vokal "ya". Beri objek saat anak meniru.',
            'frekuensi': '5-8x per sesi, 3-4 sesi/minggu',
            'indikator': 'Anak merespon dengan vokal "ya" atau "tidak" secara kontekstual 60% dari kesempatan'
        }
    },
    'Berkembang Sedang': {
        'Protesting': {
            'target': 'Ekspresi Penolakan Verbal 2 Kata',
            'metode': 'Frasa penolakan "tidak mau" + jadwal visual transisi aktivitas',
            'contoh': 'Tunjukkan jadwal: "sekarang bermain, nanti mandi". Saat anak protes, bantu ucapkan "tidak mau" sambil tunjuk jadwal.',
            'frekuensi': 'Setiap transisi, 3-4x sehari',
            'indikator': 'Anak mengucapkan "tidak mau" tanpa prompting 50% dari kesempatan'
        },
        'Requesting': {
            'target': 'Perluasan Kalimat Permintaan',
            'metode': 'Ekspansi dari 1 kata → frasa 2 kata menggunakan model + delay',
            'contoh': 'Anak: "mau" → Terapis: "mau apa?" + tunjuk gambar. Target: "mau susu", "mau main"',
            'frekuensi': '15-20 peluang per sesi',
            'indikator': 'Anak memproduksi minimal 5 frasa 2 kata berbeda untuk meminta'
        },
        'Commenting': {
            'target': 'Melengkapi Kalimat & Labeling',
            'metode': 'Teknik Cloze — bacakan kalimat dengan bagian hilang pada buku bergambar',
            'contoh': '"Ini namanya... (kucing)". Tunggu 5 detik. Jika tidakrespon, beri pilihan: "kucing atau mobil?"',
            'frekuensi': '3-5 halaman per sesi, 2 sesi/hari',
            'indikator': 'Anak melengkapi 60% kalimat cloze dengan benar'
        },
        'Answering': {
            'target': 'Menjawab Pertanyaan Apa & Siapa',
            'metode': 'Visual support — kartu bergambar + prompt bertingkat (verbal → gestur → mandiri)',
            'contoh': 'Tunjuk gambar anak sedang makan. Tanya "Apa yang dilakukan?" Beri pilihan: "makan / tidur"',
            'frekuensi': '10-12 pertanyaan per sesi',
            'indikator': 'Menjawab 70% pertanyaan "Apa" dan "Siapa" dengan benar'
        }
    },
    'Sudah Mahir': {
        'Protesting': {
            'target': 'Ekspresi Alasan Penolakan Logis',
            'metode': 'Stimulasi kalimat sebab-akibat menggunakan karena/sebab',
            'contoh': 'Anak: "tidak mau" → Terapis: "tidak mau karena...?" Bantu: "karena sudah capek"',
            'frekuensi': '3-4 kesempatan per sesi',
            'indikator': 'Anak memberikan alasan penolakan spontan dengan "karena" 70%'
        },
        'Requesting': {
            'target': 'Permintaan Struktur Lengkap + Santun',
            'metode': 'Pemodelan kalimat lengkap: "Saya mau [objek] tolong" + role play',
            'contoh': 'Role play jual-beli. Anak: "saya mau" → Terapis modelkan: "saya mau beli buku, tolong"',
            'frekuensi': '5-10 sesi bermain peran',
            'indikator': 'Anak menggunakan kalimat >4 kata dengan unsur kesantunan'
        },
        'Commenting': {
            'target': 'Narasi & Klausa Bertingkat',
            'metode': 'Story Retelling — anak menceritakan kembali cerita yang didengar dengan urutan logis',
            'contoh': 'Baca cerita 3-4 kalimat. Minta anak ceritakan ulang. Prompt: "pertama... lalu... akhirnya..."',
            'frekuensi': '1-2 cerita per sesi',
            'indikator': 'Anak menceritakan ulang dengan 3 klausa berurutan'
        },
        'Answering': {
            'target': 'Pertanyaan Kausal & Hipotetis',
            'metode': 'Stimulasi "Mengapa" dan "Bagaimana" — hubungkan dengan pengalaman konkret anak',
            'contoh': '"Mengapa kita pakai jaket?" "Bagaimana cara membuat susu?" Biarkan anak menjelaskan proses.',
            'frekuensi': '4-5 pertanyaan terbuka per sesi',
            'indikator': 'Anak menjawab "mengapa" dan "bagaimana" dengan logika yang relevan'
        }
    }
}

REKOMENDASI_KOMPONEN = {
    'Sintaksis': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Struktur kalimat sangat terbatas. "
            "Latih perluasan dari 1 kata → 2 kata: beri model S-P (\"mama tidur\"), "
            "kemudian S-P-O (\"aku mau susu\"). Targetkan +1 kata per 2 minggu."
        ),
    },
    'Leksikal': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Variasi kosakata rendah. "
            "Ekspos 3-5 kosakata baru per sesi lewat buku bergambar tematik. "
            "Gunakan teknik \"model & pause\": ucapkan kata baru, tunggu anak meniru."
        ),
    },
    'Pragmatik': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Fungsi komunikasi pragmatik terbatas. "
            "Fokus pada turn-taking: bergantian mainan 3-5 giliran. "
            "Latih kontak mata sebelum memberikan objek yang diminta."
        ),
    },
    'Echolalia': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Echolalia/repetisi signifikan. "
            "Terapkan intervensi potong-rantai: sela pola repetisi dengan prompt baru. "
            "Ajarkan respon fungsional \"Ya\" / \"Tidak\" sebagai alternatif echolalia."
        ),
    },
    'Inisiasi': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Inisiasi komunikasi rendah. "
            "Ciptakan situasi yang membutuhkan anak memulai komunikasi "
            "(letakkan mainan favorit di tempat tinggi, tunggu anak meminta tolong). "
            "Berikan reinforcement segera saat anak berinisiatif."
        ),
    },
    'ASD Level': {
        'kondisi': lambda v, maks: v / maks < 0.5,
        'rekomendasi': lambda v, maks: (
            f"**{v}/{maks}** — Baseline ASD rendah (ASD-3). "
            "Intervensi perlu disesuaikan dengan tingkat keparahan: "
            "fokus pada komunikasi dasar (PECS/isyarat) dan pengurangan "
            "perilaku repetitif sebelum menargetkan ekspansi linguistik."
        ),
    },
}

def gen_rekomendasi_tambahan(rincian_skor, mlu, echolalia):
    hasil = []
    for komponen, (skor, maks) in rincian_skor.items():
        entry = REKOMENDASI_KOMPONEN.get(komponen)
        if entry and entry['kondisi'](skor, maks):
            hasil.append((komponen, skor, maks, entry['rekomendasi'](skor, maks)))

    if mlu <= 2:
        hasil.append(("MLU", None, None,
            f"**MLU {mlu} kata** — Rerata panjang ujaran rendah. "
            "Target ekspansi: modelkan kalimat +1 kata dari yang diproduksi anak. "
            "Misal: anak \"mau\" → model \"aku mau\"."))

    if echolalia == "Ya":
        ada_echo = any(k == "Echolalia" for k, _, _, _ in hasil)
        if not ada_echo:
            hasil.append(("Echolalia", None, None,
                "**Echolalia terdeteksi** — Terapkan teknik potong-rantai: "
                "saat anak mengulang kata, beri prompt baru yang mengubah arah pembicaraan."))

    return hasil

def sanity_check_prediksi(prediksi_pemahaman, prediksi_asd, mlu, skor_ikla, is_missing_subject=False):
    korreksi_p = prediksi_pemahaman
    korreksi_a = prediksi_asd
    alasan = []

    batas_mlu_p = {
        "Belum Berkembang": (0, 1),
        "Berkembang Sedang": (2, 3),
        "Sudah Mahir": (4, float("inf")),
    }
    for level, (min_kata, max_kata) in batas_mlu_p.items():
        if min_kata <= mlu <= max_kata:
            if prediksi_pemahaman != level:
                korreksi_p = level
                alasan.append(f"MLU ({mlu}) tidak sesuai untuk '{prediksi_pemahaman}', dikoreksi ke '{level}'")
            break

    if mlu <= 1 and prediksi_asd == "ASD-1":
        korreksi_a = "ASD-2"
        alasan.append(f"MLU ({mlu}) terlalu rendah untuk ASD-1, dikoreksi ke ASD-2")
    elif mlu >= 4 and prediksi_asd == "ASD-3":
        korreksi_a = "ASD-2"
        alasan.append(f"MLU ({mlu}) terlalu tinggi untuk ASD-3, dikoreksi ke ASD-2")

    if is_missing_subject and mlu >= 3 and korreksi_p == "Sudah Mahir":
        korreksi_p = "Berkembang Sedang"
        alasan.append("Subjek tidak ditemukan pada kalimat ≥3 kata, dikoreksi ke 'Berkembang Sedang'")
    elif korreksi_p == "Sudah Mahir" and skor_ikla <= 60:
        korreksi_p = "Berkembang Sedang"
        alasan.append(f"Skor IKLA ({skor_ikla}) tidak mendukung 'Sudah Mahir', dikoreksi ke 'Berkembang Sedang'")
    elif korreksi_p == "Berkembang Sedang" and skor_ikla <= 30:
        korreksi_p = "Belum Berkembang"
        alasan.append(f"Skor IKLA ({skor_ikla}) tidak mendukung 'Berkembang Sedang', dikoreksi ke 'Belum Berkembang'")

    return korreksi_p, korreksi_a, "; ".join(alasan) if alasan else None

col_input, col_output = st.columns([1, 1.5])

with col_input:
    st.header(":material/quick_reference: Form Analisis Cepat")
    st.caption("Input disederhanakan hanya pada parameter klinis yang wajib.")

    with st.container(border=True):
        konteks = st.selectbox(":material/dashboard: Konteks Interaksi", ["Percakapan", "Bermain", "Bercerita", "Deskripsi Gambar", "Instruksi"])

        ujaran_anak = st.text_area(":material/edit_note: Teks Ujaran Anak", placeholder="Ketikkan ujaran anak di sini...", height=100)

        tombol_analisis = st.button("Auto-Parse & Analisis", type="primary", width="stretch")

with col_output:
    st.header(":material/dashboard: Dashboard Diagnosis")

    if tombol_analisis and ujaran_anak:
        with st.spinner("Mesin NLP sedang membedah kalimat..."):
            ujaran_bersih = proses_teks_sintaksis(ujaran_anak)
            token_list = ujaran_bersih.split()
            mlu_hitung = len(token_list)

            struktur_sintaksis_otomatis, pola_per_token = auto_parse_sintaksis(ujaran_bersih)

            struktur_lanjutan, pola_lanjutan = parse_sintaksis_lanjutan(ujaran_bersih)

            kompleksitas_kalimat = tentukan_kompleksitas(struktur_sintaksis_otomatis)
            intensi_komunikasi = tentukan_intensi(ujaran_bersih, konteks)

            is_neologism = struktur_sintaksis_otomatis == "Neologisme"
            is_abnormal_order = deteksi_abnormal_order(pola_lanjutan)
            is_interupsi = deteksi_interupsi_predikat(pola_lanjutan)
            has_partikel = any(w in KATA_PARTIKEL for w in token_list)
            is_missing_subject = deteksi_missing_subject(pola_lanjutan, mlu_hitung) and not has_partikel
            is_repetition = struktur_lanjutan == "Repetisi"
            is_echolalia = struktur_lanjutan == "Echolalia"
            echolalia_efektif = "Ya" if (is_repetition or is_echolalia) else "Tidak"

            if model_ready and model_ai is not None:
                data_input_ml = pd.DataFrame([{
                    'Ujaran Bersih': ujaran_bersih,
                    'Echolalia': echolalia_efektif,
                    'Struktur Sintaksis': struktur_sintaksis_otomatis,
                    'Kompleksitas Kalimat': kompleksitas_kalimat,
                    'Intensi Komunikasi': intensi_komunikasi,
                    'MLU': mlu_hitung
                }])
                preds = model_ai.predict(data_input_ml)
                prediksi_pemahaman = preds[0][0]
                prediksi_asd = preds[0][1]
            else:
                prediksi_pemahaman = "Belum Berkembang" if mlu_hitung <= 1 else "Sudah Mahir"
                prediksi_asd = "ASD-3" if mlu_hitung <= 1 else "ASD-1"

            if is_interupsi or is_abnormal_order or is_repetition:
                prediksi_pemahaman = "Belum Berkembang"
                prediksi_asd = "ASD-3"

            skor_ikla, label_diagnosis, rincian_skor = hitung_ikla(mlu_hitung, kompleksitas_kalimat, intensi_komunikasi, echolalia_efektif, konteks, mlu_hitung, prediksi_asd, is_neologism=is_neologism, is_abnormal_order=is_abnormal_order, is_repetition=is_repetition, is_missing_subject=is_missing_subject, is_interupsi=is_interupsi)

            prediksi_pemahaman, prediksi_asd, warning_sanity = sanity_check_prediksi(prediksi_pemahaman, prediksi_asd, mlu_hitung, skor_ikla, is_missing_subject=is_missing_subject)

            rekomendasi_utama = KAMUS_REKOMENDASI.get(prediksi_pemahaman, {}).get(intensi_komunikasi, None)
            rekomendasi_tambahan = gen_rekomendasi_tambahan(rincian_skor, mlu_hitung, echolalia_efektif)

            st.toast("Analisis Selesai!", icon=":material/check_circle:")

            WARNA_ROLE = {
                "S": ("#3B82F6", "white"),
                "P": ("#10B981", "white"),
                "O": ("#F59E0B", "white"),
                "Ket": ("#8B5CF6", "white"),
                "Negasi": ("#EF4444", "white"),
                "Echolalia": ("#6B7280", "white"),
                "Repetisi": ("#6B7280", "white"),
                "Neologisme": ("#9B59B6", "white"),
            }

            st.subheader("1. Pembedahan Sintaksis (Parser)")

            if pola_lanjutan and len(pola_lanjutan) == len(token_list):
                annotated_args = []
                for i, kata in enumerate(token_list):
                    role = pola_lanjutan[i]
                    bg, fg = WARNA_ROLE.get(role, ("#E5E7EB", "#111827"))
                    annotated_args.append((kata, role, bg, fg))
                    if i < len(token_list) - 1:
                        annotated_args.append(" ")
                annotated_text(*annotated_args)
            elif struktur_lanjutan in ("Echolalia", "Repetisi", "Neologisme"):
                label = struktur_lanjutan
                bg, fg = WARNA_ROLE.get(label, ("#6B7280", "white"))
                annotated_args = []
                for i, kata in enumerate(token_list):
                    annotated_args.append((kata, label, bg, fg))
                    if i < len(token_list) - 1:
                        annotated_args.append(" ")
                annotated_text(*annotated_args)
            else:
                st.code(f"Pola: {struktur_lanjutan}", language="markdown")

            st.caption(f"**Pola:** {struktur_lanjutan}  |  **Kompleksitas:** {kompleksitas_kalimat}  |  **MLU:** {mlu_hitung} kata")
            if is_echolalia:
                st.caption(":material/record_voice_over: **Echolalia terdeteksi** — Pola pengulangan frasa (ABAB) teridentifikasi secara otomatis.")

            st.subheader("2. Prediksi Kognitif & Pragmatik")
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Pemahaman:** {prediksi_pemahaman}", icon=":material/psychology:")
            c2.info(f"**ASD:** {prediksi_asd}", icon=":material/neurology:")
            c3.info(f"**Intensi:** {intensi_komunikasi}", icon=":material/chat:")
            if warning_sanity:
                st.caption(f":material/tune: Sanity check: {warning_sanity}")

            with st.expander("Skor Indeks Keparahan Linguistik (IKLA DSM-5)", expanded=True, icon=":material/analytics:"):
                pct = skor_ikla / 90 * 100
                if pct <= 33:
                    bar_color = "#e53e3e"
                elif pct <= 66:
                    bar_color = "#D69E2E"
                else:
                    bar_color = "#38a169"

                col_metrik, col_gauge = st.columns([1, 1.5])
                with col_metrik:
                    st.metric(label=":material/score: Total Skor IKLA", value=f"{skor_ikla} / 90")
                with col_gauge:
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:white;margin-bottom:0.25rem;">
                            <span>0</span>
                            <span style="font-weight:600;">{skor_ikla} / 90 ({pct:.0f}%)</span>
                            <span>90</span>
                        </div>
                        <div style="height:14px;background:#edf2f7;border-radius:10px;overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{bar_color},#2C7A7B);border-radius:10px;transition:width 0.6s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**Indikasi Klinis:** {label_diagnosis}")

                df_sub = pd.DataFrame([
                    {"Komponen": k, "Skor": v[0], "Maks": v[1]}
                    for k, v in rincian_skor.items()
                ]).set_index("Komponen")
                st.subheader(":material/bar_chart: Rincian Skor per Komponen")
                st.bar_chart(df_sub, y="Skor", stack=False, height=250)

            st.subheader("3. Rekomendasi Sistem")

            if rekomendasi_utama:
                r = rekomendasi_utama

                data_rekomendasi = {
                    "Kategori": ["🎯 Target", "📌 Metode", "💬 Contoh", "📊 Frekuensi", "✅ Indikator"],
                    "Detail": [r['target'], r['metode'], r['contoh'], r['frekuensi'], r['indikator']]
                }
                df_rekomendasi = pd.DataFrame(data_rekomendasi)

                st.dataframe(
                    df_rekomendasi,
                    hide_index=True,
                    column_config={
                        "Kategori": st.column_config.TextColumn("Kategori", width="small"),
                        "Detail": st.column_config.TextColumn("Detail", width="large")
                    }
                )

                if rekomendasi_tambahan:
                    st.markdown("**Catatan Rekomendasi Tambahan:**")
                    for label, skor, maks, teks in rekomendasi_tambahan:
                        skor_str = f" ({skor}/{maks})" if skor is not None else ""
                        with st.expander(f":material/warning: {label}{skor_str}"):
                            st.write(teks)

        if echolalia_efektif == "Ya":
            st.error("Catatan Klinis: Terapkan intervensi potong-rantai untuk mengatasi repetisi Echolalia.", icon=":material/medical_services:")

    elif tombol_analisis and not ujaran_anak:
        st.error("Ujaran anak tidak boleh kosong!", icon=":material/error:")
    else:
        st.info("Form siap. Tekan tombol untuk melihat hasil analisis.", icon=":material/info:")
