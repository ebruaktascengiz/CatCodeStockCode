import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import ssl
import certifi

# SSL sertifika sorunu için geçici çözüm
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# .env dosyası
# Önce script'in bulunduğu dizindeki .env'yi dene
script_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(script_dir, '.env')

if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"✅ .env dosyası bulundu: {env_file}")
else:
    # Çalışma dizinindeki .env'yi dene
    load_dotenv()
    print("⚠️ .env dosyası script dizininde bulunamadı, çalışma dizininde aranıyor...")

# --- Konfigürasyon ---
api_key = os.getenv("GOOGLE_API_KEY")

# Tırnak işaretlerini temizle
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

# Eğer hala yoksa, kullanıcıdan al
if not api_key:
    print("\n" + "=" * 60)
    print("⚠️  GOOGLE_API_KEY bulunamadı!")
    print("=" * 60)
    print("Lütfen aşağıdaki seçeneklerden birini yapın:")
    print("1. .env dosyası oluşturun ve içine şunu ekleyin:")
    print("   GOOGLE_API_KEY=sizin_api_anahtariniz")
    print(f"   Dosya yolu: {env_file}")
    print("\n2. Veya şimdi API anahtarınızı girin:")
    print("=" * 60)

    api_key = input("Google API Key (boş bırakırsanız program kapanır): ").strip().strip('"').strip("'")

    if not api_key:
        raise ValueError("Google API anahtarı girilmedi. Program sonlandırılıyor.")

    # Kullanıcının girdiği anahtarı kaydet (opsiyonel)
    save_key = input("\nBu anahtarı .env dosyasına kaydetmek ister misiniz? (e/h): ").lower()
    if save_key == 'e':
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(f"GOOGLE_API_KEY={api_key}\n")
            print(f"✅ API anahtarı kaydedildi: {env_file}")
        except Exception as e:
            print(f"⚠️ Kaydetme hatası: {e}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Hızlı ve uygun maliyetli

xlsx_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ru2b4 - category codes.xlsx')

# Global değişkenler
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  # Bigram desteği
text_vectors = None
df_global = None
unique_categories_map = {}
valid_codes = set()
category_keywords = {}  # Her kategori için anahtar kelimeler


def extract_category_keywords():
    """
    Her kategori için en sık kullanılan kelimeleri çıkarır.
    Bu, AI'a daha iyi context vermek için kullanılır.
    """
    global category_keywords
    category_keywords = {}

    for code in valid_codes:
        category_data = df_global[df_global['CategoryCode'] == code]
        all_text = ' '.join(category_data['Description'].str.lower().fillna('').tolist())
        words = all_text.split()
        # En sık kullanılan 5 kelime
        common_words = [word for word, count in Counter(words).most_common(5) if len(word) > 2]
        category_keywords[code] = common_words


def update_learning_data():
    """
    Global DataFrame'i kullanarak TF-IDF vektörlerini yeniden hesaplar.
    Arama için Description ve CategoryName sütunlarını birleştirir.
    """
    global vectorizer, text_vectors, df_global

    df_global['SearchableText'] = (
            df_global['Description'].str.lower().fillna('') + ' ' +
            df_global['CategoryName'].str.lower().fillna('')
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    text_vectors = vectorizer.fit_transform(df_global['SearchableText'])

    extract_category_keywords()
    print("🧠 Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    """
    Veriyi yükleyip temizleyip, ilk öğrenme setini hazırlar.
    """
    global df_global, unique_categories_map, valid_codes
    try:
        df_global = pd.read_excel(file_path)
        df_global.dropna(subset=['Description', 'CategoryCode', 'CategoryName'], inplace=True)

        # Boşlukları temizle
        df_global['Description'] = df_global['Description'].str.strip()
        df_global['CategoryCode'] = df_global['CategoryCode'].str.strip()
        df_global['CategoryName'] = df_global['CategoryName'].str.strip()

        unique_categories_map = df_global.drop_duplicates(subset=['CategoryCode']).set_index('CategoryCode')[
            'CategoryName'].to_dict()
        valid_codes = set(df_global['CategoryCode'])

        update_learning_data()
        print(f"✅ {len(df_global)} adet geçerli örnekle öğrenim verisi hazırlandı.")
        print(f"📊 Toplam {len(valid_codes)} farklı kategori bulundu.")
        return True
    except FileNotFoundError:
        print(f"❌ HATA: Dosya bulunamadı: {file_path}")
        return False
    except Exception as e:
        print(f"❌ HATA: Veri hazırlanırken bir sorun oluştu: {e}")
        return False


def add_new_example(description, code):
    """
    Yeni ve doğrulanan bir örneği Excel dosyasına ve hafızadaki DataFrame'e ekler.
    """
    global df_global, valid_codes, unique_categories_map

    if code not in valid_codes:
        print(f"⚠️ Uyarı: '{code}' kodu mevcut kategori listesinde yok. Yeni kategori ekleniyor.")
        new_category_name = input(f"Lütfen '{code}' kodu için bir kategori adı girin: ").strip()
        if not new_category_name:
            new_category_name = "Yeni Eklenen Kategori"
        unique_categories_map[code] = new_category_name
        valid_codes.add(code)

    category_name = unique_categories_map.get(code)
    new_row = pd.DataFrame([{
        'Description': description,
        'CategoryCode': code,
        'CategoryName': category_name
    }])
    df_global = pd.concat([df_global, new_row], ignore_index=True)

    try:
        df_global.to_excel(xlsx_file_path, index=False, engine='openpyxl')
        print("💾 Yeni örnek Excel dosyasına kaydedildi.")
    except Exception as e:
        print(f"❌ HATA: Excel dosyasına yazılırken sorun oluştu: {e}")
        df_global.drop(df_global.tail(1).index, inplace=True)
        return

    update_learning_data()


def find_most_similar_examples(user_description: str, top_n: int = 5):
    """
    Birleşik arama metnini kullanarak en benzer N örneği bulur.
    """
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, text_vectors).flatten()

    if len(cosine_similarities) == 0 or cosine_similarities.max() < 0.05:
        return pd.DataFrame(), []

    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    similarity_scores = cosine_similarities[most_similar_indices]

    return df_global.iloc[most_similar_indices], similarity_scores


def get_top_candidate_categories(user_description: str, top_n: int = 3):
    """
    TF-IDF benzerliğine göre en olası kategorileri bulur.
    """
    similar_examples, scores = find_most_similar_examples(user_description, top_n=15)

    if similar_examples.empty:
        return []

    # En sık görülen kategorileri say
    category_counts = similar_examples['CategoryCode'].value_counts().head(top_n)
    return category_counts.index.tolist()


def get_smart_suggestion(user_description: str, category_list: dict):
    """
    Google Gemini AI'dan optimize edilmiş prompt ile öneri alır.
    """
    similar_examples, similarity_scores = find_most_similar_examples(user_description, top_n=5)

    # En olası 3-5 kategoriyi bul
    candidate_categories = get_top_candidate_categories(user_description, top_n=5)

    example_prompt_part = ""
    if not similar_examples.empty:
        print("\n🔎 Benzer örnekler bulundu:")
        example_texts = []
        for idx, (_, row) in enumerate(similar_examples.iterrows()):
            similarity_pct = similarity_scores[idx] * 100
            print(f"   [{similarity_pct:.1f}%] '{row['Description']}' → {row['CategoryCode']} ({row['CategoryName']})")
            example_texts.append(
                f"- Tanım: '{row['Description']}' | Kod: {row['CategoryCode']} | Kategori: {row['CategoryName']}"
            )

        example_prompt_part = (
                "Sistemdeki benzer örnekler:\n" + "\n".join(example_texts)
        )

    # Sadece en olası kategorileri AI'a gönder (performans için)
    if candidate_categories:
        focused_categories = {code: category_list[code] for code in candidate_categories if code in category_list}
        categories_info = "\n".join(
            f"• {code}: {name} (Anahtar kelimeler: {', '.join(category_keywords.get(code, []))})"
            for code, name in focused_categories.items()
        )
    else:
        # Fallback: tüm kategoriler
        categories_info = "\n".join(f"• {code}: {name}" for code, name in list(category_list.items())[:50])

    prompt_message = f"""Sen bir satınalma kategorizasyon uzmanısın. Görevin malzeme tanımını doğru kategoriye atamak.

YENI MALZEME TANIMI: "{user_description}"

{example_prompt_part}

EN OLASI KATEGORİLER:
{categories_info}

TALİMATLAR:
1. Yukarıdaki benzer örnekleri ve kategori anahtar kelimelerini analiz et
2. Yeni malzeme tanımı için EN UYGUN kategori kodunu seç
3. SADECE kategori kodunu döndür (örnek: ABC123)
4. Başka açıklama, noktalama veya metin yazma

CEVAP (sadece kod):"""

    try:
        print("\n🤖 Gemini AI'dan öneri alınıyor...")

        response = model.generate_content(
            prompt_message,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Düşük temperature = daha tutarlı
                max_output_tokens=20,
                top_p=0.95,
            )
        )

        suggested_code = response.text.strip().upper()
        # Temizlik: noktalama işaretleri, yeni satırlar vb. kaldır
        suggested_code = ''.join(c for c in suggested_code if c.isalnum() or c == '-')

        return suggested_code

    except Exception as e:
        print(f"⚠️ API Hatası: {e}")
        # Fallback: En benzer örneğin kodunu döndür
        if not similar_examples.empty:
            return similar_examples.iloc[0]['CategoryCode']
        return "API-HATA"


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    if preprocess_and_load_data(xlsx_file_path):
        print("=" * 60)
        print("🚀 KENDİ KENDİNİ GELİŞTİREN KATEGORİ ÖNERİ SİSTEMİ")
        print("   (Google Gemini AI ile güçlendirildi)")
        print("=" * 60)
        print("Komutlar: 'çıkış' veya 'exit' - Programdan çık")
        print("         'istatistik' - Sistem istatistiklerini göster")
        print("=" * 60)

        session_correct = 0
        session_total = 0

        while True:
            user_input = input("\n➡️  Malzeme tanımı girin: ").strip()

            if user_input.lower() in ['çıkış', 'cikis', 'exit', 'quit']:
                if session_total > 0:
                    accuracy = (session_correct / session_total) * 100
                    print(f"\n📊 Oturum İstatistikleri:")
                    print(f"   Toplam öneri: {session_total}")
                    print(f"   Doğru öneri: {session_correct}")
                    print(f"   Başarı oranı: {accuracy:.1f}%")
                print("\n💾 Değişiklikler kaydedildi. Görüşmek üzere! 👋")
                break

            if user_input.lower() == 'istatistik':
                print(f"\n📊 Sistem İstatistikleri:")
                print(f"   Toplam örnek: {len(df_global)}")
                print(f"   Kategori sayısı: {len(valid_codes)}")
                print(f"   En çok örneği olan kategoriler:")
                top_cats = df_global['CategoryCode'].value_counts().head(5)
                for code, count in top_cats.items():
                    print(f"      {code} ({unique_categories_map[code]}): {count} örnek")
                continue

            if not user_input:
                continue

            session_total += 1
            suggestion = get_smart_suggestion(user_input, unique_categories_map)

            print("\n" + "─" * 60)
            print(f"📝 Girilen: '{user_input}'")

            if suggestion == "API-HATA":
                print("⚠️  API ile iletişim kurulamadı.")
                session_total -= 1
                continue

            if suggestion in valid_codes:
                category_name = unique_categories_map[suggestion]
                print(f"🎯 ÖNERİ: {suggestion} - {category_name}")
            else:
                print(f"⚠️  ÖNERİ: {suggestion} (Geçersiz kod - kontrol edin)")

            # Geri bildirim döngüsü
            while True:
                feedback = input("\n✓ Doğru mu? (e/h): ").lower()
                if feedback == 'e':
                    print("✅ Harika! Sistem doğru önerdi.")
                    session_correct += 1
                    break
                elif feedback == 'h':
                    correct_code = input("🔧 Doğru kodu girin: ").strip().upper()
                    if not correct_code:
                        print("⚠️  Geçerli kod girilmedi. İşlem iptal.")
                        session_total -= 1
                        break

                    print("📚 Sistem öğreniyor...")
                    add_new_example(user_input, correct_code)
                    print("✅ Yeni bilgi kaydedildi!")
                    break
                else:
                    print("⚠️  Lütfen 'e' veya 'h' girin.")

            print("─" * 60)