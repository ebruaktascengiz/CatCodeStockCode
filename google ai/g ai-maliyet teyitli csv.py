import os
import pandas as pd
import openai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# .env dosyasındaki environment değişkenlerini yükle
load_dotenv()

# --- Konfigürasyon ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API anahtarı bulunamadı...")
client = openai.OpenAI(api_key=api_key)
# Kullanıcının verdiği dosya yolunu kullanıyoruz
csv_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ru2b8 - category codes.csv')

# Global değişkenler
vectorizer = TfidfVectorizer()
description_vectors = None
df_global = None
unique_categories_map = {}
valid_codes = set()


def update_learning_data():
    """
    Global DataFrame'i kullanarak TF-IDF vektörlerini yeniden hesaplar.
    Bu fonksiyon, yeni bir örnek eklendiğinde çağrılır.
    """
    global vectorizer, description_vectors, df_global
    # Veriyi küçük harfe çevirmek genellikle doğruluğu artırır
    df_global['Description_processed'] = df_global['Description'].str.lower().fillna('')
    vectorizer = TfidfVectorizer()
    description_vectors = vectorizer.fit_transform(df_global['Description_processed'])
    print("🧠 Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    """
    Veriyi yükler, temizler ve ilk öğrenme setini hazırlar.
    """
    global df_global, unique_categories_map, valid_codes
    try:
        df_global = pd.read_csv(file_path)
        df_global.dropna(subset=['Description', 'CategoryCode', 'CategoryName'], inplace=True)

        # Kategorilerin benzersiz listesini ve geçerli kodları oluştur
        unique_categories_map = df_global.drop_duplicates(subset=['CategoryCode']).set_index('CategoryCode')[
            'CategoryName'].to_dict()
        valid_codes = set(df_global['CategoryCode'])

        update_learning_data()  # İlk vektörleri oluştur
        print(f"✅ {len(df_global)} adet geçerli örnekle öğrenim verisi hazırlandı.")
        return True
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı: {file_path}")
        return False
    except Exception as e:
        print(f"HATA: Veri hazırlanırken bir sorun oluştu: {e}")
        return False


def add_new_example(description, code):
    """
    Yeni ve doğru bir örneği CSV dosyasına ve hafızadaki DataFrame'e ekler.
    """
    global df_global, valid_codes, unique_categories_map

    if code not in valid_codes:
        print(f"⚠️ Uyarı: Girdiğiniz '{code}' kodu mevcut kategori listesinde yok. Yine de ekleniyor.")
        # Eğer kod yeni ise, kategori adı için bir varsayımda bulunalım
        unique_categories_map[code] = "Yeni Eklenen Kategori"
        valid_codes.add(code)

    category_name = unique_categories_map.get(code)

    new_row = pd.DataFrame([{'Description': description, 'CategoryCode': code, 'CategoryName': category_name}])

    # 1. Kalıcı olarak CSV'ye ekle (append modu, header olmadan)
    try:
        new_row.to_csv(csv_file_path, mode='a', header=False, index=False, encoding='utf-8')
    except Exception as e:
        print(f"HATA: CSV dosyasına yazılırken sorun oluştu: {e}")
        return

    # 2. Anında öğrenme için hafızadaki DataFrame'e ekle
    df_global = pd.concat([df_global, new_row], ignore_index=True)

    # 3. Öğrenim verisini yeni bilgiyle güncelle
    update_learning_data()


def find_most_similar_examples(user_description: str, top_n: int = 5):
    # (Bu fonksiyon öncekiyle aynı, değişiklik yok)
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, description_vectors).flatten()
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    if cosine_similarities[most_similar_indices[0]] < 0.05:  # Eşiği biraz düşürdük
        return pd.DataFrame()
    return df_global.iloc[most_similar_indices]


def get_smart_suggestion(user_description: str):
    """
    Maliyet-etkin model ve optimize edilmiş prompt ile öneri alır.
    """
    similar_examples = find_most_similar_examples(user_description)

    example_prompt_part = ""
    if not similar_examples.empty:
        print("\n🔎 Öğrenim dosyasından benzer örnekler bulundu:")
        example_texts = [f"- '{row['Description']}' -> '{row['CategoryCode']}'" for _, row in
                         similar_examples.iterrows()]
        example_prompt_part = "İşte bazı benzer örnekler:\n" + "\n".join(example_texts)
        # Konsola da yazdıralım
        for text in example_texts:
            print(f"   {text}")

    # --- MALİYET OPTİMİZE EDİLMİŞ PROMPT ---
    # Genel kategori listesi kaldırıldı, talimatlar daha net hale getirildi.
    prompt_message = f"""
    Sen, verilen örneklere dayanarak malzeme tanımını doğru kategori koduyla eşleştiren bir asistansın.

    {example_prompt_part}

    Bu örneklere bakarak, aşağıdaki yeni tanım için en uygun kategori kodunu bul.
    Yeni Tanım: "{user_description}"

    Sadece ve sadece kategori kodunu döndür. Başka hiçbir şey yazma.
    """

    try:
        print("\n🧠 Akıllı yapay zekadan öneri isteniyor...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # MALİYET DÜŞÜRMEK İÇİN MODEL DEĞİŞTİRİLDİ
            messages=[{"role": "user", "content": prompt_message}],
            temperature=0,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API-HATA: {e}"


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    if preprocess_and_load_data(csv_file_path):
        print("-" * 50)
        print("Kendi Kendini Geliştiren Kategori Önerme Aracı")
        print("Çıkmak için 'çıkış' veya 'exit' yazabilirsiniz.")
        print("-" * 50)

        while True:
            user_input = input("\n➡️ Kategori önerisi için malzeme tanımını girin: ").strip()

            if user_input.lower() in ['çıkış', 'cikis', 'exit', 'quit']:
                print("Değişiklikler kaydedildi. Programdan çıkılıyor...")
                break
            if not user_input: continue

            suggestion = get_smart_suggestion(user_input)

            print("-" * 30)
            print(f"💬 Girilen Tanım: '{user_input}'")

            if suggestion.startswith("API-HATA:"):
                print(f"⚠️ Hata: {suggestion}")
                continue

            if suggestion in valid_codes:
                category_name = unique_categories_map.get(suggestion, "")
                print(f"🤖 ÖNERİ: {suggestion} ({category_name})")
            else:
                print(f"🤖 ÖNERİ (Geçersiz Kod): {suggestion}")

            # --- GERİ BİLDİRİM DÖNGÜSÜ ---
            while True:
                feedback = input("Bu öneri doğru mu? (e: evet / h: hayır): ").lower()
                if feedback == 'e':
                    print("👍 Harika! Bu bilgi sisteme ekleniyor...")
                    add_new_example(user_input, suggestion)
                    break
                elif feedback == 'h':
                    correct_code = input("Lütfen doğru kodu girin: ").strip().upper()
                    if correct_code in valid_codes:
                        print("📝 Anlaşıldı. Doğru bilgi sisteme ekleniyor...")
                        add_new_example(user_input, correct_code)
                    else:
                        print("Girdiğiniz kod sistemde yok ama yine de yeni bir bilgi olarak ekleniyor.")
                        add_new_example(user_input, correct_code)
                    break
                else:
                    print("Lütfen sadece 'e' veya 'h' girin.")
            print("-" * 30)