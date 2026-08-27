#pip install --upgrade certifi
#pip install google-generativeai
#pip show google-generativeai


import os
import certifi
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


# .env dosyası
load_dotenv()

# --- Konfigürasyon ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API anahtarı bulunamadı. Lütfen .env dosyasına GOOGLE_API_KEY ekleyin.")

genai.configure(api_key=api_key)
xlsx_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ru2b4 - category codes.xlsx')

# Global değişkenler
vectorizer = TfidfVectorizer()
text_vectors = None
df_global = None
unique_categories_map = {}
valid_codes = set()


def update_learning_data():
    """
    Global DataFrame'i kullanarak TF-IDF vektörlerini yeniden hesaplar.
    Arama için Description ve CategoryName sütunlarını birleştirir.
    """
    global vectorizer, text_vectors, df_global

    # Hem malzeme tanımından hem de kategori adından öğrenmek için birleşik bir metin oluşturuyoruz
    # Daha akıllı arama için vektörleştirdik
    df_global['SearchableText'] = df_global['Description'].str.lower().fillna('') + ' ' + df_global[
        'CategoryName'].str.lower().fillna('')

    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df_global['SearchableText'])

    print("🧠 Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    """
    Veriyi yükleyip temizleyip, ilk öğrenme setini hazırlayacağız.
    """
    global df_global, unique_categories_map, valid_codes
    try:
        df_global = pd.read_excel(file_path)
        df_global.dropna(subset=['Description', 'CategoryCode', 'CategoryName'], inplace=True)

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
    Yeni ve doğrulanan bir örneği Excel dosyasına ve hafızadaki data frame'e ekler.
    """
    global df_global, valid_codes, unique_categories_map

    if code not in valid_codes:
        print(f"⚠️ Uyarı: Girdiğiniz '{code}' kodu mevcut kategori listesinde yok. Yeni bir kategori olarak ekleniyor.")
        new_category_name = input(f"Lütfen '{code}' kodu için bir kategori adı girin: ").strip()
        if not new_category_name:
            new_category_name = "Yeni Eklenen Kategori"
        unique_categories_map[code] = new_category_name
        valid_codes.add(code)

    category_name = unique_categories_map.get(code)
    new_row = pd.DataFrame([{'Description': description, 'CategoryCode': code, 'CategoryName': category_name}])
    df_global = pd.concat([df_global, new_row], ignore_index=True)

    try:
        df_global.to_excel(xlsx_file_path, index=False, engine='openpyxl')
    except Exception as e:
        print(f"HATA: Excel dosyasına yazılırken sorun oluştu: {e}")
        df_global.drop(df_global.tail(1).index, inplace=True)  # Hata olursa eklemeyi geri al
        return

    update_learning_data()


def find_most_similar_examples(user_description: str, top_n: int = 5):
    """
    Birleşik arama metnini kullanarak en benzer N örneği bulur.
    """
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, text_vectors).flatten()

    if len(cosine_similarities) == 0 or cosine_similarities.max() < 0.1:
        return pd.DataFrame()

    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df_global.iloc[most_similar_indices]


def get_smart_suggestion(user_description: str, category_list: dict):
    """
    Google Gemini API'sinden en doğru kategori önerisini alır.
    Gelişmiş prompt ile optimize edilmiştir.
    """
    similar_examples = find_most_similar_examples(user_description)

    example_prompt_part = "Sistemde bu tanıma benzer örnek bulamadım."
    if not similar_examples.empty:
        print("\n🔎 Öğrenim dosyasından benzer örnekler bulundu:")
        example_texts = []
        for _, row in similar_examples.iterrows():
            # Kullanıcıya gösterilecek çıktı
            print(f"   - '{row['Description']}' ({row['CategoryName']}) -> {row['CategoryCode']}")
            # AI'a gönderilecek metin
            example_texts.append(f"- '{row['Description']}' tanımı için doğru kod '{row['CategoryCode']}'.")

        example_prompt_part = (
                "Sana yol göstermesi için sistemimden bulduğum bazı benzer örnekler şunlar:\n"
                + "\n".join(example_texts)
        )

    # Tüm kategori listesini formata uygun hale getir
    formatted_categories_str = "\n".join(f"{code}: {name}" for code, name in category_list.items())

    # --- GOOGLE GEMİNİ İÇİN OPTİMİZE EDİLMİŞ PROMPT ---
    prompt_message = f"""
Sen bir satınalma kategorizasyon uzmanısın. Görevin, sana verilen yeni bir malzeme tanımını, mevcut kategori listesine göre en doğru şekilde sınıflandırmak.

{example_prompt_part}

Şimdi, bu örnekleri ve aşağıdaki genel kategori listesini kullanarak, şu yeni tanımı sınıflandır:
Yeni Malzeme Tanımı: "{user_description}"

---
GEÇERLİ KATEGORİ LİSTESİ (Kod: İsim):
{formatted_categories_str}
---

GÖREVİN: Yukarıdaki bilgileri analiz et ve yeni malzeme tanımı için en uygun olan kategorinin SADECE 'CategoryCode'unu döndür.
Cevabın KESİNLİKLE sadece kod olmalı. Başka hiçbir açıklama, selamlama veya metin içermemelidir.

Örnek doğru yanıt: A123
Örnek yanlış yanıt: Kategori kodu A123'tür.
"""

    try:
        print("\n🧠 Google Gemini Flash'tan öneri isteniyor...")

        # Gemini 1.5 Flash modeli (en ucuz ve hızlı)
        model = genai.GenerativeModel('gemini-1.5-flash')

        response = model.generate_content(
            prompt_message,
            generation_config=genai.types.GenerationConfig(
                temperature=0,  # Tutarlı sonuçlar için
                max_output_tokens=20,
                top_p=1.0,
                top_k=1
            )
        )

        return response.text.strip()

    except Exception as e:
        return f"API-HATA: {e}"


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    if preprocess_and_load_data(xlsx_file_path):
        print("-" * 50)
        print("🚀 Google Gemini API ile Çalışan")
        print("Kendi Kendini Geliştiren Kategori Önerme Aracı")
        print("Çıkmak için 'çıkış' veya 'exit' yazabilirsiniz.")
        print("-" * 50)

        while True:
            user_input = input("\n➡️ Kategori önerisi için malzeme tanımını girin: ").strip()

            if user_input.lower() in ['çıkış', 'cikis', 'exit', 'quit']:
                print("Değişiklikler kaydedildi. Programdan çıkılıyor...")
                break
            if not user_input:
                continue

            suggestion = get_smart_suggestion(user_input, unique_categories_map)

            print("-" * 30)
            print(f"💬 Girilen Tanım: '{user_input}'")

            if suggestion.startswith("API-HATA:"):
                print(f"⚠️ Hata: {suggestion}")
                continue

            if suggestion in valid_codes:
                category_name = unique_categories_map.get(suggestion, "Bilinmeyen")
                print(f"🤖 ÖNERİ: {suggestion} ({category_name})")
            else:
                print(f"🤖 ÖNERİ (Geçersiz/Yeni Kod): {suggestion}")

            # --- GERİ BİLDİRİM DÖNGÜSÜ ---
            while True:
                feedback = input("Bu öneri doğru mu? (e: evet / h: hayır): ").lower()
                if feedback == 'e':
                    print("👍 Harika! Bilgi doğrulandı.")
                    break
                elif feedback == 'h':
                    correct_code = input("Lütfen doğru kodu girin: ").strip().upper()
                    if not correct_code:
                        print("Geçerli bir kod girmediniz. İşlem iptal edildi.")
                        break

                    print("📝 Anlaşıldı. Doğru bilgi sisteme öğretiliyor...")
                    add_new_example(user_input, correct_code)
                    break
                else:
                    print("Lütfen sadece 'e' veya 'h' girin.")
            print("-" * 30)