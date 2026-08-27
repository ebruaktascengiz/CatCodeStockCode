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
xlsx_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ru2b4 - category codes.xlsx')

# Global değişkenler
vectorizer = TfidfVectorizer()
text_vectors = None
df_global = None
unique_categories_map = {}
valid_codes = set()

# --- Güven Eşikleri ---
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.20


def update_learning_data():
    """Global DataFrame'i kullanarak TF-IDF vektörlerini yeniden hesaplar."""
    global vectorizer, text_vectors, df_global
    df_global['SearchableText'] = df_global['Description'].str.lower().fillna('') + ' ' + df_global[
        'CategoryName'].str.lower().fillna('')
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df_global['SearchableText'])
    print("🧠 Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    """Veriyi yükler, temizler ve ilk öğrenme setini hazırlar."""
    global df_global, unique_categories_map, valid_codes
    try:
        df_global = pd.read_excel(file_path)
        df_global.dropna(subset=['Description', 'CategoryCode', 'CategoryName'], inplace=True)
        unique_categories_map = df_global.drop_duplicates(subset=['CategoryCode']).set_index('CategoryCode')[
            'CategoryName'].to_dict()
        valid_codes = set(df_global['CategoryCode'])
        update_learning_data()
        print(f"✅ {len(df_global)} adet geçerli örnekle öğrenim verisi hazırlandı.")
        return True
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı: {file_path}")
        return False
    except Exception as e:
        print(f"HATA: Veri hazırlanırken bir sorun oluştu: {e}")
        return False


def add_new_example(description, code):
    """Yeni ve DOĞRULANMIŞ bir örneği Excel dosyasına ve hafızadaki DataFrame'e ekler."""
    global df_global, valid_codes, unique_categories_map
    if code not in valid_codes:
        print(f"⚠️ Uyarı: Girdiğiniz '{code}' kodu mevcut kategori listesinde yok. Yeni bir kategori olarak ekleniyor.")
        new_category_name = input(f"Lütfen '{code}' kodu için bir kategori adı girin: ").strip()
        if not new_category_name: new_category_name = "Yeni Eklenen Kategori"
        unique_categories_map[code] = new_category_name
        valid_codes.add(code)
    category_name = unique_categories_map.get(code)
    new_row = pd.DataFrame([{'Description': description, 'CategoryCode': code, 'CategoryName': category_name}])
    df_global = pd.concat([df_global, new_row], ignore_index=True)
    try:
        df_global.to_excel(xlsx_file_path, index=False, engine='openpyxl')
    except Exception as e:
        print(f"HATA: Excel dosyasına yazılırken sorun oluştu: {e}")
        df_global.drop(df_global.tail(1).index, inplace=True)
        return
    update_learning_data()


def find_most_similar_examples(user_description: str, top_n: int = 15):
    """Benzer örnekleri ve benzerlik skorlarını döndürür."""
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, text_vectors).flatten()

    if len(cosine_similarities) == 0:
        return pd.DataFrame(), []

    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    similar_scores = cosine_similarities[most_similar_indices]

    return df_global.iloc[most_similar_indices], similar_scores


def get_smart_suggestion(user_description: str):
    """
    AKILLI HİBRİT YAKLAŞIM: Güven seviyesine göre en doğru ve maliyet-etkin yöntemi seçer.
    """
    similar_examples_pool, similar_scores = find_most_similar_examples(user_description, top_n=5)

    max_similarity = 0.0 if similar_examples_pool.empty else similar_scores[0]

    if max_similarity > HIGH_CONFIDENCE_THRESHOLD and len(similar_examples_pool) >= 3 and similar_examples_pool.head(3)[
        'CategoryCode'].nunique() == 1:
        suggestion = similar_examples_pool.iloc[0]['CategoryCode']
        print("\n✅ Yüksek Güvenli Yerel Eşleşme Bulundu (API kullanılmadı).")
        return suggestion

    example_texts = []
    if not similar_examples_pool.empty:
        print("\n🔎 Öğrenim dosyasından benzer örnekler bulundu:")

        # --- DÜZELTME BURADA ---
        # `enumerate` kullanarak döngüye bir pozisyon sayacı (pos) ekliyoruz.
        for pos, (original_index, row) in enumerate(similar_examples_pool.head(5).iterrows()):
            # similar_scores'a erişmek için artık `pos` kullanıyoruz.
            print(f"   - (Benzerlik: {similar_scores[pos]:.2f}) '{row['Description']}' -> {row['CategoryCode']}")
            example_texts.append(f"- '{row['Description']}' tanımı için doğru kod '{row['CategoryCode']}'.")
        # --- DÜZELTME SONU ---

        example_prompt_part = "Sana yol göstermesi için sistemimden bulduğum bazı benzer örnekler şunlar:\n" + "\n".join(
            example_texts)
    else:
        example_prompt_part = "Sistemimde bu tanıma benzer örnek bulamadım."

    if max_similarity > LOW_CONFIDENCE_THRESHOLD:
        print("\n🧠 Standart Güven: Yapay zekaya ilgili kategorilerle soruluyor...")
        relevant_cats_df = similar_examples_pool[['CategoryCode', 'CategoryName']].drop_duplicates()
        category_list_for_api = pd.Series(relevant_cats_df.CategoryName.values,
                                          index=relevant_cats_df.CategoryCode).to_dict()
        category_list_header = "İLGİLİ KATEGORİ LİSTESİ (Kod: İsim):"
    else:
        print("\n⚠️ Düşük Güven: En doğru sonuç için yapay zekaya tüm kategori listesiyle soruluyor (Güvenlik Ağı).")
        category_list_for_api = unique_categories_map
        category_list_header = "TÜM KATEGORİ LİSTESİ (Kod: İsim):"

    formatted_categories_str = "\n".join(f"{code}: {name}" for code, name in category_list_for_api.items())

    prompt_message = f"""
    Sen bir satınalma kategorizasyon uzmanısın. Görevin, sana verilen yeni bir malzeme tanımını, sunulan kategori listesindeki en uygun kod ile eşleştirmek.

    {example_prompt_part}

    Bu bilgileri ve aşağıdaki kategori listesini kullanarak şu tanımı sınıflandır:
    Yeni Tanım: "{user_description}"

    ---
    {category_list_header}
    {formatted_categories_str}
    ---

    GÖREVİN: Yukarıdaki yeni tanım için bu listedeki en uygun kategorinin SADECE 'CategoryCode'unu döndür. Başka hiçbir şey yazma.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt_message}],
            temperature=0,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API-HATA: {e}"


# --- Ana Çalıştırma Bloğu (DEĞİŞİKLİK YOK) ---
if __name__ == "__main__":
    if preprocess_and_load_data(xlsx_file_path):
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
                category_name = unique_categories_map.get(suggestion, "Bilinmeyen")
                print(f"🤖 ÖNERİ: {suggestion} ({category_name})")
            else:
                print(f"🤖 ÖNERİ (Geçersiz/Yeni Kod): {suggestion}")

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