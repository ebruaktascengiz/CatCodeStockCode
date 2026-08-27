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


def update_learning_data():
    """
    Global DataFrame'i kullanarak TF-IDF vektörlerini yeniden hesaplar.
    Arama için Description ve CategoryName sütunlarını birleştirir.
    """
    global vectorizer, text_vectors, df_global
    df_global['SearchableText'] = df_global['Description'].str.lower().fillna('') + ' ' + df_global[
        'CategoryName'].str.lower().fillna('')
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df_global['SearchableText'])
    print("🧠 Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    """
    Veriyi yükler, temizler ve ilk öğrenme setini hazırlar.
    """
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
    """
    Yeni ve DOĞRULANMIŞ bir örneği Excel dosyasına ve hafızadaki DataFrame'e ekler.
    """
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


def find_most_similar_examples(user_description: str, top_n: int = 15):  # Daha geniş bir havuz alıyoruz
    """
    Birleşik arama metnini kullanarak en benzer N örneği bulur.
    """
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, text_vectors).flatten()
    if len(cosine_similarities) == 0 or cosine_similarities.max() < 0.1:
        return pd.DataFrame()
    # argsort en küçükten büyüğe sıralar, biz sondan N taneyi alırız
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df_global.iloc[most_similar_indices]


# YENİ FONKSİYON: Alakalı kategorileri ayıklamak için
def get_relevant_categories_from_examples(examples_df: pd.DataFrame) -> dict:
    """Verilen örnek DataFrame'inden benzersiz kategori kodlarını ve adlarını çıkarır."""
    if examples_df.empty:
        return {}
    # Sadece 'CategoryCode' ve 'CategoryName' sütunlarını alıp, tekrarları kaldırıyoruz.
    relevant_cats_df = examples_df[['CategoryCode', 'CategoryName']].drop_duplicates()
    # Sözlük formatına çeviriyoruz.
    return pd.Series(relevant_cats_df.CategoryName.values, index=relevant_cats_df.CategoryCode).to_dict()


def get_smart_suggestion(user_description: str):
    """
    MALİYET-OPTİMİZE EDİLMİŞ: Yapay zekaya sadece en alakalı bağlamı göndererek öneri ister.
    """
    # 1. Adım: Geniş bir benzer örnek havuzu bul (API maliyeti yok)
    similar_examples_pool = find_most_similar_examples(user_description, top_n=15)

    example_prompt_part = ""
    # 2. Adım: Gönderilecek kategori listesini belirle
    if not similar_examples_pool.empty:
        # En alakalı kategorileri bu havuzdan çıkar
        relevant_categories = get_relevant_categories_from_examples(similar_examples_pool)

        # Kullanıcıya ve AI'a gösterilecek ilk 5 örneği hazırla
        top_5_examples = similar_examples_pool.head(5)
        print("\n🔎 Öğrenim dosyasından benzer örnekler bulundu:")
        example_texts = []
        for _, row in top_5_examples.iterrows():
            print(f"   - '{row['Description']}' ({row['CategoryName']}) -> {row['CategoryCode']}")
            example_texts.append(f"- '{row['Description']}' tanımı için doğru kod '{row['CategoryCode']}'.")
        example_prompt_part = "Sana yol göstermesi için sistemimden bulduğum bazı benzer örnekler şunlar:\n" + "\n".join(
            example_texts)

        # API'ye gönderilecek kategori listesi artık çok daha kısa!
        category_list_for_api = relevant_categories
        category_list_header = "İLGİLİ KATEGORİ LİSTESİ (Kod: İsim):"
    else:
        # Fallback: Hiç benzer örnek bulunamazsa, tüm listeyi gönder (nadir durum)
        print("\n⚠️ Benzer örnek bulunamadı. Yapay zeka tüm kategori listesini kullanarak tahminde bulunacak.")
        example_prompt_part = "Sistimde bu tanıma benzer örnek bulamadım."
        category_list_for_api = unique_categories_map
        category_list_header = "TÜM KATEGORİ LİSTESİ (Kod: İsim):"

    formatted_categories_str = "\n".join(f"{code}: {name}" for code, name in category_list_for_api.items())

    # --- MALİYET-OPTİMİZE EDİLMİŞ PROMPT ---
    prompt_message = f"""
    Sen, sana verilen bir malzeme tanımını, sunulan kategori listesindeki en uygun kod ile eşleştiren bir uzmansın.

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
        print("\n🧠 Maliyet-optimize yapay zekadan öneri isteniyor...")
        response = client.chat.completions.create(
            # MALİYET DÜŞÜRMEK İÇİN MODEL DEĞİŞTİRİLEBİLİR. gpt-4-turbo da kalabilir.
            # gpt-3.5-turbo çok daha ucuzdur ve bu odaklanmış görevde genellikle yeterlidir.
            model="gpt-3.5-turbo",
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

            # DEĞİŞİKLİK: get_smart_suggestion artık tüm listeyi parametre olarak almıyor.
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