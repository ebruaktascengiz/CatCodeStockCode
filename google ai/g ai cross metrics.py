# -*- coding: utf-8 -*-

import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
import httpx
from google import genai


# --- config ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY .env dosyasında bulunamadı!")

client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-2.5-flash"  # Erişimin varsa "gemini-2.5-pro" da kullanılabilir

SOURCE_FILE_NAME = 'category_training.xlsx'
XLSX_FILE_PATH = os.path.join(os.path.expanduser('~'), 'Documents', SOURCE_FILE_NAME)
REQUIRED_COLUMNS = ['MR_No', 'Line_No', 'Description', 'CostCode', 'CostCodeDescription', 'UnitOfMeasurement',
                    'CategoryCode', 'CategoryCodeDescription']

# --- Global Değişkenler ---
vectorizer = TfidfVectorizer()
text_vectors = None
df_global = None
unique_categories_map = {}
valid_codes = set()


def update_learning_data():
    global vectorizer, text_vectors, df_global
    print("🧠 Öğrenim verisi güncelleniyor.")
    df_global['SearchableText'] = (
            df_global['Description'].str.lower().fillna('') + ' ' +
            df_global['UnitOfMeasurement'].str.lower().fillna('') + ' ' +
            df_global['CostCode'].astype(str).str.lower().fillna('') + ' ' +
            df_global['CategoryCodeDescription'].str.lower().fillna('')
    )
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df_global['SearchableText'])
    print("✅ Öğrenim verisi yeni bilgilerle güncellendi.")


def preprocess_and_load_data(file_path):
    global df_global, unique_categories_map, valid_codes
    try:
        df_global = pd.read_excel(file_path)
        for col in REQUIRED_COLUMNS:
            if col not in df_global.columns:
                if col in ['CostCodeDescription', 'CategoryCodeDescription']:
                    df_global[col] = ''
                else:
                    raise ValueError(f"Kaynak dosyada zorunlu '{col}' kolonu bulunamadı!")

        df_global['CategoryCode'] = df_global['CategoryCode'].astype(str)
        df_global.dropna(subset=['Description', 'CategoryCode'], inplace=True)
        df_global['MR_No'] = df_global['MR_No'].astype(str)
        df_global['Line_No'] = df_global['Line_No'].astype(str)
        unique_categories_map = df_global.dropna(
            subset=['CategoryCode', 'CategoryCodeDescription']
        ).drop_duplicates(subset=['CategoryCode']).set_index('CategoryCode')['CategoryCodeDescription'].to_dict()
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


def add_new_example(description: str, code: str):
    global df_global, valid_codes, unique_categories_map
    code = str(code)
    if code not in valid_codes:
        print(f"⚠️ Uyarı: Girdiğiniz '{code}' kodu mevcut kategori listesinde yok. Yeni bir kategori olarak ekleniyor.")
        new_category_name = input(f"Lütfen '{code}' kodu için bir kategori açıklaması girin: ").strip()
        if not new_category_name:
            new_category_name = "Yeni Eklenen Kategori"
        unique_categories_map[code] = new_category_name
        valid_codes.add(code)

    category_description = unique_categories_map.get(code)
    new_row_data = {
        'MR_No': 'MANUAL_ADD', 'Line_No': '1', 'Description': description,
        'CostCode': '', 'CostCodeDescription': '', 'UnitOfMeasurement': '',
        'CategoryCode': code, 'CategoryCodeDescription': category_description
    }
    new_row = pd.DataFrame([new_row_data])
    df_global = pd.concat([df_global, new_row], ignore_index=True)

    base_name, extension = os.path.splitext(XLSX_FILE_PATH)
    temp_file_path = f"{base_name}_temp{extension}"

    try:
        output_cols = [col for col in REQUIRED_COLUMNS if col in df_global.columns]
        df_global[output_cols].to_excel(temp_file_path, index=False, engine='openpyxl')
        if os.path.exists(XLSX_FILE_PATH):
            os.remove(XLSX_FILE_PATH)
        os.rename(temp_file_path, XLSX_FILE_PATH)
    except PermissionError:
        print(f"\n❌ HATA: Excel dosyasına yazma izni alınamadı.")
        print(
            f"Lütfen '{os.path.basename(XLSX_FILE_PATH)}' dosyasının başka bir programda açık olmadığından emin olun.")
        df_global.drop(df_global.tail(1).index, inplace=True)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return
    except Exception as e:
        print(f"\n❌ HATA: Excel dosyasına yazılırken beklenmedik bir sorun oluştu: {e}")
        df_global.drop(df_global.tail(1).index, inplace=True)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return

    update_learning_data()


def find_most_similar_examples(user_description: str, top_n: int = 5):
    user_vector = vectorizer.transform([user_description.lower()])
    cosine_similarities = cosine_similarity(user_vector, text_vectors).flatten()
    if len(cosine_similarities) == 0 or cosine_similarities.max() < 0.1:
        return pd.DataFrame()
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df_global.iloc[most_similar_indices]


def get_smart_suggestion(user_description: str):
    similar_examples = find_most_similar_examples(user_description)
    context_prompt_part = "Sistemde bu tanıma benzer anlamlı bir örnek bulamadım."
    if not similar_examples.empty:
        print("\n🔎 Öğrenim dosyasından benzer örnekler bulundu:")
        top_hit = similar_examples.iloc[0]
        mr_no_context = top_hit['MR_No']
        if mr_no_context != 'MANUAL_ADD':
            mr_context_df = df_global[df_global['MR_No'] == mr_no_context]
            for _, row in similar_examples.head(3).iterrows():
                print(f"   - '{row['Description']}' ({row['CategoryCodeDescription']}) -> {row['CategoryCode']}")
            mr_context_items = []
            for _, row in mr_context_df.iterrows():
                is_top_hit = " <<< (En Benzer Örnek Bu)" if row.name == top_hit.name else ""
                mr_context_items.append(
                    f"- Tanım: '{row['Description']}', Ölçü Birimi: '{row['UnitOfMeasurement']}' --> ATANAN KOD: {row['CategoryCode']}{is_top_hit}")
            context_prompt_part = (
                        f"Sana yol göstermesi için, sistemimde bulduğum en benzer örnek '{mr_no_context}' numaralı talepte (MR) yer alıyordu.\n"
                        f"O talebin tamamı, yani ilgili ürünlerin o gün nasıl gruplandığı aşağıdadır:\n\n" + "\n".join(
                    mr_context_items))
        else:
            context_prompt_part = (f"Sana yol göstermesi için sistemimde bulduğum bazı benzer örnekler:\n"
                                   f"- Tanım: '{top_hit['Description']}' --> ATANAN KOD: {top_hit['CategoryCode']}")

    formatted_categories_str = "\n".join(f"{code}: {name}" for code, name in sorted(unique_categories_map.items()))
    prompt_message = f"""
Sen bir satınalma kategorizasyon uzmanısın. Görevin, sana verilen yeni bir malzeme tanımını, geçmişteki benzer taleplerin bütününü ve mevcut kategori listesini analiz ederek en doğru şekilde sınıflandırmaktır. Iscilik, hizmet gibi kelimeler gecerse S ile baslayan kodlardan secmelisin. bulamazsan yeni kod önerme ama. kategori kodlar 4 haneli.
{context_prompt_part}
Şimdi, bu bağlamı ve aşağıdaki genel kategori listesini kullanarak, şu yeni tanımı sınıflandır:
Yeni Malzeme Tanımı: "{user_description}"
---
GEÇERLİ KATEGORİ LİSTESİ (Kod: İsim):
{formatted_categories_str}
---
GÖREVİN: Yukarıdaki bilgileri analiz et ve yeni malzeme tanımı için en uygun olan kategorinin SADECE 'CategoryCode'unu döndür. Cevabın KESİNLİKLE sadece kod olmalı. Başka hiçbir açıklama, selamlama veya metin içermemelidir.
"""
    try:
        print("\n🧠 Gelişmiş yapay zekadan (MR bağlamı ile) öneri isteniyor...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt_message,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=20,
            )
        )
        result_text = response.text
        if result_text is None:
            return "API-HATA: Modelden boş yanıt döndü."
        return result_text.strip()
    except Exception as e:
        return f"API-HATA: {e}"


if __name__ == "__main__":
    if preprocess_and_load_data(XLSX_FILE_PATH):
        print("-" * 50)
        print("   Kendi Kendini Geliştiren Kategori Önerme Aracı")
        print("   (MR Bağlamı ile Güçlendirilmiş Model)")
        print("   Çıkmak için 'çıkış' veya 'exit' yazabilirsiniz.")
        print("-" * 50)
        while True:
            user_input = input("\n➡️ Kategori önerisi için malzeme tanımını girin: ").strip()
            if user_input.lower() in ['çıkış', 'cikis', 'exit', 'quit']:
                print("\nProgramdan çıkılıyor...")
                break
            if not user_input:
                continue
            suggestion = get_smart_suggestion(user_input)
            print("-" * 30)
            print(f"💬 Girilen Tanım: '{user_input}'")
            if "API-HATA" in suggestion:
                print(f"⚠️ Hata: {suggestion}")
                continue
            suggestion = str(suggestion)
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