import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from typing import List, Tuple, Dict
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS


class CategorySuggestionSystem:
    def __init__(self, csv_file_path: str, openai_api_key: str):
        """
        AI destekli kategori kodu önerme sistemi

        Args:
            csv_file_path: CSV dosyasının yolu
            openai_api_key: OpenAI API anahtarı
        """
        self.csv_file_path = csv_file_path
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

        # Veri yükleme ve ön işleme
        self.load_and_preprocess_data()

        # TF-IDF vektörizör
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000
        )

        # Vektörleri hesapla
        self.fit_vectorizer()

    def load_and_preprocess_data(self):
        """CSV dosyasını yükle ve ön işleme yap"""
        try:
            # CSV dosyasını yükle
            self.df = pd.read_csv(self.csv_file_path, encoding='utf-8')

            # Kolonları kontrol et
            required_columns = ['Description', 'CategoryCode', 'CategoryName']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV dosyasında şu kolonlar bulunmalı: {required_columns}")

            # Boş değerleri temizle
            self.df = self.df.dropna(subset=['Description', 'CategoryCode'])

            # Tanımları temizle ve normalize et
            self.df['clean_description'] = self.df['Description'].apply(self.clean_text)

            print(f"✅ {len(self.df)} kayıt yüklendi")

        except Exception as e:
            print(f"❌ CSV dosyası yüklenirken hata: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Metni temizle ve normalize et"""
        if pd.isna(text):
            return ""

        # Küçük harfe çevir
        text = str(text).lower()

        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s]', ' ', text)

        # Çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def fit_vectorizer(self):
        """TF-IDF vektörizörü eğit"""
        descriptions = self.df['clean_description'].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        print("✅ TF-IDF vektörizör eğitildi")

    def get_similarity_suggestions(self, description: str, top_k: int = 5) -> List[Dict]:
        """Benzerlik tabanlı öneriler al"""
        try:
            # Giriş metnini temizle ve vektörle
            clean_desc = self.clean_text(description)
            query_vector = self.vectorizer.transform([clean_desc])

            # Cosine benzerliği hesapla
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # En benzer olanları al
            top_indices = similarities.argsort()[-top_k:][::-1]

            suggestions = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum benzerlik eşiği
                    suggestions.append({
                        'category_code': self.df.iloc[idx]['CategoryCode'],
                        'category_name': self.df.iloc[idx]['CategoryName'],
                        'similarity_score': float(similarities[idx]),
                        'reference_description': self.df.iloc[idx]['Description']
                    })

            return suggestions

        except Exception as e:
            print(f"❌ Benzerlik hesaplarken hata: {e}")
            return []

    def get_chatgpt_suggestion(self, description: str) -> Dict:
        """ChatGPT ile kategori önerisi al"""
        try:
            # API key kontrolü
            if not self.openai_api_key:
                return {
                    'suggested_code': None,
                    'explanation': 'OpenAI API Key bulunamadı - sadece benzerlik analizi kullanılıyor',
                    'confidence': 'no_api_key'
                }

            # Mevcut kategorilerin örneklerini hazırla
            category_examples = self.df.groupby('CategoryCode').agg({
                'CategoryName': 'first',
                'Description': lambda x: list(x)[:3]  # Her kategoriden 3 örnek
            }).to_dict('index')

            # Prompt hazırla
            prompt = f"""
Sen bir satınalma uzmanısın. Aşağıda verilen malzeme tanımı için en uygun kategori kodunu öner.

Malzeme Tanımı: "{description}"

Mevcut Kategori Kodları ve Örnekleri:
"""

            # Her kategoriden örnekler ekle (fazla uzun olmaması için ilk 20 kategori)
            for code, info in list(category_examples.items())[:20]:
                prompt += f"\n{code} - {info['CategoryName']}: {', '.join(info['Description'][:2])}"

            prompt += """

Lütfen sadece kategori kodunu ve kısa bir açıklama döndür.
Format: KATEGORI_KODU - Açıklama
"""

            # OpenAI API çağrısı
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sen bir satınalma kategorizasyon uzmanısın."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            # Sonucu parse et
            if " - " in result:
                code, explanation = result.split(" - ", 1)
                return {
                    'suggested_code': code.strip(),
                    'explanation': explanation.strip(),
                    'confidence': 'ai_suggestion'
                }
            else:
                return {
                    'suggested_code': result,
                    'explanation': 'AI önerisi',
                    'confidence': 'ai_suggestion'
                }

        except Exception as e:
            print(f"❌ ChatGPT önerisi alınırken hata: {e}")
            return {
                'suggested_code': None,
                'explanation': f'AI önerisi alınamadı: {str(e)}',
                'confidence': 'error'
            }

    def suggest_category(self, description: str) -> Dict:
        """Ana öneri fonksiyonu - hem benzerlik hem AI kullanır"""
        try:
            # 1. Benzerlik tabanlı öneriler
            similarity_suggestions = self.get_similarity_suggestions(description, top_k=3)

            # 2. ChatGPT önerisi
            ai_suggestion = self.get_chatgpt_suggestion(description)

            # 3. Sonuçları birleştir
            result = {
                'input_description': description,
                'similarity_suggestions': similarity_suggestions,
                'ai_suggestion': ai_suggestion,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # En iyi önerileri belirle
            if similarity_suggestions and similarity_suggestions[0]['similarity_score'] > 0.7:
                result['recommended_code'] = similarity_suggestions[0]['category_code']
                result['recommendation_source'] = 'similarity_high_confidence'
            elif ai_suggestion.get('suggested_code'):
                result['recommended_code'] = ai_suggestion['suggested_code']
                result['recommendation_source'] = 'ai_suggestion'
            elif similarity_suggestions:
                result['recommended_code'] = similarity_suggestions[0]['category_code']
                result['recommendation_source'] = 'similarity_low_confidence'
            else:
                result['recommended_code'] = None
                result['recommendation_source'] = 'no_suggestion'

            return result

        except Exception as e:
            return {
                'input_description': description,
                'error': str(e),
                'recommended_code': None,
                'recommendation_source': 'error'
            }


# Flask Web API
app = Flask(__name__)
CORS(app)

# Global değişken - sistem başlatıldığında yüklenecek
suggestion_system = None


@app.route('/health', methods=['GET'])
def health_check():
    """Sistem sağlık kontrolü"""
    return jsonify({
        'status': 'healthy',
        'system_loaded': suggestion_system is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })


@app.route('/suggest-category', methods=['POST'])
def suggest_category_api():
    """Kategori önerisi API endpoint'i"""
    try:
        if not suggestion_system:
            return jsonify({'error': 'Sistem henüz yüklenmedi'}), 500

        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({'error': 'description parametresi gerekli'}), 400

        description = data['description']
        if not description.strip():
            return jsonify({'error': 'Boş tanım gönderildi'}), 400

        # Öneri al
        result = suggestion_system.suggest_category(description)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/categories', methods=['GET'])
def get_categories():
    """Mevcut kategorileri listele"""
    try:
        if not suggestion_system:
            return jsonify({'error': 'Sistem henüz yüklenmedi'}), 500

        categories = suggestion_system.df[['CategoryCode', 'CategoryName']].drop_duplicates()
        return jsonify({
            'categories': categories.to_dict('records'),
            'total_count': len(categories)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_system():
    """Sistemi başlat"""
    global suggestion_system

    try:
        # CSV dosyası yolunu kontrol et
        possible_paths = [
            r'C:\Users\ebru.aktas\Documents\ru2b8 - category codes.csv',  # Sizin dosya yolunuz
            "Documents/ru2b8 - category codes.csv",
            "ru2b8 - category codes.csv",
            "./Documents/ru2b8 - category codes.csv",
            "../Documents/ru2b8 - category codes.csv",
            os.path.expanduser("~/Documents/ru2b8 - category codes.csv")
        ]

        CSV_FILE_PATH = None
        for path in possible_paths:
            if os.path.exists(path):
                CSV_FILE_PATH = path
                print(f"✅ CSV dosyası bulundu: {path}")
                break

        if not CSV_FILE_PATH:
            print("❌ CSV dosyası bulunamadı!")
            print("📁 Aşağıdaki konumlardan birinde olmalı:")
            for path in possible_paths:
                print(f"   - {path}")

            # Kullanıcıdan dosya yolunu al
            custom_path = input("\n📝 CSV dosyasının tam yolunu girin (Enter ile geç): ").strip()
            if custom_path and os.path.exists(custom_path):
                CSV_FILE_PATH = custom_path
                print(f"✅ Özel yol kabul edildi: {custom_path}")
            else:
                print("❌ Dosya yolu geçersiz veya dosya bulunamadı!")
                return False

        # OpenAI API Key kontrol
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

        if not OPENAI_API_KEY:
            print("\n⚠️  OPENAI_API_KEY çevre değişkeni ayarlanmamış!")
            print("🔧 Çözüm seçenekleri:")
            print("   1. Terminal: export OPENAI_API_KEY='your_api_key_here'")
            print("   2. Windows CMD: set OPENAI_API_KEY=your_api_key_here")
            print("   3. Manual giriş (şimdi)")

            # Kullanıcıdan API key al
            manual_key = input("\n🔑 OpenAI API Key'inizi buraya girin (Enter ile geç): ").strip()
            if manual_key:
                OPENAI_API_KEY = manual_key
                os.environ['OPENAI_API_KEY'] = manual_key
                print("✅ API Key manuel olarak ayarlandı")
            else:
                print("⚠️  API Key olmadan sadece benzerlik analizi çalışacak")

        # Sistemi başlat
        print("\n🚀 Kategori önerme sistemi başlatılıyor...")
        suggestion_system = CategorySuggestionSystem(CSV_FILE_PATH, OPENAI_API_KEY)
        print("✅ Sistem başarıyla yüklendi!")

        return True

    except Exception as e:
        print(f"❌ Sistem başlatılırken hata: {e}")
        return False


# Test fonksiyonu
def test_system():
    """Sistemi test et"""
    if not suggestion_system:
        print("❌ Sistem yüklü değil!")
        return

    test_descriptions = [
        "Bilgisayar monitörü",
        "Ofis masası",
        "Yazıcı kağıdı",
        "Temizlik malzemesi"
    ]

    print("\n🧪 Test Sonuçları:")
    print("=" * 50)

    for desc in test_descriptions:
        result = suggestion_system.suggest_category(desc)
        print(f"\n📝 Tanım: {desc}")
        print(f"💡 Önerilen Kod: {result.get('recommended_code', 'Bulunamadı')}")
        print(f"🎯 Kaynak: {result.get('recommendation_source', 'Bilinmiyor')}")

        if result.get('similarity_suggestions'):
            print(f"📊 Benzerlik Skoru: {result['similarity_suggestions'][0]['similarity_score']:.3f}")


if __name__ == "__main__":
    # Sistemi başlat
    if initialize_system():
        # Test et
        test_system()

        # Web sunucusunu başlat
        print("\n🌐 Web API başlatılıyor...")
        print("📍 API Endpoints:")
        print("   POST /suggest-category - Kategori önerisi al")
        print("   GET  /categories - Mevcut kategorileri listele")
        print("   GET  /health - Sistem durumu")
        print("\n💡 Kullanım örneği:")
        print("   curl -X POST http://localhost:5000/suggest-category \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"description\": \"Bilgisayar monitörü\"}'")

        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Sistem başlatılamadı!")

# Örnek JavaScript fetch kodu (frontend için)
"""
// Frontend'de kullanım örneği:

async function suggestCategory(description) {
    try {
        const response = await fetch('http://localhost:5000/suggest-category', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ description: description })
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Kategori önerisi alınırken hata:', error);
        return null;
    }
}

// Butona tıklandığında çalışacak fonksiyon
async function onSuggestButtonClick() {
    const descriptionInput = document.getElementById('material-description');
    const categoryCodeInput = document.getElementById('category-code');
    const suggestButton = document.getElementById('suggest-button');

    // Butonu devre dışı bırak
    suggestButton.disabled = true;
    suggestButton.textContent = 'Öneriliyor...';

    try {
        const result = await suggestCategory(descriptionInput.value);

        if (result && result.recommended_code) {
            categoryCodeInput.value = result.recommended_code;

            // Kullanıcıya bilgi ver
            alert(`Önerilen Kategori: ${result.recommended_code}\nKaynak: ${result.recommendation_source}`);
        } else {
            alert('Uygun kategori bulunamadı. Lütfen manuel olarak seçin.');
        }
    } catch (error) {
        alert('Kategori önerisi alınırken hata oluştu.');
    } finally {
        // Butonu tekrar aktif et
        suggestButton.disabled = false;
        suggestButton.textContent = 'Kategori Öner';
    }
}
"""