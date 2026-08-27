import os
import re
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
import urllib

# =====================================================================
# 🚀 GÜVENLİK DUVARI VE SSL ENGELEME YAMASI (MONKEY-PATCH)
# =====================================================================
import httpx
import warnings

warnings.filterwarnings("ignore")

_original_client_init = httpx.Client.__init__


def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_client_init

_original_async_client_init = httpx.AsyncClient.__init__


def _patched_async_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_async_client_init(self, *args, **kwargs)


httpx.AsyncClient.__init__ = _patched_async_client_init
# =====================================================================

from google import genai
from google.genai import types

load_dotenv()


class CategoryAIExpress:
    GENERIC_KEYWORDS = {
        'genel', 'general', 'diger', 'muhtelif', 'misc', 'miscellaneous',
        'na', 'n/a', 'yok', 'belirsiz', 'other', 'others', 'various', 'unknown', 'bilinmiyor'
    }

    # %90 ve üzeri "yüksek güven" kabul edilir; kullanıcıya sade kod gösterilir.
    HIGH_CONFIDENCE_THRESHOLD = 90

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY .env dosyasında bulunamadı!")

        self.client = genai.Client(api_key=api_key)
        # gemini-2.5-flash artık yeni kullanıcılara kapalı. Güncel GA model: gemini-3.5-flash
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")

        conn_str = os.getenv("SQL_CONNECTION_STRING")
        if not conn_str:
            raise ValueError("SQL_CONNECTION_STRING .env dosyasında bulunamadı!")

        params = urllib.parse.quote_plus(conn_str)
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        self.vectorizer = TfidfVectorizer()
        self.text_vectors = None
        self.df_verified = None
        self.unique_categories_map = {}
        self.valid_codes = set()
        # normalize edilmiş CostCodeDescription -> Counter({CategoryCode: adet})
        # CostCode numarası proje bazında değiştiği için referans olarak KOD değil,
        # açıklama metni kullanılır. Bu dağılım tek bir kategoriye kilitlenmez;
        # aynı CostCodeDescription birden fazla kategori koduna gidebilir, bu yüzden
        # yüzdesel bir ipucu olarak sunulur.
        self.costdesc_category_counts = {}

        self.reload_learning_data()

    @staticmethod
    def _normalize(text_value: str) -> str:
        if not text_value:
            return ""
        t = str(text_value).strip().lower()
        t = t.translate(str.maketrans('çğıöşü', 'cgiosu'))
        t = re.sub(r'\s+', ' ', t)
        return t

    def _is_generic(self, description: str) -> bool:
        normalized = self._normalize(description)
        if not normalized:
            return True
        return any(keyword in normalized for keyword in self.GENERIC_KEYWORDS)

    def reload_learning_data(self):
        """
        SQL'den Verified=1 olan verileri çekip belleğe yazar, TF-IDF'i eğitir ve
        CostCodeDescription -> CategoryCode yüzdesel dağılımını önceden hesaplar.
        """
        try:
            query = """
                SELECT MR_No, Line_No, Description, CostCode, CostCodeDescription, 
                       UnitOfMeasurement, CategoryCode, CategoryCodeDescription 
                FROM SCM.dbo.CatCodeAI 
                WHERE Verified = 1 AND Description IS NOT NULL AND CategoryCode IS NOT NULL
            """
            with self.engine.connect() as conn:
                self.df_verified = pd.read_sql(text(query), conn)

            if self.df_verified.empty:
                print("⚠️ SQL'de Verified=1 olan kayıt bulunamadı! Lütfen önce verileri yükleyin.")
                return

            self.df_verified['CategoryCode'] = self.df_verified['CategoryCode'].astype(str).str.strip()
            self.df_verified['MR_No'] = self.df_verified['MR_No'].astype(str).str.strip()
            self.df_verified['Line_No'] = self.df_verified['Line_No'].astype(str).str.strip()

            df_clean_cats = self.df_verified.dropna(subset=['CategoryCode', 'CategoryCodeDescription'])
            df_unique_cats = df_clean_cats.drop_duplicates(subset=['CategoryCode'])
            self.unique_categories_map = df_unique_cats.set_index('CategoryCode')['CategoryCodeDescription'].to_dict()
            self.valid_codes = set(self.df_verified['CategoryCode'])

            # --- CostCodeDescription (normalize) -> CategoryCode dağılımı ---
            self.costdesc_category_counts = {}
            norm_desc = self.df_verified['CostCodeDescription'].fillna('').map(self._normalize)
            for norm, group_idx in norm_desc.groupby(norm_desc).groups.items():
                if not norm or norm in self.GENERIC_KEYWORDS:
                    continue
                cats = self.df_verified.loc[group_idx, 'CategoryCode']
                self.costdesc_category_counts[norm] = Counter(cats)

            # Hibrit Arama Metnini Vektörleştirme
            self.df_verified['SearchableText'] = (
                    self.df_verified['Description'].str.lower().fillna('') + ' ' +
                    self.df_verified['UnitOfMeasurement'].str.lower().fillna('') + ' ' +
                    self.df_verified['CostCodeDescription'].str.lower().fillna('') + ' ' +
                    self.df_verified['CategoryCodeDescription'].str.lower().fillna('')
            )

            self.text_vectors = self.vectorizer.fit_transform(self.df_verified['SearchableText'])
            print(f"🚀 Başarılı: {len(self.df_verified)} kayıt yüklendi "
                  f"({len(self.costdesc_category_counts)} farklı CostCodeDescription için dağılım hesaplandı).")

        except Exception as e:
            print(f"❌ Bellek yüklenirken SQL hatası oluştu: {str(e)}")

    def find_most_similar_examples(self, user_description: str, cost_code_description: str = None,
                                    unit: str = None, top_n: int = 5):
        """Cosine similarity ile en benzer geçmiş kayıtları arar (ANA sinyal)."""
        if self.text_vectors is None or self.df_verified is None or self.df_verified.empty:
            return pd.DataFrame()

        query_parts = [user_description.lower()]
        if unit:
            query_parts.append(str(unit).lower())
        if cost_code_description and not self._is_generic(cost_code_description):
            query_parts.append(str(cost_code_description).lower())

        user_vector = self.vectorizer.transform([' '.join(query_parts)])
        cosine_similarities = cosine_similarity(user_vector, self.text_vectors).flatten()
        if len(cosine_similarities) == 0:
            return pd.DataFrame()

        df_temp = self.df_verified.copy()
        df_temp['similarity_score'] = cosine_similarities
        df_temp = df_temp[df_temp['similarity_score'] >= 0.10].sort_values(by='similarity_score', ascending=False)
        return df_temp.drop_duplicates(subset=['CategoryCode']).head(top_n)

    def _build_supporting_context(self, cost_code_description: str, mr_no: str, line_no: str = None) -> str:
        """
        İKİNCİL sinyal: CostCodeDescription bazlı yüzdesel kod dağılımı + MR paket bağlamı.
        Bunlar ana sinyalin (description benzerliği) yerine değil, karasız kalınan
        durumlarda ona yardımcı/tie-breaker olarak kullanılır.
        """
        lines = []

        if cost_code_description:
            norm = self._normalize(cost_code_description)
            counts = self.costdesc_category_counts.get(norm)
            if self._is_generic(cost_code_description):
                lines.append(f"- CostCode açıklaması ('{cost_code_description}') genel/belirsiz, düşük ağırlık ver.")
            elif counts:
                total = sum(counts.values())
                top3 = counts.most_common(3)
                dist = ", ".join(f"{code} (%{round(100 * cnt / total)})" for code, cnt in top3)
                lines.append(
                    f"- '{cost_code_description}' açıklamalı geçmiş kayıtlarda görülen kod dağılımı: {dist}. "
                    f"NOT: Bu açıklama TEK bir kategoriye kilitli değildir, birden fazla koda gidebilir; "
                    f"bu yüzden kesin kanıt değil, olasılıksal bir ipucudur."
                )
            else:
                lines.append(f"- '{cost_code_description}' için geçmiş kayıt yok, bu sinyal kullanılamaz.")

        if mr_no:
            sibling_df = self.df_verified[self.df_verified['MR_No'] == str(mr_no).strip()]
            if line_no:
                sibling_df = sibling_df[sibling_df['Line_No'] != str(line_no).strip()]
            if not sibling_df.empty:
                items = "; ".join(
                    f"'{r['Description']}'->{r['CategoryCode']}" for _, r in sibling_df.head(8).iterrows()
                )
                lines.append(f"- Aynı MR paketindeki diğer satırlar (en zayıf ipucu): {items}")

        if not lines:
            return "Ek bağlam bilgisi verilmedi (CostCode açıklaması / MR No boş)."
        return "\n".join(lines)

    def get_smart_suggestion(self, user_description: str, cost_code_description: str = None,
                              unit: str = None, mr_no: str = None, line_no: str = None) -> str:
        """
        Malzeme/hizmet tanımına göre en doğru kategori kodunu ve olasılık yüzdesini döndürür.
        Ana dayanak her zaman `user_description`dır (semantik benzerlik); CostCodeDescription
        ve MR bağlamı sadece tanım benzerliği karasız kaldığında destekleyici ipucudur.
        """
        try:
            similar_examples = self.find_most_similar_examples(
                user_description, cost_code_description=cost_code_description, unit=unit, top_n=5
            )

            if similar_examples.empty:
                similarity_context = "Sistemde bu tanıma doğrudan benzeyen geçmiş bir kayıt bulunamadı."
            else:
                example_texts = [
                    f"  - '{r['Description']}' (Birim: {r['UnitOfMeasurement']}) --> KOD: {r['CategoryCode']} "
                    f"(benzerlik: {r['similarity_score']:.2f})"
                    for _, r in similar_examples.iterrows()
                ]
                similarity_context = "Tanım benzerliğine göre bulunan adaylar:\n" + "\n".join(example_texts)

            supporting_context = self._build_supporting_context(cost_code_description, mr_no, line_no)

            formatted_categories_str = "\n".join(
                f"{code}: {name}" for code, name in sorted(self.unique_categories_map.items())
            ) if self.unique_categories_map else "Kategori listesi boş veya yüklenemedi."

            prompt_message = f"""
            Sen kurumsal bir satınalma katalog uzmanısın. Kategori kodları, bir malzeme/hizmetin hangi
            tür tedarikçiden alınacağını belirler. Görevin, verilen tanımı aşağıdaki GEÇERLİ KATEGORİ
            LİSTESİ'nden en uygun 4 haneli kodla eşleştirmek. Kullanıcılar aynı malzemeyi farklı
            yazılışlarla ifade edebilir; anlamsal/işlevsel benzerliğe odaklan, birebir kelimeye takılma.

            Sınıflandırılacak Tanım: "{user_description}"

            ANA SİNYAL (en güvenilir - kararının birincil dayanağı budur) - Tanım benzerliği:
            {similarity_context}

            İKİNCİL SİNYAL (destekleyici ipucu, TEK BAŞINA KANIT SAYILMAZ) - CostCode bağlamı + MR paketi:
            {supporting_context}
            NOT: CostCodeDescription tek bir kategoriye kilitli değildir; aynı açıklama farklı projelerde
            farklı kategori kodlarına gidebilir. Ancak kodun İLK HARFİ çoğunlukla kategori GRUBUNU işaret
            eder (örn. M ile başlayanlar genelde Mekanik grup, A ile başlayanlar genelde Mimari grup, S ile
            başlayanlar Servis/işçilik grubudur; U ile başlayan kodlarda bu kurala istisnalar daha sık
            görülür). Bu ikincil sinyali, ANA SİNYAL iki veya daha fazla kategori arasında net ayrım
            yapmana izin vermediğinde bir tie-breaker/olasılık ayarlayıcı olarak kullan.

            GEÇERLİ KATEGORİ LİSTESİ:
            {formatted_categories_str}

            KURALLAR:
            1. Tanımda 'işçilik', 'hizmet', 'montaj', 'bakım', 'onarım', 'dizayn', 'tasarım' gibi
               kelimeler varsa ÖNCELİKLE 'S' ile başlayan servis kodlarından seç.
            2. Kod kesinlikle 4 hanelidir ve listede olmalı; ASLA uydurma kod yazma.
            3. Kararını önce ANA SİNYAL'e göre ver. ANA SİNYAL iki/daha fazla kategori arasında net ayrım
               yapmana izin vermiyorsa, İKİNCİL SİNYAL'i kullanarak en olası kodu seç.
            4. Hiçbir sinyal makul bir eşleşme sağlamıyorsa KOD alanına "BULUNAMADI" yaz ve YUZDE'yi 0 yap.
            5. YUZDE, senin bu kodun doğru olduğuna dair gerçek olasılık tahminindir (0-100 arası tam sayı).
               Tanım net ve tek bir kodla güçlü şekilde örtüşüyorsa yüksek (90-100), birden fazla kod
               arasında zorlukla seçim yaptıysan düşük/orta (40-75) bir değer ver. Yüzdeyi rastgele
               yüksek yazma; gerçekten ne kadar emin olduğunu yansıtsın.

            CEVABINI KESİNLİKLE şu formatta ver, başka hiçbir açıklama/kelime ekleme:
            KOD;YUZDE

            Örnek: 3210;96
            Örnek: 2110;55
            """

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_message,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                )
            )

            raw_text = None
            if response and hasattr(response, 'text') and response.text:
                raw_text = response.text
            if not raw_text:
                try:
                    raw_text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError, TypeError, KeyError):
                    raw_text = None

            if not raw_text:
                try:
                    finish_reason = response.candidates[0].finish_reason
                except Exception:
                    finish_reason = "bilinmiyor"
                print(f"⚠️ Model boş cevap döndürdü. finish_reason={finish_reason}, "
                      f"model={self.model_name}, tanım='{user_description}'")
                return "Belirlenemedi"

            return self._parse_model_response(raw_text, user_description, similar_examples, cost_code_description)

        except Exception as e:
            return f"API-HATA: {str(e)}"

    def _parse_model_response(self, raw_text: str, user_description: str,
                               similar_examples: pd.DataFrame, cost_code_description: str) -> str:
        """'KOD;YUZDE' formatını parse eder; belirsizliği ve hataları gizlemez."""
        first_line = raw_text.strip().splitlines()[0].strip() if raw_text.strip() else ""
        parts = [p.strip() for p in first_line.split(';')]

        if len(parts) != 2:
            candidate = re.sub(r'[^A-Za-z0-9]', '', first_line).upper()
            if candidate in self.valid_codes:
                print(f"⚠️ Beklenmeyen format ama geçerli kod bulundu: '{first_line}'")
                return f"{candidate} (Format uyarısı: yüzde bilgisi alınamadı)"
            print(f"⚠️ Beklenmeyen model formatı: '{raw_text}' (tanım='{user_description}')")
            return "Belirlenemedi (model beklenmeyen formatta cevap verdi)"

        code = parts[0].strip().upper()
        try:
            yuzde = max(0, min(100, int(re.sub(r'[^0-9]', '', parts[1]) or 0)))
        except ValueError:
            yuzde = 0

        if code == "BULUNAMADI":
            return "Belirlenemedi"
        if self.valid_codes and code not in self.valid_codes:
            print(f"⚠️ Geçersiz kod önerildi: '{code}' (tanım='{user_description}')")
            return "Belirlenemedi (model listede olmayan bir kod önerdi)"

        # Teşhis amaçlı: önerilen kod ne ANA sinyalde ne de destekleyici sinyalde
        # yer almıyorsa (modelin tamamen kendi çıkarımı), konsola not düş.
        signal_codes = set(similar_examples['CategoryCode']) if not similar_examples.empty else set()
        if cost_code_description:
            norm = self._normalize(cost_code_description)
            signal_codes |= set(self.costdesc_category_counts.get(norm, {}).keys())
        if signal_codes and code not in signal_codes:
            print(f"ℹ️ Model, benzerlik/costcode sinyallerinde görünmeyen bir kod önerdi: "
                  f"'{code}' (tanım='{user_description}') - muhtemelen saf semantik çıkarım.")

        if yuzde >= self.HIGH_CONFIDENCE_THRESHOLD:
            return code
        return f"{code} (%{yuzde} olasılıkla)"

    def log_and_save_action(self, description, suggested_code, final_code, is_satisfied, username="web_user",
                            mr_no=None, line_no=None, unit="", cost_code="", cost_code_description=""):
        """Kullanıcı sonucu onayladığında/değiştirdiğinde SQL'e kayıt atar."""
        try:
            category_desc = self.unique_categories_map.get(final_code, "Yeni Onaylanan Kategori")
            verified_status = 1 if int(is_satisfied) == 1 else 0

            insert_query = text("""
                INSERT INTO SCM.dbo.CatCodeAI 
                (MR_No, Line_No, Description, CostCode, CostCodeDescription, UnitOfMeasurement, CategoryCode, CategoryCodeDescription, SuggestedCode, IsUserSatisfied, Username, Verified)
                VALUES 
                (:mr_no, :line_no, :description, :cost_code, :cost_code_description, :unit, :final_code, :category_desc, :suggested_code, :is_satisfied, :username, :verified_status)
            """)

            with self.engine.begin() as conn:
                conn.execute(insert_query, {
                    'mr_no': mr_no if mr_no else 'WEB_REQUEST',
                    'line_no': line_no if line_no else '1',
                    'description': description,
                    'cost_code': cost_code,
                    'cost_code_description': cost_code_description,
                    'unit': unit,
                    'final_code': final_code,
                    'category_desc': category_desc,
                    'suggested_code': suggested_code,
                    'is_satisfied': is_satisfied,
                    'username': username,
                    'verified_status': verified_status
                })

            if verified_status == 1:
                self.reload_learning_data()

        except Exception as e:
            print(f"❌ SQL Kayıt Hatası: {str(e)}")


# --- HIZLI TEST BLOĞU ---
if __name__ == "__main__":
    print("🔄 Sistem başlatılıyor ve SQL verileri yükleniyor...")
    ai_engine = CategoryAIExpress()

    test_metin = "silecek"
    print(f"\n🔍 Test sorgusu yapılıyor: '{test_metin}'")
    oneri = ai_engine.get_smart_suggestion(test_metin)
    print(f"🤖 Yapay Zeka Önerisi (sadece tanım): {oneri}")

    # CostCodeDescription / MR bilgisiyle zenginleştirilmiş örnek çağrı:
    # oneri2 = ai_engine.get_smart_suggestion(
    #     user_description="beton delici uc",
    #     cost_code_description="Elektrik Taşeron İşi",
    #     unit="ADET",
    #     mr_no="MR-2026-001",
    #     line_no="3",
    # )
    # print(f"🤖 Yapay Zeka Önerisi (zenginleştirilmiş): {oneri2}")