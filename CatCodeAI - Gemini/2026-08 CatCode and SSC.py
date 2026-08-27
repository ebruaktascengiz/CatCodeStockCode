import os
import re
import urllib
import warnings
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

# SSL Güvenlik Yaması (Monkey-Patch)
import httpx

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

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


# Structured Output (JSON Şeması) Tanımı
class AICategoryResponse(BaseModel):
    category_code: str = Field(
        description="Geçerli kategori listesinden seçilen 4 haneli kategori kodu (Örn: FB02, DB01, M0301)")
    confidence: int = Field(description="0-100 arası tahmin güven skoru")
    root_code: str = Field(
        description="Malzemenin İngilizce karşılığının anlamlı 3 karakterlik BÜYÜK HARFLİ kısaltması. Örn: Bearing -> BRG, Cable -> CBL, Valve -> VLV, Pipe -> PIP, Belt -> BLT, Seal/Gasket -> SEAL/GST")


class CategoryAIExpress:
    GENERIC_KEYWORDS = {
        'genel', 'general', 'diger', 'muhtelif', 'misc', 'miscellaneous',
        'na', 'n/a', 'yok', 'belirsiz', 'other', 'others', 'various', 'unknown', 'bilinmiyor'
    }

    HIGH_CONFIDENCE_THRESHOLD = 90

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY .env dosyasında bulunamadı!")

        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

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
        """SQL'den Verified=1 verileri çekip belleğe yükler ve TF-IDF modelini eğitir."""
        try:
            query = """
                SELECT MR_No, Line_No, Description, CostCode, CostCodeDescription, 
                       UnitOfMeasurement, CategoryCode, CategoryCodeDescription,
                       SupplyChainCode 
                FROM SCM.dbo.CatCodeAI 
                WHERE Verified = 1 AND Description IS NOT NULL AND CategoryCode IS NOT NULL
            """
            with self.engine.connect() as conn:
                self.df_verified = pd.read_sql(text(query), conn)

            if self.df_verified.empty:
                print("⚠ SQL'de Verified=1 olan kayıt bulunamadı.")
                return

            self.df_verified['CategoryCode'] = self.df_verified['CategoryCode'].astype(str).str.strip()
            self.df_verified['MR_No'] = self.df_verified['MR_No'].astype(str).str.strip()
            self.df_verified['Line_No'] = self.df_verified['Line_No'].astype(str).str.strip()

            df_clean_cats = self.df_verified.dropna(subset=['CategoryCode', 'CategoryCodeDescription'])
            df_unique_cats = df_clean_cats.drop_duplicates(subset=['CategoryCode'])
            self.unique_categories_map = df_unique_cats.set_index('CategoryCode')['CategoryCodeDescription'].to_dict()
            self.valid_codes = set(self.df_verified['CategoryCode'])

            self.costdesc_category_counts = {}
            norm_desc = self.df_verified['CostCodeDescription'].fillna('').map(self._normalize)
            for norm, group_idx in norm_desc.groupby(norm_desc).groups.items():
                if not norm or norm in self.GENERIC_KEYWORDS:
                    continue
                cats = self.df_verified.loc[group_idx, 'CategoryCode']
                self.costdesc_category_counts[norm] = Counter(cats)

            self.df_verified['SearchableText'] = (
                    self.df_verified['Description'].str.lower().fillna('') + ' ' +
                    self.df_verified['UnitOfMeasurement'].str.lower().fillna('') + ' ' +
                    self.df_verified['CostCodeDescription'].str.lower().fillna('') + ' ' +
                    self.df_verified['CategoryCodeDescription'].str.lower().fillna('')
            )

            self.text_vectors = self.vectorizer.fit_transform(self.df_verified['SearchableText'])
            print(f" Başarılı: {len(self.df_verified)} öğrenme kaydı veritabanından yüklendi.")

        except Exception as e:
            print(f"Veri yüklenirken SQL hatası oluştu: {str(e)}")

    def _find_existing_supply_chain_code(self, normalized_desc: str) -> str:
        """[SCM].[dbo].[supplychaincode] tablosundan daha önce aynı açıklama için üretilmiş kod var mı bakar."""
        try:
            sql = text("""
                SELECT TOP 1 SupplyChainCode 
                FROM SCM.dbo.supplychaincode 
                WHERE NormalizedDescription = :norm_desc
            """)
            with self.engine.connect() as conn:
                res = conn.execute(sql, {"norm_desc": normalized_desc}).fetchone()
                if res:
                    return res[0]
        except Exception as e:
            print(f"⚠ Supply chain code sorgulama hatası: {str(e)}")
        return None

    def _generate_and_save_supply_chain_code(self, category_code: str, root_code: str, normalized_desc: str,
                                             original_desc: str) -> str:
        """
        Gelen Kategori Kodu ve Kök Koda göre:
        Format: {CategoryCode}.{RootCode}-{SequenceNo:06d}
        Örnek: FB02.BRG-000001
        """
        try:
            category_code = category_code.upper().strip()
            root_code = root_code.upper().strip()

            # İlgili (CategoryCode + RootCode) için en son kullanılan sıra numarasını al ve +1 artır
            sql_seq = text("""
                SELECT ISNULL(MAX(SequenceNo), 0) + 1 
                FROM SCM.dbo.supplychaincode 
                WHERE CategoryCode = :cat_code AND RootCode = :root_code
            """)

            with self.engine.begin() as conn:
                next_seq = conn.execute(sql_seq, {"cat_code": category_code, "root_code": root_code}).scalar()

                # Birebir İstenen Format Yapısı
                scm_code = f"{category_code}.{root_code}-{next_seq:06d}"

                # Veritabanına kaydet
                insert_sql = text("""
                    INSERT INTO SCM.dbo.supplychaincode 
                    (SupplyChainCode, CategoryCode, RootCode, SequenceNo, NormalizedDescription, OriginalDescription)
                    VALUES (:scm_code, :cat_code, :root_code, :seq, :norm_desc, :orig_desc)
                """)
                conn.execute(insert_sql, {
                    "scm_code": scm_code,
                    "cat_code": category_code,
                    "root_code": root_code,
                    "seq": next_seq,
                    "norm_desc": normalized_desc,
                    "orig_desc": original_desc
                })

                print(f" Yeni Supply Chain Code üretildi ve veritabanına kaydedildi: {scm_code}")
                return scm_code

        except Exception as e:
            print(f"⚠ Supply Chain Code kaydetme hatası: {str(e)}")
            return f"{category_code}.{root_code}-000001"

    def find_most_similar_examples(self, user_description: str, cost_code_description: str = None,
                                   unit: str = None, top_n: int = 5):
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
        lines = []
        if cost_code_description:
            norm = self._normalize(cost_code_description)
            counts = self.costdesc_category_counts.get(norm)
            if self._is_generic(cost_code_description):
                lines.append(f"- CostCode açıklaması ('{cost_code_description}') genel/belirsiz.")
            elif counts:
                total = sum(counts.values())
                top3 = counts.most_common(3)
                dist = ", ".join(f"{code} (%{round(100 * cnt / total)})" for code, cnt in top3)
                lines.append(f"- '{cost_code_description}' geçmiş kod dağılımı: {dist}.")

        if mr_no:
            sibling_df = self.df_verified[self.df_verified['MR_No'] == str(mr_no).strip()]
            if line_no:
                sibling_df = sibling_df[sibling_df['Line_No'] != str(line_no).strip()]
            if not sibling_df.empty:
                items = "; ".join(
                    f"'{r['Description']}'->{r['CategoryCode']}" for _, r in sibling_df.head(8).iterrows())
                lines.append(f"- Aynı MR paketindeki diğer satırlar: {items}")

        return "\n".join(lines) if lines else "Ek bağlam bilgisi yok."

    def get_smart_suggestion(self, user_description: str, cost_code_description: str = None,
                             unit: str = None, mr_no: str = None, line_no: str = None) -> dict:
        try:
            norm_user_desc = self._normalize(user_description)

            # 1. KONTROL: Veritabanında daha önceden oluşmuş bir stok kodu var mı?
            existing_scm_code = self._find_existing_supply_chain_code(norm_user_desc)
            if existing_scm_code:
                print(f"ℹ Mevcut Supply Chain Code veritabanından çekildi: {existing_scm_code}")

            similar_examples = self.find_most_similar_examples(
                user_description, cost_code_description=cost_code_description, unit=unit, top_n=5
            )

            similarity_context = "Doğrudan eşleşen geçmiş kayıt yok."
            if not similar_examples.empty:
                example_texts = [
                    f"- '{r['Description']}' --> KOD: {r['CategoryCode']}"
                    for _, r in similar_examples.iterrows()
                ]
                similarity_context = "Benzer Kayıtlar:\n" + "\n".join(example_texts)

            supporting_context = self._build_supporting_context(cost_code_description, mr_no, line_no)
            formatted_categories_str = "\n".join(f"{code}: {name}" for code, name in sorted(
                self.unique_categories_map.items())) if self.unique_categories_map else "Liste yüklenemedi."

            prompt_message = f"""
            Sen kurumsal bir satınalma katalog ve ambardaki malzeme stok kodu (Supply Chain Code) uzmanısın.

            GÖREVİN:
            1. Verilen malzeme tanımına en uygun 4 haneli Kategori Kodunu (category_code) belirlemek.
            2. Bu malzemenin stok sınıflandırması (Kök Kod - root_code) için malzemenin İngilizce kelime karşılığının 3 veya 4 karakterlik İNGİLİZCE BÜYÜK HARFLİ kısaltmasını türetmek.

            KÖK KOD (root_code) KURALLARI (İNGİLİZCE KISALTMA OLMALIDIR):
            - Rulman / Yatak -> Bearing -> BRG
            - V Kayışı / Kayış -> Belt -> BLT
            - Conta / Sızdırmazlık -> Gasket / Seal -> GST veya SEAL
            - Kablo -> Cable -> CBL
            - Priz -> Socket / Outlet -> SKT
            - Sigorta -> Fuse / Breaker -> BKR veya FUS
            - Vana -> Valve -> VLV
            - Boru -> Pipe -> PIP
            - İşçilik / Montaj -> Installation / Assembly -> ASM veya MNT
            - Pompa -> Pump -> PMP
            - Civata / Somun -> Bolt / Nut -> BLT

            Sınıflandırılacak Malzeme Tanımı: "{user_description}"
            Geçmiş Benzerlik Bilgisi: {similarity_context}
            Ek Bağlam Bilgisi: {supporting_context}

            GEÇERLİ KATEGORİ LİSTESİ:
            {formatted_categories_str}

            KURALLAR:
            1. category_code kesinlikle listede bulunan geçerli bir 4 haneli kod olmalıdır.
            2. root_code kesinlikle malzemenin İngilizce isminin 3-4 karakterlik BÜYÜK HARFLİ kısaltması olmalıdır.
            3. confidence (0-100) tahminine ne kadar güvendiğini gösterir.
            """

            # Yapılandırılmış JSON Modu (Structured Outputs)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_message,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AICategoryResponse,
                    temperature=0.0,
                )
            )

            # JSON Parsing
            try:
                ai_data = AICategoryResponse.model_validate_json(response.text)
            except Exception as parse_err:
                print(f"⚠ JSON Ayrıştırma Hatası: {str(parse_err)} - Ham Yanıt: {response.text}")
                return {"category_code": "Format Hatası", "confidence": 0, "root_code": "GEN",
                        "supply_chain_code": "Belirlenemedi"}

            category_code = ai_data.category_code.strip().upper()
            yuzde = ai_data.confidence
            root_code = re.sub(r'[^A-Z0-9]', '', ai_data.root_code.strip().upper()) or "GEN"

            # Kategori Kodunun Doğrulanması
            if self.valid_codes and category_code not in self.valid_codes:
                print(f"⚠ AI listede olmayan bir kategori kodu önerdi: {category_code}")
                return {"category_code": "Geçersiz Kod", "confidence": 0, "root_code": root_code,
                        "supply_chain_code": "Belirlenemedi"}

            # 2. AŞAMA: Supply Chain Code Yönetimi ({CategoryCode}.{RootCode}-{Seq})
            if existing_scm_code:
                final_scm_code = existing_scm_code
            else:
                final_scm_code = self._generate_and_save_supply_chain_code(
                    category_code=category_code,
                    root_code=root_code,
                    normalized_desc=norm_user_desc,
                    original_desc=user_description
                )

            return {
                "category_code": category_code,
                "confidence": yuzde,
                "root_code": root_code,
                "supply_chain_code": final_scm_code
            }

        except Exception as e:
            print(f"⚠ get_smart_suggestion içinde hata: {str(e)}")
            return {"error": f"API/Sistem Hatası: {str(e)}"}


# --- TEST ÇALIŞTIRMASI ---
if __name__ == "__main__":
    print("🚀 Sistem başlatılıyor ve veritabanı verileri yükleniyor...")
    ai_engine = CategoryAIExpress()

    test_metin = "çelik burunlu kış için iş güvenliği ayakkabısı"
    print(f"\n1. Test Sorgusu (İlk defa soruluyor): '{test_metin}'")

    sonuc = ai_engine.get_smart_suggestion(test_metin)

    print("\n--- YAPAY ZEKA VE SCM ÇIKTISI ---")
    print(f"📌 Kategori Kodu       : {sonuc.get('category_code')}")
    print(f"🎯 Güven Skoru (%)      : %{sonuc.get('confidence')}")
    print(f"🏷 Kök Kod (Root)       : {sonuc.get('root_code')}")
    print(f"📦 Supply Chain Code    : {sonuc.get('supply_chain_code')}")

    print(f"\n2. Test Sorgusu (Aynı Tanım İkinci Kez Soruluyor): '{test_metin}'")
    sonuc_tekrar = ai_engine.get_smart_suggestion(test_metin)
    print(f"📦 Supply Chain Code    : {sonuc_tekrar.get('supply_chain_code')}")