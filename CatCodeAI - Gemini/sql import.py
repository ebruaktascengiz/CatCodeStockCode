import os
import pandas as pd
from sqlalchemy import create_engine, text
import urllib
from dotenv import load_dotenv

# .env dosyasındaki bağlantı bilgilerini yükle
load_dotenv()

# Excel dosyasının adı ve konumu (Python dosyasıyla aynı klasörde olduğunu varsayıyoruz)
excel_file = "category_training.xlsx"

# SQL Bağlantı dizesi (.env dosyanızdan otomatik okunur)
conn_str = os.getenv("SQL_CONNECTION_STRING")
if not conn_str:
    # Eğer .env okunamazsa manuel buraya da yazabilirsiniz:
    conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=istrp01.enka.com;DATABASE=SCM;Trusted_Connection=yes;"

# SQLAlchemy motorunu oluştur
params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

try:
    print("Excel dosyası okunuyor")
    # Excel'i oku
    df = pd.read_excel(excel_file)

    # Kolon isimlerini temizle (başında/sonunda boşluk varsa gitsin)
    df.columns = df.columns.str.strip()

    # SQL tablonuzda yer alan ve Excel'de olmayan yeni durum kolonlarını ekliyoruz:
    # Bu veriler bizim temel eğitim kümemiz (baz verimiz) olacağı için hepsini Verified = 1 yapıyoruz.
    df['SuggestedCode'] = None
    df['IsUserSatisfied'] = None
    df['Username'] = 'SISTEM_BASLANGIC_YUKLEME'
    df['Verified'] = 1  # 1 olması yapay zekanın bu satırları referans almasını sağlar

    print("🧹 Veri tipleri ve boş değerler düzenleniyor...")
    # SQL uyumluluğu için NaN (boş) değerleri temizleme
    df['MR_No'] = df['MR_No'].astype(str).replace('nan', '')
    df['Line_No'] = df['Line_No'].astype(str).replace('nan', '1')
    df['CostCode'] = df['CostCode'].astype(str).replace('nan', '')
    df['CostCodeDescription'] = df['CostCodeDescription'].astype(str).replace('nan', '')
    df['UnitOfMeasurement'] = df['UnitOfMeasurement'].astype(str).replace('nan', '')
    df['CategoryCode'] = df['CategoryCode'].astype(str).replace('nan', '')
    df['CategoryCodeDescription'] = df['CategoryCodeDescription'].astype(str).replace('nan', '')

    print(" SQL Server'a aktarım başladı")

    # pyodbc hızlı yazma moduyla veriyi SQL tablosuna gönder (Tablo ismi küçük-büyük harfe duyarlı olabilir)
    df.to_sql(
        name='CatCodeAI',
        schema='dbo',
        con=engine,
        if_exists='append',  # Mevcut tablonun üzerine ekle (truncate etmez)
        index=False,
        chunksize=10000  # Onar binlik paketler halinde göndererek hafızayı korur ve hızlandırır
    )

    print("Aktarım başarıyla tamamlandı")

except FileNotFoundError:
    print(f"❌ Hata: '{excel_file}' dosyası script ile aynı klasörde bulunamadı!")
except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")