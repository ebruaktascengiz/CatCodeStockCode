import os
import uuid
import pyodbc
import numpy as np
from openai import OpenAI
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# --- KONFİGÜRASYON ---
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
DB_SERVER = 'istrp01.enka.com'
DB_DATABASE = 'SCM'
DB_DRIVER = '{ODBC Driver 17 for SQL Server}'

# --- UYGULAMA BAŞLANGICI ---
client = OpenAI()
app = Flask(__name__)

# --- ANLAMSAL MODEL ---
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# --- KILAVUZ VERİSİ ---
documents = []
doc_embeddings = None
try:
    with open('guide.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        documents = [line.strip() for line in lines if line.strip()]
        doc_embeddings = semantic_model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
except Exception as e:
    print(f"Guide yüklenemedi: {e}")

# --- VERİTABANI ---
def get_db_connection():
    try:
        conn_str = f'DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};Trusted_Connection=yes;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"DB bağlantı hatası: {e}")
        return None

def log_to_db(session_id, user_name, user_email, project_code, user_message, bot_response):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = "INSERT INTO SCM.dbo.EGPSChatbot (SessionID, UserName, UserEmail, ProjectCode, UserMessage, BotResponse) VALUES (?, ?, ?, ?, ?, ?)"
            cursor.execute(sql, session_id, user_name, user_email, project_code, user_message, bot_response)
            conn.commit()
        except Exception as e:
            print(f"DB kayıt hatası: {e}")
        finally:
            conn.close()

# --- ANLAMSAL ARAMA ---
def find_relevant_docs(query, top_k=3, threshold=0.6):
    if doc_embeddings is None:
        return []
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results_indices = np.argsort(-cos_scores)[:top_k]
    relevant_docs = []
    for idx in top_results_indices:
        score = cos_scores[idx].item()
        if score >= threshold:
            relevant_docs.append(documents[idx.item()])
    return relevant_docs

# --- GPT CEVAP OLUŞTURMA ---
def get_chat_response(query, context_docs, language):
    system_prompt = f"""
    Sen ENKA satınalma sistemi EGPS için uzman bir asistansın. Görevin, kullanıcının sorduğu soruyu sana verilen bağlam metinlerinden hareketle cevaplamaktır.

    Kullanıcının Sorusu: {query}

    Bağlam:
    ---
    {'\n---\n'.join(context_docs)}
    ---

    Sadece bağlamda açıkça bulunan bilgilerle, kullanıcının dilinde ({language}) kısa ve net bir cevap ver. Bilgi yoksa: 'Bu konuda bilgim bulunmuyor, daha fazla yardım için egpssupport@enka.com adresine e-posta gönderebilirsiniz.'
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API hatası: {e}")
        return "Üzgünüm, bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

# --- YARDIMCI ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'tr'

# --- API ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    return jsonify({"sessionId": str(uuid.uuid4())})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('sessionId')
    user_info = data.get('userInfo', {})

    language = detect_language(user_message)
    relevant_docs = find_relevant_docs(user_message)

    if not relevant_docs:
        bot_response = "Bu konuda bilgim bulunmuyor, daha fazla yardım için egpssupport@enka.com adresine e-posta gönderebilirsiniz."
    else:
        bot_response = get_chat_response(user_message, relevant_docs, language)

    if session_id:
        log_to_db(session_id, user_info.get('name'), user_info.get('email'), user_info.get('projectCode'), user_message, bot_response)

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
