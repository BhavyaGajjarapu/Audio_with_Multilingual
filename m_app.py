from flask import Flask, render_template, request, jsonify
from m_main import semantic_search_audio_catalog, process_audio_blob, get_emotion_from_query
import os, json, logging, pyodbc
from datetime import datetime, timezone, timedelta
from m_main import autocorrect_query
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# === Load environment ===
load_dotenv()
app = Flask(__name__)

# === Azure Blob Storage ===
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "audiofiles")
AZURE_CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECT_STR)

def upload_audio_to_blob(file, filename):
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        logging.info(f"[AZURE] Uploaded {filename}")
        return True
    except Exception as e:
        logging.error(f"[AZURE ERROR] {e}")
        return False

def generate_audio_sas_url(blob_name, expiry_minutes=60):
    try:
        sas_token = generate_blob_sas(
            account_name=AZURE_ACCOUNT_NAME,
            container_name=AZURE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=AZURE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes)
        )
        return f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}?{sas_token}"
    except Exception as e:
        logging.error(f"[AZURE SAS ERROR] {e}")
        return None

# === Logging ===
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Database ===
DB_DRIVER = os.getenv("DB_DRIVER")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
connection_string = f'DRIVER={DB_DRIVER};SERVER={DB_HOST},{DB_PORT};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD}'

def get_db_connection():
    try:
        return pyodbc.connect(connection_string)
    except Exception as e:
        logging.error(f"[DB ERROR] {e}")
        return None
    
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  

LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "zh-cn": "Chinese (Simplified)",
    "zh": "Chinese",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "bn": "Bengali",
    "ur": "Urdu",
}

def detect_language_fullname(text: str) -> str:
    """
    Detect language and return full readable name (e.g. 'English', 'German').
    Falls back to 'Unknown' if detection fails.
    """
    try:
        code = detect(text).strip().lower()
        # normalize variants (e.g., 'zh-cn', 'zh-tw')
        if code.startswith("zh"):
            code = "zh"
        # lookup map
        full_name = LANGUAGE_MAP.get(code)
        if full_name:
            return full_name
        else:
            return code.upper()  # fallback like 'FI' if unknown code
    except Exception:
        return "Unknown"



# === Insert into Audio_search_logging ===
def insert_search_log(search_query, results=None, error_message=None, query_lang=None, translated_query=None):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()

        if error_message:
            status = error_message
            num_results = "No results due to error"
            matched_files = "None"
        elif not results or len(results) == 0:
            status = "No Results"
            num_results = "No results found"
            matched_files = "None"
        else:
            status = "Success"
            num_results = f"{len(results)} results found"
            matched_files = json.dumps([r.get("file_name") for r in results])

        cursor.execute("""
            INSERT INTO Audio_search_logging (
                user_query,
                query_language,
                translated_query,
                status,
                num_results,
                matched_files,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            search_query,
            query_lang or "unknown",
            translated_query if translated_query else search_query,
            status,
            num_results,
            matched_files,
            datetime.now(timezone.utc)
        ))
        conn.commit()
    except Exception as e:
        logging.error(f"[DB ERROR][LOGGING] {e}")
    finally:
        conn.close()



# === Insert into Audio_upload_logging ===
def insert_upload_log(filename, status, error_message=None):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Audio_upload_logging (FileName, UploadTime, Status, ErrorMessage)
            VALUES (?, ?, ?, ?)
        """, (
            filename,
            datetime.now(timezone.utc),
            status,
            error_message if error_message else "None"
        ))
        conn.commit()
    except Exception as e:
        logging.error(f"[DB ERROR][UPLOAD LOGGING] {e}")
    finally:
        conn.close()

def parse_tags_field(row_dict):
    tags_value = row_dict.get("tags")
    if not tags_value:
        return []
    try:
        parsed = json.loads(tags_value)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except:
        return [tags_value]

def get_all_metadata_for_listing(limit=1000):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TOP ({limit})
                file_name, tags, transcript, summary,
                upload_time, audio_url
            FROM audio_metadata
            ORDER BY upload_time DESC
        """)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        results = []
        for row in rows:
            item = dict(zip(columns, row))
            item["tags"] = parse_tags_field(item)
            item["timestamp"] = None
            results.append(item)
        return results
    except Exception as e:
        logging.error(f"[DB ERROR] {e}")
        return []
    finally:
        conn.close()

def get_all_metadata_for_search():
    """
    Returns list of metadata items. Ensures transcript_en is present (fallback to transcript).
    """
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_name, transcript, transcript_en, tags, audio_url, word_timestamp, emotion
            FROM audio_metadata
        """)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        results = []
        for row in rows:
            item = dict(zip(columns, row))

            # Parse tags
            item["tags"] = parse_tags_field(item)

            # Parse word timestamps
            if item.get("word_timestamp"):
                try:
                    item["word_timestamp"] = json.loads(item["word_timestamp"])
                except:
                    item["word_timestamp"] = []
            else:
                item["word_timestamp"] = []

            # Ensure emotion is normalized
            if item.get("emotion"):
                item["emotion"] = item["emotion"].lower()
            else:
                item["emotion"] = None

            # Ensure transcript_en fallback
            if not item.get("transcript_en"):
                item["transcript_en"] = item.get("transcript") or ""

            results.append(item)

        return results
    except Exception as e:
        logging.error(f"[DB ERROR] Failed to fetch search metadata: {e}")
        return []
    finally:
        conn.close()

def save_metadata_to_db(meta):
    conn = get_db_connection()
    if not conn:
        return False

    def coerce(value):
        if value in [None, "", [], {}]:
            return "Not available for the audio"
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    try:
        audio_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{meta.get('file_name')}"
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audio_metadata (
                file_name, tags, tag_probs, transcript, transcript_en, emotion,
                diarization, entities, summary,
                action_items, word_timestamp,
                upload_time, audio_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            coerce(meta.get("file_name")),
            coerce(meta.get("tags")),
            coerce(meta.get("tag_probs")),
            coerce(meta.get("transcript")),
            coerce(meta.get("transcript_en")),
            coerce(meta.get("emotion")),
            coerce(meta.get("diarization")),
            coerce(meta.get("entities")),
            coerce(meta.get("summary")),
            coerce(meta.get("action_items")),
            coerce(meta.get("word_timestamp")),
            datetime.now(timezone.utc),
            audio_url
        ))
        conn.commit()
        return True
    except Exception as e:
        logging.exception(f"[DB ERROR] {e}")
        return False
    finally:
        conn.close()

@app.route('/autocorrect', methods=['POST'])
def autocorrect_api():
    data = request.get_json()
    word = data.get("word", "")
    corrected = autocorrect_query(word)
    return jsonify({"corrected": corrected})

def process_and_add_audio(blob_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
        _ = blob_client.download_blob().readall()
        meta = process_audio_blob(blob_name)
        save_metadata_to_db(meta)
    except Exception as e:
        logging.error(f"[PROCESS ERROR] {e}")

# === Routes ===
from langdetect import detect

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    query = request.form.get('query', '').strip()

    if request.method == 'POST' and 'audiofile' in request.files:
        files = request.files.getlist('audiofile')
        messages = []
        existing_files = [item["file_name"] for item in get_all_metadata_for_listing(limit=5000)]
        for file in files:
            if file and file.filename:
                filename = file.filename
                if filename in existing_files:
                    messages.append(f"File '{filename}' already exists.")
                    insert_upload_log(filename, "Error", "File already exists")
                else:
                    if upload_audio_to_blob(file, filename):
                        try:
                            process_and_add_audio(filename)
                            insert_upload_log(filename, "Success")
                            messages.append(f"Uploaded & processed: {filename}")
                        except Exception as e:
                            insert_upload_log(filename, "Error", str(e))
                            messages.append(f"Processing failed: {filename}")
                    else:
                        insert_upload_log(filename, "Error", "Upload to Azure failed")
                        messages.append(f"Upload failed: {filename}")
        message = " | ".join(messages)

    if query:
        all_data = get_all_metadata_for_search()
        corrected_query = autocorrect_query(query)

        # Detect language
        query_lang = detect_language_fullname(query)

        try:
            results, translated_query = semantic_search_audio_catalog(corrected_query, all_data)
            insert_search_log(query, results, query_lang=query_lang, translated_query=translated_query)
        except Exception as e:
            insert_search_log(query, None, str(e), query_lang=query_lang)
            results = []
    else:
        results = get_all_metadata_for_listing()

    for item in results:
        item["stream_url"] = generate_audio_sas_url(item.get("file_name"))
        ts = item.get("timestamp")
        if ts is not None:
            try:
                ts = float(ts)
                if ts <= 0:
                    ts = None
            except:
                ts = None
        item["timestamp"] = ts

    return render_template('a_index.html', results=results, query=query, message=message)


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    all_data = get_all_metadata_for_search()
    corrected_query = autocorrect_query(query)

    # Detect language
    query_lang = detect_language_fullname(query)

    try:
        results, translated_query = semantic_search_audio_catalog(corrected_query, all_data)
        insert_search_log(query, results, query_lang=query_lang, translated_query=translated_query)
    except Exception as e:
        insert_search_log(query, None, str(e), query_lang=query_lang)
        results = []

    for item in results:
        item["stream_url"] = generate_audio_sas_url(item.get("file_name"))
        ts = item.get("timestamp")
        if ts is not None:
            try:
                ts = float(ts)
                if ts <= 0:
                    ts = None
            except:
                ts = None
        item["timestamp"] = ts
    return render_template('a_index.html', results=results, query=query)


if __name__ == "__main__":
    app.run(debug=True)
