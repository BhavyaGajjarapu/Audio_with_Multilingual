import os
import json
import re
import tempfile
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import whisper
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from panns_inference import AudioTagging
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from pyannote.audio import Pipeline
from datetime import datetime, timezone
import string
from collections import Counter
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# Fix langdetect randomness
DetectorFactory.seed = 0

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# === Azure Storage ===
from azure.storage.blob import BlobServiceClient
AZURE_CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "audiofiles")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECT_STR)

# === FFMPEG Setup ===
AudioSegment.converter = "ffmpeg"  # ensure ffmpeg is in PATH

# === Models ===
# Use a multilingual embedder for cross-lingual semantic search
try:
    model_embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
except Exception as e:
    # fallback to English model if multilingual not available
    model_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Spacy English for NER (we run NER on English-translated text)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    # If model not present, fallback to a blank English model
    from spacy.lang.en import English
    nlp = English()

# Whisper multilingual (small) for language detection and transcription
try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    whisper_model = whisper.load_model("small")  


from transformers import MarianMTModel, MarianTokenizer

# 🌍 Lightweight multilingual translator (90+ languages → English)
try:
    fallback_model_name = "Helsinki-NLP/opus-mt-mul-en"
    fallback_tokenizer = MarianTokenizer.from_pretrained(fallback_model_name)
    fallback_model = MarianMTModel.from_pretrained(fallback_model_name)
    print("[INFO] ✅ Loaded multilingual translation model (Helsinki-NLP/opus-mt-mul-en)")
except Exception as e:
    fallback_model = None
    fallback_tokenizer = None
    print(f"[WARN] ⚠️ Could not load translation model: {e}")



# === Sentiment/emotion (text based) ===
try:
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    sentiment_classifier = None

# === Diarization ===
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=HF_TOKEN
    )
except Exception as e:
    diarization_pipeline = None
    print(f"[WARN] Diarization model not loaded: {e}")

def run_diarization(path):
    """Runs speaker diarization and returns speaker segments."""
    if not diarization_pipeline:
        return "Diarization model not available"
    diarization = diarization_pipeline(path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return segments

# === Summarization ===
try:
    summary_extractor = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    summary_extractor = None
    print(f"[WARN] Summarizer not loaded: {e}")

def generate_summary(text: str):
    if not text or not text.strip():
        return "No summary available."
    if not summary_extractor:
        # fallback: simple truncation if no model
        return (text.strip()[:300] + "...") if len(text) > 300 else text.strip()
    prompt = f"Summarize the following text in 1-2 sentences:\n\n{text}\n\nSummary:"
    try:
        response = summary_extractor(prompt, max_length=256, do_sample=False)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"Summary failed: {e}"

# === Action Item Extraction ===
try:
    action_extractor = pipeline("text2text-generation", model="google/flan-t5-large")
except Exception as e:
    action_extractor = None
    print(f"[WARN] Action extractor not loaded: {e}")

def extract_action_items(text: str):
    if not text or not text.strip():
        return ["No clear action items found."]
    if not action_extractor:
        return ["Action extraction model unavailable."]
    prompt = (
        "From the following text, extract only actionable tasks. "
        "A task is something someone must do (e.g., send, prepare, review, update). "
        "Give 1 action item only.\n\n"
        f"{text}\n\nAction items:\n-"
    )
    try:
        response = action_extractor(prompt, max_length=300, min_length=20, do_sample=False)
        extracted = response[0]["generated_text"].strip()
        items = [item.strip(" -•") for item in extracted.split("\n") if item.strip()]
        return items if items else ["No clear action items found."]
    except Exception as e:
        return [f"Action item extraction failed: {e}"]

# === Audio Tagging (PANNs) ===
def detect_audio_tags(path):
    import csv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tagger = AudioTagging(device=device)
    waveform, _ = librosa.load(path, sr=32000, mono=True)
    waveform = waveform[None, :]
    result = tagger.inference(waveform)
    output = result[0]
    if isinstance(output, np.ndarray):
        probs = output
    else:
        probs = output.detach().cpu().numpy()
    # Adjust the CSV path to your environment; keep original path for compatibility
    class_csv = os.getenv("PANN_CLASS_CSV", "C:/Users/Bhavya Gajjarapu/panns_data/class_labels_indices.csv")
    try:
        with open(class_csv) as f:
            reader = csv.DictReader(f)
            class_names = [r["display_name"] for r in reader]
    except Exception:
        # if csv not found, provide generic labels
        class_names = [f"tag_{i}" for i in range(len(probs))]
    probs = probs[0] if probs.ndim > 1 else probs
    top_indices = np.argsort(probs)[::-1][:5]
    return [(class_names[i], float(probs[i])) for i in top_indices]

# === Transcription ===
def transcribe_with_timestamps(path):
    """
    Uses Whisper to transcribe and detect language.
    Returns: (text, word_data, detected_language_code)
    """
    # We use the base small model which outputs language and segments
    try:
        result = whisper_model.transcribe(path, word_timestamps=True)
    except Exception:
        result = whisper_model.transcribe(path, word_timestamps=True)  # try again
    transcript = result.get("text", "")
    language = result.get("language", "en")
    word_data = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word_data.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"]
            })
    return transcript, word_data, language

# === Speech Emotion Recognition (Audio-based) ===
speech_emotion = pipeline(
    task="audio-classification",
    model="superb/hubert-large-superb-er"
)

def emotion_recognition(path):
    """
    Detects the dominant emotion/mood of the audio.
    - If it's speech → run Speech Emotion Recognition (SER).
    - If it's music → use PANNs tags for mood/genre.
    """
    try:
        tags = detect_audio_tags(path)

        # Check if the audio is speech
        if any("speech" in t.lower() for t, _ in tags):
            preds = speech_emotion(path)
            return preds[0]['label'] if preds else "neutral"

        # Check if the audio is music
        elif any("music" in t.lower() for t, _ in tags):
            # Return the top tag, e.g. "sad music", "classical music"
            return tags[0][0]

        # Fallback: if neither speech nor music detected
        else:
            return "neutral"

    except Exception as e:
        return f"emotion_failed: {e}"

def map_emotion(raw_label):
    """Maps raw emotion labels to common categories."""
    mapping = {
        "joy": "joyful/happy",
        "happy": "happy",
        "hap": "happy",
        "sadness": "sad",
        "sad": "sad",
        "ang": "angry",
        "anger": "angry",
        "angry": "angry",
        "fear": "fearful",
        "fearful": "fearful",
        "disgust": "disgust",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    try:
        label = raw_label.lower()
    except Exception:
        return "neutral"
    return mapping.get(label, "neutral")

EMOTION_SYNONYMS = {
    "angry": ["angry", "furious", "enraged", "mad", "irritated"],
    "happy": ["happy", "joyful", "cheerful", "elated"],
    "sad": ["sad", "down", "depressed", "unhappy"],
    "fearful": ["fearful", "scared", "frightened", "anxious"],
    "disgust": ["disgusted", "repulsed", "nauseated"],
    "surprise": ["surprised", "astonished", "amazed"],
    "neutral": ["neutral", "calm", "indifferent"]
}

def get_emotion_from_query(query):
    """Match query to known emotions (works across languages if English words are used)."""
    query = (query or "").lower()
    for main_emotion, synonyms in EMOTION_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in query:
                return main_emotion
    return None

def analyze_text(text):
    """Return list of (text, label) from Spacy's NER (run on English text)."""
    if not text:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def aggregate_conversational_sentiment(conv_sentiment):
    if not conv_sentiment or "error" in conv_sentiment[0]:
        return {"overall_sentiment": "Unknown", "top_emotions": []}
    sentiments = []
    emotions = []
    for entry in conv_sentiment:
        if "sentiment" in entry:
            sentiments.append(entry["sentiment"])
        if "emotions" in entry:
            emotions.extend([emo["label"] for emo in entry["emotions"]])
    sentiment_counter = Counter(sentiments)
    overall_sentiment = sentiment_counter.most_common(1)[0][0] if sentiments else "Neutral"
    emotion_counter = Counter(emotions)
    top_emotions = [emo for emo, _ in emotion_counter.most_common(3)]
    return {"overall_sentiment": overall_sentiment, "top_emotions": top_emotions}

# === Azure File Download & Processing ===
def download_blob_to_tempfile(blob_name):
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    download_stream = blob_client.download_blob()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_file.write(download_stream.readall())
    tmp_file.close()
    return tmp_file.name

from spellchecker import SpellChecker
spell = SpellChecker()

def autocorrect_query(query: str) -> str:
    """
    Auto-corrects spelling ONLY if the query language is English.
    Avoids distorting words in other languages.
    """
    if not query or not query.strip():
        return query

    try:
        lang_code = detect_language_code(query)
    except Exception:
        lang_code = "en"

    # ✅ Skip autocorrect if not English
    if lang_code.lower() != "en":
        return query

    words = query.split()
    corrected_words = []
    for word in words:
        if word.isnumeric() or len(word) <= 2:
            corrected_words.append(word)
            continue
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word if corrected_word else word)
    return " ".join(corrected_words)


# === Translation via MBART50 ===
# Map detected language to MBART src_lang token codes that MBART50 uses
MBART_SRC_MAP = {
    # a small helpful mapping (extend as needed)
    "hi": "hi_IN",
    "en": "en_XX",
    "es": "es_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    "pt": "pt_XX",
    "it": "it_IT",
    "nl": "nl_XX",
    "ru": "ru_RU",
    "zh-cn": "zh_CN",
    "zh": "zh_CN",
    "ar": "ar_AR"
}

def translate_to_english(text: str) -> str:
    """
    Fast multilingual translation to English using Helsinki-NLP opus-mt-mul-en.
    Works well for short queries.
    """
    if not text or not text.strip():
        return text
    if not fallback_model or not fallback_tokenizer:
        return text

    try:
        batch = fallback_tokenizer([text], return_tensors="pt", padding=True)
        generated = fallback_model.generate(**batch, max_length=128, num_beams=4)
        translation = fallback_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translation.strip()
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return text




def detect_language_code(text: str):
    """
    Detects language code using langdetect; returns short code like 'en','es','hi' etc.
    """
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"

# === Clean transcript ===
def clean_transcript(text):
    if not text:
        return ""
    words = text.split()
    seen = set()
    cleaned_words = []
    for w in words:
        w_clean = re.sub(r"[^\w\s]", "", w)
        if w_clean not in seen:
            cleaned_words.append(w)
            seen.add(w_clean)
    return " ".join(cleaned_words)

# === NEW FUNCTION: Universal Query Translator ===
def translate_query_if_needed(query: str) -> str:
    """
    Detects if query is non-English and translates it to English.
    """
    if not query or not query.strip():
        return query
    try:
        lang_code = detect_language_code(query)
        print(f"[INFO] Detected query language: {lang_code}")
        if lang_code.lower() != "en":
            translated_query = translate_to_english(query)
            print(f"[INFO] Translated query to English: {translated_query}")
            return translated_query
        return query
    except Exception as e:
        print(f"[WARN] Query translation failed: {e}")
        return query




# === Main Audio Processor ===
def process_audio_blob(blob_name):
    """
    Downloads blob, processes audio, returns metadata including transcript_en and language.
    """
    temp_path = download_blob_to_tempfile(blob_name)
    file_id = os.path.splitext(os.path.basename(temp_path))[0]
    audio = AudioSegment.from_file(temp_path).set_frame_rate(16000).set_channels(1)
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio.export(wav_path, format="wav")

    tags = detect_audio_tags(wav_path)
    transcript, word_data, detected_lang = transcribe_with_timestamps(wav_path)

    # Clean transcript
    transcript_clean = clean_transcript(transcript)

    # Determine MBART src code if possible
    src_lang_token = MBART_SRC_MAP.get(detected_lang.lower(), None)

    # Translate to English if not english
    if detected_lang and detected_lang.lower() != "en":
        transcript_en = translate_to_english(transcript_clean)
    else:
        transcript_en = transcript_clean

    remotion = emotion_recognition(wav_path)
    emotion = map_emotion(remotion)
    entities = analyze_text(transcript_en)
    diarization = run_diarization(wav_path)
    summary = generate_summary(transcript_en)
    action_items = extract_action_items(transcript_en)

    return {
        "file_name": blob_name,
        "tags": [t for t, _ in tags],
        "tag_probs": {t: p for t, p in tags},
        "transcript": transcript,               # original transcript (possibly non-en)
        "transcript_en": transcript_en,         # English normalized transcript
        "language": detected_lang,
        "emotion": emotion,
        "entities": entities,
        "word_timestamp": word_data,
        "diarization": diarization,
        "summary": summary,
        "action_items": action_items,
        "upload_time": datetime.now(timezone.utc).isoformat()
    }

# === NLP / Search Utilities ===
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

import re, string, unicodedata
from sentence_transformers import util

def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize accents."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()

def semantic_search_audio_catalog(query, catalog, threshold=0.4):
    """
    Enhanced multilingual + emotion-aware semantic search.

    Fixes:
      ✓ Works even if query partially matches transcript.
      ✓ Properly translates non-English queries.
      ✓ Combines semantic + keyword scores.
      ✓ Emotion detection integrated with context.
      ✓ Robust timestamp word-matching.
    """
    import re, string
    from sentence_transformers import util

    if not query or not catalog:
        return []

    results = []

    # --- Step 1: Auto-translate non-English query to English ---
    # ✅ FIX: Translate BEFORE emotion detection or embedding
    query_for_search = translate_query_if_needed(query)

    # --- Step 2: Detect emotion from the translated query ---
    # ✅ FIX: use translated query for emotion detection
    detected_emotion = get_emotion_from_query(query_for_search)
    emotion_matched_items = []

    # --- Step 3: Embed translated query ---
    try:
        query_emb = model_embedder.encode(query_for_search, convert_to_tensor=True)
    except Exception:
        query_emb = None

    print(f"[INFO] Final query used for search: {query_for_search}")
    print(f"[INFO] Detected emotion keyword: {detected_emotion}")

    # --- Step 4: Loop through catalog items ---
    for item in catalog:
        transcript_en = (item.get("transcript_en") or item.get("transcript") or "").strip()
        tags = " ".join(item.get("tags", []))
        combined_text = f"{tags} {transcript_en}".strip()

        # --- Skip if nothing to compare ---
        if not combined_text:
            continue

        # --- Emotion filter priority ---
        item_emotion = (item.get("emotion") or "").lower()
        if detected_emotion and detected_emotion.lower() in item_emotion:
            item_copy = dict(item)
            item_copy["match_score"] = 1.0
            item_copy["matched"] = f"Matched emotion: {detected_emotion}"
            item_copy["timestamp"] = None
            emotion_matched_items.append(item_copy)
            continue

        # --- Compute semantic similarity ---
        semantic_score = 0.0
        if query_emb is not None:
            try:
                combined_emb = model_embedder.encode(combined_text, convert_to_tensor=True)
                semantic_score = float(util.cos_sim(query_emb, combined_emb).item())
            except Exception:
                semantic_score = 0.0

        # --- Compute keyword-based score (fallback or blend) ---
        keyword_score = 0.0
        if query_for_search.lower() in combined_text.lower():
            keyword_score = 0.8
        else:
            query_words = re.findall(r'\b\w+\b', query_for_search.lower())
            transcript_words = re.findall(r'\b\w+\b', combined_text.lower())
            overlap = len(set(query_words) & set(transcript_words))
            if overlap > 0:
                keyword_score = min(0.7, overlap / len(query_words))

        # --- Combine both scores for final ranking ---
        final_score = max(semantic_score, keyword_score)

        if final_score >= threshold:
            # --- Timestamp matching ---
            matched_timestamp = None
            if "word_timestamp" in item and item["word_timestamp"]:
                for w in item["word_timestamp"]:
                    word_clean = w["word"].lower().strip(string.punctuation + " ")
                    if word_clean in query_for_search.lower():
                        matched_timestamp = w.get("start")
                        break

            item_copy = dict(item)
            item_copy["match_score"] = round(final_score, 3)
            item_copy["matched"] = query_for_search
            item_copy["timestamp"] = (
                round(float(matched_timestamp), 2) if matched_timestamp else None
            )
            results.append(item_copy)

    # --- Combine emotion matches + semantic matches ---
    all_results = emotion_matched_items + results
    all_results.sort(key=lambda x: x.get("match_score", 0.0), reverse=True)
    return list(all_results), str(query_for_search)
