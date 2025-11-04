import os
import re
import ast
import gc
import string
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- App & paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ---------- Load Standards CSV (robust) ----------
CSV_PATH = os.path.join(DATA_DIR, "standards_keywords.csv")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # trim + lowercase headers
    df = df.rename(columns={c: re.sub(r"\s+", " ", str(c)).strip() for c in df.columns})
    df = df.rename(columns={c: c.lower() for c in df.columns})

    alias = {
        "standard": "standard",
        "name": "standard",
        "publication date": "publication_date",
        "publication_date": "publication_date",
        "pub date": "publication_date",
        "tfidf keywords": "tfidf_keywords",
        "tf-idf keywords": "tfidf_keywords",
        "tfidf": "tfidf_keywords",
        "contextual keywords": "contextual_keywords",
        "context keywords": "contextual_keywords",
        "contextual": "contextual_keywords",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})
    for need in ["standard", "publication_date", "tfidf_keywords", "contextual_keywords"]:
        if need not in df.columns:
            df[need] = pd.NA
    for c in ["standard", "publication_date", "tfidf_keywords", "contextual_keywords"]:
        df[c] = df[c].astype(str).map(lambda x: x.strip() if x is not None else x)
    return df

def load_standards_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(
            {
                "standard": ["Example Standard"],
                "publication_date": ["2020-Jan"],
                "tfidf_keywords": ["['example standard', 'keyword list']"],
                "contextual_keywords": ["['context word 1', 'context word 2']"],
            }
        )
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python", dtype=str, on_bad_lines="skip")
    return _normalize_columns(df)

standards_df = load_standards_df(CSV_PATH)
standards_list = sorted(
    [s for s in standards_df["standard"].dropna().unique().tolist() if s and s.lower() != "nan"]
)

# ---------- Stopwords & cleaning ----------
CUSTOM_STOPWORDS = set([
    'shall','among','best','would','like','see','needs','•','their','to','“should”','‘should’',
    'requires','“shall”','within','may','lot','etc','b','with','without','pdfs','shows','tells',
    'e','g','also','always','however','go','–','by','for','that','and','or','0c','meet','includes',
    'could','example','examples','chapter','an','a','on','in','as','box','additionally','particularly',
    'thereafter','please','the','The','there','has','to','have','this','welcome','website','appendix','‘can’',
    'we','re',"we’re",'we’re','we','re','should','be','com','rbc','at','from','ceo','appendices',
    'endnotes','volunteerismappendices','is','ii','of','our'
])

def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    sentence = text.translate(str.maketrans({p: " " for p in string.punctuation}))
    sentence = re.sub(r"\s+", " ", sentence)
    kept = []
    for w in sentence.split():
        lw = w.lower()
        if not lw.isdigit() and lw not in CUSTOM_STOPWORDS:
            kept.append(lw)
    return " ".join(kept)

# ---------- TF-IDF bigram keywords ----------
def extract_tfidf_keywords(text: str, top_n: int = 5) -> List[str]:
    clean = remove_stopwords(text)
    if not clean.strip():
        return []
    vec = TfidfVectorizer(ngram_range=(2,2), min_df=1)
    X = vec.fit_transform([clean])
    vocab = vec.get_feature_names_out()
    scores = X.toarray().ravel()
    rank = scores.argsort()[::-1][:top_n]
    return [vocab[i] for i in rank if i < len(vocab)]

# ---------- Contextual keywords (KeyBERT + gte-large-en) with fallback ----------
_CONTEXT_READY = False
_kw_model = None
_keybert = None

def _maybe_load_contextual():
    global _CONTEXT_READY, _kw_model, _keybert
    if _CONTEXT_READY:
        return
    try:
        from sentence_transformers import SentenceTransformer
        from keybert import KeyBERT
        _kw_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        _keybert = KeyBERT(_kw_model)
        _CONTEXT_READY = True
    except Exception:
        _CONTEXT_READY = False

def extract_contextual_keywords(text: str, top_n: int = 5) -> List[str]:
    _maybe_load_contextual()
    if _CONTEXT_READY and text:
        try:
            pairs = _keybert.extract_keywords(
                text, keyphrase_ngram_range=(2,2), top_n=top_n
            )
            return [p[0] for p in pairs]
        except Exception:
            pass
    # Fallback (YAKE-like surrogate using TF-IDF on original text)
    if not text:
        return []
    vec = TfidfVectorizer(ngram_range=(2,2), min_df=1)  # use original text (with stopwords for context)
    X = vec.fit_transform([text])
    vocab = vec.get_feature_names_out()
    scores = X.toarray().ravel()
    rank = scores.argsort()[::-1][:top_n]
    return [vocab[i] for i in rank if i < len(vocab)]

# ---------- Combine keywords ----------
def combine_keywords(contextual: List[str], tfidf: List[str]) -> List[str]:
    seen = set()
    combined = []
    for src in (contextual, tfidf):
        for kw in src:
            if kw not in seen:
                combined.append(kw)
                seen.add(kw)
    return combined

# ---------- PDF helpers ----------
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def read_pdf_text(path: str, maxS_chars: int = 60000) -> str:
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt:
                chunks.append(txt)
            if sum(len(c) for c in chunks) > max_chars:
                break
        return "\n".join(chunks)
    except Exception as e:
        return f"(Failed to read PDF: {e})"

_MONTHS = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
def detect_publication_date(text: str) -> str:
    # heuristic: Month YYYY or YYYY
    m = re.search(fr"{_MONTHS}\s+20\d{{2}}", text, flags=re.IGNORECASE)
    if m:
        return m.group(0)
    y = re.search(r"\b20\d{2}\b", text)
    if y:
        return y.group(0)
    return ""

# ---------- Summarization (T5-base) with graceful fallback ----------
_SUMMARIZER = None
def _maybe_load_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is not None:
        return
    try:
        from transformers import pipeline
        _SUMMARIZER = pipeline("summarization", model="t5-base")
    except Exception:
        _SUMMARIZER = False  # explicitly mark failure

def summarize_text(text: str) -> str:
    _maybe_load_summarizer()
    if not text:
        return ""
    if _SUMMARIZER:
        try:
            # T5 has 512 tokens limit; keep short input and ask a medium summary
            trimmed = text[:2000]
            out = _SUMMARIZER(trimmed, max_length=220, min_length=80, do_sample=False)
            return out[0]["summary_text"]
        except Exception:
            pass
    # fallback: first paragraph-ish
    para = re.split(r"\n\s*\n", text.strip())
    return para[0][:600]

# ---------- Standards helpers ----------
def _maybe_list(x: str) -> List[str]:
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        val = ast.literal_eval(x.strip())
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [seg.strip() for seg in x.split(",") if seg.strip()]

def lookup_standard(std_name: str) -> Dict[str, Any]:
    row = standards_df.loc[(standards_df["standard"].fillna("").str.strip() == std_name)]
    if row.empty:
        return {"name": std_name, "publication_date": "", "tfidf_keywords": [], "contextual_keywords": []}
    r = row.iloc[0]
    return {
        "name": r.get("standard", std_name),
        "publication_date": r.get("publication_date", ""),
        "tfidf_keywords": _maybe_list(r.get("tfidf_keywords", "")),
        "contextual_keywords": _maybe_list(r.get("contextual_keywords", "")),
    }

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        standards=standards_list,
        selected=None,
        bank=None,
        std_info=None,
        error=None,
        message=None,
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template("index.html", standards=standards_list, selected=None, bank=None, std_info=None, error="Please select a standard.", message=None)

    if (not pdf_file) or not pdf_file.filename:
        return render_template("index.html", standards=standards_list, selected=std, bank=None, std_info=None, error="Please upload a bank ESG report (PDF).", message=None)

    if not allowed_file(pdf_file.filename):
        return render_template("index.html", standards=standards_list, selected=std, bank=None, std_info=None, error="The uploaded file should be a PDF.", message=None)

    # Save file
    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    pdf_file.save(fpath)

    # Read PDF & process
    raw_text = read_pdf_text(fpath, max_chars=60000)
    preview = (raw_text[:2500] + " ...") if len(raw_text) > 2500 else raw_text
    pub_date = detect_publication_date(raw_text)

    tfidf_top5 = extract_tfidf_keywords(raw_text, top_n=5)
    ctx_top5 = extract_contextual_keywords(raw_text, top_n=5)
    combined = combine_keywords(ctx_top5, tfidf_top5)

    summary = summarize_text(raw_text)

    bank = {
        "filename": fname,
        "publication_date": pub_date,
        "tfidf_keywords": tfidf_top5,
        "contextual_keywords": ctx_top5,
        "combined_keywords": combined,
        "preview": preview,
        "summary": summary,
        "standard": std,
    }

    std_info = lookup_standard(std)

    # clean up RAM a bit when running locally
    gc.collect()

    return render_template(
        "index.html",
        standards=standards_list,
        selected=std,
        bank=bank,
        std_info=std_info,
        error=None,
        message=None,
    )

if __name__ == "__main__":
    # Azure assigns a PORT env var; fall back to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    # For local dev this is fine; in Azure we’ll run via gunicorn (see STARTUP_COMMAND)
    app.run(host="0.0.0.0", port=port, debug=False)
