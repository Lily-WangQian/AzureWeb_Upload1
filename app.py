from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import pdfplumber
import pandas as pd

app = Flask(__name__)

# ---------------- Config ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Standards list ----------------
# Your repo already uses a "standards keywords.csv" file.
# Its standard column is sometimes named "Standards" (you used this previously). :contentReference[oaicite:0]{index=0}
standard_keywords_df = pd.read_csv("standards keywords.csv")
standard_keywords_df.columns = standard_keywords_df.columns.str.strip()
if "Standards" in standard_keywords_df.columns and "Standard" not in standard_keywords_df.columns:
    standard_keywords_df.rename(columns={"Standards": "Standard"}, inplace=True)

standards = (
    standard_keywords_df["Standard"]
    .dropna()
    .astype(str)
    .str.strip()
    .sort_values()
    .unique()
    .tolist()
)

# ---------------- Helpers ----------------
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path: str, max_chars: int = 2000) -> str:
    """Quick text extraction + light cleaning for preview."""
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            chunks.append(t)
            if sum(map(len, chunks)) >= max_chars:
                break
    text = re.sub(r"\s+", " ", "\n".join(chunks)).strip()
    return text[:max_chars] if text else "(no text extracted)"

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        standards=standards,
        selected=None,
        result=None,
        error=None,
        message="Deployed via Azure!"
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf = request.files.get("bank_pdf")

    # Validate inputs
    if not std:
        return render_template("index.html", standards=standards, selected=None, result=None,
                               error="Please select a standard.", message=None)
    if not pdf or pdf.filename == "":
        return render_template("index.html", standards=standards, selected=std, result=None,
                               error="Please upload a bank ESG report (PDF).", message=None)
    if not allowed_file(pdf.filename):
        return render_template("index.html", standards=standards, selected=std, result=None,
                               error="The uploaded file should be a PDF.", message=None)

    # Save & parse
    fname = secure_filename(pdf.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    pdf.save(fpath)

    preview = extract_text_from_pdf(fpath, max_chars=2000)

    result = {
        "filename": fname,
        "standard": std,
        "preview": preview,
    }

    return render_template("index.html", standards=standards, selected=std,
                           result=result, error=None, message=None)

if __name__ == "__main__":
    # Azure sets WEBSITES_PORT; we still bind to 8000 locally.
    app.run(host="0.0.0.0", port=8000)
