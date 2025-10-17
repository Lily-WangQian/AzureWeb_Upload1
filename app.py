from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import pdfplumber
import pandas as pd

app = Flask(__name__)

# ---- config ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- standards list (from your CSV) ----
# Same data source you already use for standards:
#   'standards keywords.csv' contains a "Standard" (or "Standards") column
standard_keywords_df = pd.read_csv("standards keywords.csv")
standard_keywords_df.columns = standard_keywords_df.columns.str.strip()
if "Standards" in standard_keywords_df.columns and "Standard" not in standard_keywords_df.columns:
    standard_keywords_df.rename(columns={"Standards": "Standard"}, inplace=True)
standards = (
    standard_keywords_df["Standard"].dropna().astype(str).str.strip().sort_values().unique().tolist()
)

# ---- helpers ----
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path: str, max_chars: int = 2000) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
            if sum(len(p) for p in parts) >= max_chars:
                break
    text = "\n".join(parts)
    # quick cleaning
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

# ---- routes ----
@app.route("/")
def home():
    return render_template("index.html", standards=standards, selected=None, result=None, error=None, message="Deployed via Azure!")

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf = request.files.get("bank_pdf")

    if not std:
        return render_template("index.html", standards=standards, selected=None, result=None,
                               error="Please choose a standard.", message=None)

    if not pdf or pdf.filename == "":
        return render_template("index.html", standards=standards, selected=std, result=None,
                               error="Please upload a bank ESG report (PDF).", message=None)

    if not allowed_file(pdf.filename):
        return render_template("index.html", standards=standards, selected=std, result=None,
                               error="The uploaded file should be a PDF.", message=None)

    # save & parse
    fname = secure_filename(pdf.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    pdf.save(fpath)

    preview = extract_text_from_pdf(fpath, max_chars=2000) or "(no text extracted)"
    result = {"filename": fname, "standard": std, "preview": preview}

    return render_template("index.html", standards=standards, selected=std, result=result, error=None, message=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
