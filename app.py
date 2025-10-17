import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pdfplumber

ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")  # needed for flash()
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Single required standard per your request
STANDARDS = ["AzureWeb_5"]

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(path: str) -> str:
    """Read text from all pages of a PDF; skip pages that have no extractable text."""
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
    return "\n\n".join(parts).strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", standards=STANDARDS, result=None)

@app.route("/analyze", methods=["POST"])
def analyze():
    std = request.form.get("standard", "").strip()
    file = request.files.get("bank_pdf")

    if not std:
        flash("Please choose a standard.")
        return redirect(url_for("home"))
    if std not in STANDARDS:
        flash("Invalid standard selected.")
        return redirect(url_for("home"))
    if not file or file.filename == "":
        flash("Please upload a PDF file.")
        return redirect(url_for("home"))
    if not allowed_file(file.filename):
        flash("Only .pdf files are allowed.")
        return redirect(url_for("home"))

    # Save upload
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Extract text
    try:
        text = extract_pdf_text(path)
        if not text:
            text = "(No extractable text found in the PDF.)"
    except Exception as e:
        text = f"(Error reading PDF: {e})"

    # Show up to ~10,000 characters to keep page responsive
    preview = text[:10000] + ("..." if len(text) > 10000 else "")

    return render_template(
        "index.html",
        standards=STANDARDS,
        result={
            "filename": filename,
            "standard": std,
            "text": preview
        }
    )

if __name__ == "__main__":
    # Local run: python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
