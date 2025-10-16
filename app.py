import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")  # needed for flash messages
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def is_pdf_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/analyze", methods=["POST"])
def analyze():
    bank_file = request.files.get("bank_pdf")
    std_file  = request.files.get("std_pdf")

    # Validate inputs
    if not bank_file or bank_file.filename.strip() == "":
        flash("Please upload the **Bank ESG PDF**.")
        return redirect(url_for("home"))
    if not std_file or std_file.filename.strip() == "":
        flash("Please upload the **Standard PDF**.")
        return redirect(url_for("home"))

    if not is_pdf_file(bank_file.filename):
        flash("The uploaded Bank file must be a **PDF (.pdf)**.")
        return redirect(url_for("home"))
    if not is_pdf_file(std_file.filename):
        flash("The uploaded Standard file must be a **PDF (.pdf)**.")
        return redirect(url_for("home"))

    # Save files (you can remove saving if you don’t need it yet)
    bank_name = secure_filename(bank_file.filename)
    std_name  = secure_filename(std_file.filename)

    bank_path = os.path.join(app.config["UPLOAD_FOLDER"], bank_name)
    std_path  = os.path.join(app.config["UPLOAD_FOLDER"], std_name)
    bank_file.save(bank_path)
    std_file.save(std_path)

    # For now we only show names; you can later add text extraction & similarity here.
    return render_template(
        "index.html",
        result={"bank_name": bank_name, "std_name": std_name}
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
