# data_loader.py
# Author: [Your Name] | Index: [Your Index Number]

import os
import re
import requests
import pandas as pd
import pdfplumber

# Dataset URLs
CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")
PDF_PATH = os.path.join(DATA_DIR, "2025_Budget.pdf")


def _download(url, dest):
    """Download a file only if it doesn't already exist."""
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"Saved: {dest}")


def _clean_text(text):
    """Remove extra spaces and non-readable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return text.strip()


def load_csv():
    """Download and clean the Election CSV. Returns list of row dicts."""
    _download(CSV_URL, CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop empty rows
    df.dropna(how="all", inplace=True)

    # Strip whitespace from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    records = df.to_dict(orient="records")
    print(f"CSV loaded: {len(records)} rows")
    return records


def csv_records_to_text(records):
    """Convert each CSV row into a readable sentence."""
    texts = []
    for r in records:
        parts = [
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in r.items()
            if str(v).lower() not in ("nan", "none", "")
        ]
        sentence = " | ".join(parts)
        texts.append(_clean_text(sentence))
    return texts


def load_pdf():
    """Download and extract text from the Budget PDF. Returns list of page dicts."""
    _download(PDF_URL, PDF_PATH)

    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if len(cleaned) >= 50:  # skip blank/image pages
                pages.append({"page_num": i + 1, "text": cleaned})

    print(f"PDF loaded: {len(pages)} usable pages")
    return pages


def pdf_pages_to_text(pages):
    """Return just the text from each page dict."""
    return [p["text"] for p in pages]