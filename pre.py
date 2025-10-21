import os
import json
import re
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download("punkt")

# --- Paths ---
bills_dir = "bills"
summaries_dir = "summaries"
output_file = "dataset.jsonl"

# --- Get number of bill-summary pairs dynamically ---
bill_files = sorted([f for f in os.listdir(bills_dir) if f.endswith(".pdf")])
summary_files = sorted([f for f in os.listdir(summaries_dir) if f.endswith(".txt")])
num_pairs = min(len(bill_files), len(summary_files))

# --- Utility Functions ---

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_summary(text):
    disclaimer_keywords = ["DISCLAIMER:", "PRS Legislative Research"]
    for keyword in disclaimer_keywords:
        if keyword in text:
            text = text[:text.index(keyword)]
    return text.strip()

def remove_boilerplate_and_clean(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"[\n\r]{2,}", "\n", text)
    text = re.sub(r"[\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def preprocess_text(text):
   return text.strip()


# --- Main Preprocessing Loop ---

dataset = []

for i in tqdm(range(1, num_pairs + 1), desc="Processing pairs"):
    bill_path = os.path.join(bills_dir, f"{i}.pdf")
    summary_path = os.path.join(summaries_dir, f"{i}.txt")

    if not os.path.exists(bill_path) or not os.path.exists(summary_path):
        print(f"⚠️ Skipping pair {i:02d}: file not found")
        continue

    bill_raw = extract_text_from_pdf(bill_path)
    summary_raw = extract_text_from_txt(summary_path)

    bill_cleaned = remove_boilerplate_and_clean(bill_raw)
    summary_cleaned = clean_summary(summary_raw)

    bill_final = preprocess_text(bill_cleaned)
    summary_final = summary_cleaned

    dataset.append({
        "text": bill_final,
        "summary": summary_final
    })

# --- Save to JSONL ---
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        json.dump(item, f)
        f.write("\n")

print(f"\n✅ Preprocessing complete! {len(dataset)} clean pairs saved to: {output_file}")
