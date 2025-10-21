from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path
import pdfplumber
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load models ===
bart_model_path = "./bart_model"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path).to(device)
bart_model.eval()

t5_model_path = "./flant5_model"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(device)
t5_model.eval()

# === Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    text = re.sub(r"\s+", " ", text)
    return text

# === Chunking ===
def split_into_chunks(text, max_tokens=1024):
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]
    return chunks

# === BART Extractive Summary ===
def bart_summarize_chunk(chunk_tensor):
    try:
        inputs = {
            "input_ids": chunk_tensor.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(chunk_tensor.unsqueeze(0)).to(device),
        }
        summary_ids = bart_model.generate(
            **inputs,
            max_length=350,
            num_beams=3,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print("⚠️ BART Error:", e)
        return "[BART GENERATION FAILED]"

# === T5 Abstractive Summary ===
def t5_summarize_text(text):
    try:
        inputs = t5_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        summary_ids = t5_model.generate(
            **inputs,
            max_length=350,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print("⚠️ T5 Error:", e)
        return "[T5 GENERATION FAILED]"

# === Process a single bill ===
def process_bill(pdf_path):
    print(f"\n📄 Processing: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    print(f"📄 Extracted total characters: {len(text)}")

    chunks = split_into_chunks(text)
    print(f"🧩 Total Chunks: {len(chunks)}")

    final_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1}/{len(chunks)} ---")
        print(f"📏 Tokens in chunk: {len(chunk)}")

        if len(chunk) < 100:
            chunk_text = bart_tokenizer.decode(chunk, skip_special_tokens=True)
            print("⚠️ Chunk too short — copying original text.")
            final_summaries.append(chunk_text)
            continue

        # Step 1: BART extractive summary
        bart_summary = bart_summarize_chunk(chunk)
        print(f"🧱 Extractive (BART): {bart_summary[:300]}...")

        # Step 2: T5 abstractive summary
        t5_summary = t5_summarize_text(bart_summary)
        print(f"✨ Abstractive (T5): {t5_summary[:300]}...\n")

        final_summaries.append(t5_summary)

    full_summary = "\n\n".join(final_summaries)
    out_path = Path("summaries_final_finetuned") / f"{pdf_path.stem}.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(full_summary, encoding="utf-8")
    print(f"💾 Saved summary to: {out_path}")

# === Process all PDFs ===
def summarize_all_bills(folder_path="./bills"):
    pdf_paths = list(Path(folder_path).glob("*.pdf"))
    print(f"📁 Found {len(pdf_paths)} PDF(s) in {folder_path}")

    for pdf_path in pdf_paths:
        try:
            process_bill(pdf_path)
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name}: {e}")

# === Run ===
summarize_all_bills()
