from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path
import pdfplumber
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model and tokenizer ===
model_path = "./flant5_model"  # Replace with your actual fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# === Helper: Read and clean PDF ===
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    text = re.sub(r"\s+", " ", text)
    return text

# === Helper: Split into chunks ===
def split_into_chunks(text, max_tokens=1024):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]
    return chunks

# === Summarize each chunk ===
def summarize_chunk(chunk_tensor):
    try:
        inputs = {
            "input_ids": chunk_tensor.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(chunk_tensor.unsqueeze(0)).to(device),
        }
        summary_ids = model.generate(
            **inputs,
            max_length=350,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print("⚠️ Error during generation:", e)
        return "[GENERATION FAILED]"

# === Process one bill ===
def process_bill(pdf_path):
    print(f"\n📄 Processing: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    print(f"📄 Extracted total characters: {len(text)}")

    chunks = split_into_chunks(text)
    print(f"🧩 Total Chunks: {len(chunks)}")

    summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1}/{len(chunks)} ---")
        print(f"📏 Tokens in chunk: {len(chunk)}")

        if len(chunk) < 100:
            print("⚠️ Chunk too short — skipping generation, copying original text.")
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            print(f"📝 Copied Chunk: {chunk_text[:300]}...\n")
            summaries.append(chunk_text)
            continue

        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        print(f"📝 Chunk preview: {chunk_text[:300]}...")
        summary = summarize_chunk(chunk)
        print(f"✅ Summary: {summary[:300]}...\n")
        summaries.append(summary)

    full_summary = "\n\n".join(summaries)
    out_path = Path("summaries_flant5_finetuned") / f"{pdf_path.stem}.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(full_summary, encoding="utf-8")
    print(f"💾 Saved summary to: {out_path}")

# === Process all PDFs in bills folder ===
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
