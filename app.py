import re
from pathlib import Path
from typing import Dict, Tuple

import pdfplumber
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
)


st.set_page_config(page_title="Legal Bill Summarizer", page_icon="⚖️", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, Tuple[object, object]] = {}


def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_bart(model_path: str) -> Tuple[BartTokenizer, BartForConditionalGeneration]:
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model


def load_t5(model_path: str) -> Tuple[AutoTokenizer, T5ForConditionalGeneration]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model


def get_model(model_key: str):
    if model_key in MODELS:
        return MODELS[model_key]

    if model_key == "BART Baseline":
        MODELS[model_key] = load_bart("facebook/bart-large-cnn")
    elif model_key == "BART Finetuned":
        MODELS[model_key] = load_bart("./bart_model")
    elif model_key == "FLAN-T5 Baseline":
        MODELS[model_key] = load_t5("google/flan-t5-large")
    elif model_key == "FLAN-T5 Finetuned":
        MODELS[model_key] = load_t5("./flant5_model")
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    return MODELS[model_key]


def split_into_chunks(tokenizer, text: str, max_tokens: int):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    return [input_ids[i : i + max_tokens] for i in range(0, len(input_ids), max_tokens)]


def summarize_with_bart(tokenizer, model, chunk_tensor) -> str:
    inputs = {
        "input_ids": chunk_tensor.unsqueeze(0).to(DEVICE),
        "attention_mask": torch.ones_like(chunk_tensor.unsqueeze(0)).to(DEVICE),
    }
    summary_ids = model.generate(
        **inputs,
        max_length=350,
        num_beams=3,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_with_t5(tokenizer, model, chunk_tensor_or_text, use_prompt: bool) -> str:
    if isinstance(chunk_tensor_or_text, str):
        input_text = chunk_tensor_or_text
    else:
        input_text = tokenizer.decode(chunk_tensor_or_text, skip_special_tokens=True)

    if use_prompt:
        input_text = "summarize: " + input_text.strip()

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    summary_ids = model.generate(
        **encoded,
        max_length=350,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def run_pipeline(text: str, model_choice: str) -> str:
    if model_choice in ("BART Baseline", "BART Finetuned"):
        tokenizer, model = get_model(model_choice)
        chunks = split_into_chunks(tokenizer, text, max_tokens=1024)
        summaries = []
        for chunk in chunks:
            if len(chunk) < 100:
                summaries.append(tokenizer.decode(chunk, skip_special_tokens=True))
                continue
            summaries.append(summarize_with_bart(tokenizer, model, chunk))
        return "\n\n".join(summaries)

    if model_choice in ("FLAN-T5 Baseline", "FLAN-T5 Finetuned"):
        tokenizer, model = get_model(model_choice)
        max_tokens = 512 if model_choice == "FLAN-T5 Baseline" else 1024
        chunks = split_into_chunks(tokenizer, text, max_tokens=max_tokens)
        summaries = []
        for chunk in chunks:
            if len(chunk) < 100:
                summaries.append(tokenizer.decode(chunk, skip_special_tokens=True))
                continue
            use_prompt = model_choice == "FLAN-T5 Baseline"
            summaries.append(summarize_with_t5(tokenizer, model, chunk, use_prompt=use_prompt))
        return "\n\n".join(summaries)

    if model_choice == "Hybrid (BART Finetuned -> FLAN-T5 Finetuned)":
        bart_tokenizer, bart_model = get_model("BART Finetuned")
        t5_tokenizer, t5_model = get_model("FLAN-T5 Finetuned")
        chunks = split_into_chunks(bart_tokenizer, text, max_tokens=1024)
        summaries = []
        for chunk in chunks:
            if len(chunk) < 100:
                summaries.append(bart_tokenizer.decode(chunk, skip_special_tokens=True))
                continue
            bart_summary = summarize_with_bart(bart_tokenizer, bart_model, chunk)
            summaries.append(summarize_with_t5(t5_tokenizer, t5_model, bart_summary, use_prompt=False))
        return "\n\n".join(summaries)

    raise ValueError(f"Unknown model choice: {model_choice}")


def main():
    st.title("Legal Bill Summarization GUI")
    st.caption("Upload an Indian legal bill PDF and generate a summary with baseline, finetuned, or hybrid models.")

    with st.sidebar:
        st.subheader("Configuration")
        model_choice = st.selectbox(
            "Choose summarization mode",
            [
                "BART Baseline",
                "FLAN-T5 Baseline",
                "BART Finetuned",
                "FLAN-T5 Finetuned",
                "Hybrid (BART Finetuned -> FLAN-T5 Finetuned)",
            ],
        )
        st.markdown(f"**Device:** `{DEVICE}`")
        if not Path("bart_model").exists() or not Path("flant5_model").exists():
            st.warning("Finetuned model folders were not found. Baseline modes will still work.")

    uploaded_pdf = st.file_uploader("Upload bill PDF", type=["pdf"])

    if uploaded_pdf is None:
        st.info("Upload a PDF to begin summarization.")
        return

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Extracting PDF text..."):
            text = extract_text_from_pdf(uploaded_pdf)

        if not text:
            st.error("No extractable text was found in this PDF.")
            return

        with st.expander("Extracted Text Preview", expanded=False):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

        with st.spinner("Running summarization..."):
            try:
                summary = run_pipeline(text, model_choice)
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")
                return

        st.subheader("Generated Summary")
        st.text_area("Summary Output", value=summary, height=350)

        output_name = Path(uploaded_pdf.name).stem + "_summary.txt"
        st.download_button(
            "Download Summary",
            data=summary,
            file_name=output_name,
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
