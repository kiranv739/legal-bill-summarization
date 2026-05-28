import re
from io import BytesIO
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

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BART_FINETUNED_DIR = PROJECT_ROOT / "bart_model"
DEFAULT_T5_FINETUNED_DIR = PROJECT_ROOT / "flant5_model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, Tuple[object, object]] = {}


def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    file_bytes = uploaded_file.getvalue()
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        total_pages = len(pdf.pages)
        progress = st.progress(0, text=f"Reading PDF pages: 0/{total_pages}")
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            progress.progress(
                int((idx / total_pages) * 100),
                text=f"Reading PDF pages: {idx}/{total_pages}",
            )
        progress.empty()
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


@st.cache_resource(show_spinner=False)
def get_model(model_key: str, bart_path: str, t5_path: str):
    if model_key in MODELS:
        return MODELS[model_key]

    if model_key == "BART Baseline":
        MODELS[model_key] = load_bart("facebook/bart-large-cnn")
    elif model_key == "BART Finetuned":
        bart_dir = Path(bart_path)
        if not bart_dir.exists():
            raise FileNotFoundError(f"Finetuned BART folder not found: {bart_dir}")
        MODELS[model_key] = load_bart(str(bart_dir))
    elif model_key == "FLAN-T5 Baseline":
        MODELS[model_key] = load_t5("google/flan-t5-large")
    elif model_key == "FLAN-T5 Finetuned":
        t5_dir = Path(t5_path)
        if not t5_dir.exists():
            raise FileNotFoundError(f"Finetuned FLAN-T5 folder not found: {t5_dir}")
        MODELS[model_key] = load_t5(str(t5_dir))
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    return MODELS[model_key]


def split_into_chunks(tokenizer, text: str, max_tokens: int):
    # Avoid tokenizing the entire document at once (can trigger max-length warnings/errors).
    words = text.split()
    chunks = []
    word_window = 1200

    for i in range(0, len(words), word_window):
        piece = " ".join(words[i : i + word_window])
        tokenized = tokenizer(
            piece,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_overflowing_tokens=True,
        )
        input_ids = tokenized["input_ids"]
        # HF can return either List[int] (single chunk) or List[List[int]] (overflow chunks).
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]

        for token_chunk in input_ids:
            if token_chunk:
                chunks.append(torch.tensor(token_chunk, dtype=torch.long))

    return chunks


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


def run_pipeline(text: str, model_choice: str, bart_path: str, t5_path: str) -> str:
    if model_choice in ("BART Baseline", "BART Finetuned"):
        tokenizer, model = get_model(model_choice, bart_path, t5_path)
        chunks = split_into_chunks(tokenizer, text, max_tokens=1024)
        summaries = []
        progress = st.progress(0, text=f"Summarizing chunks: 0/{len(chunks)}")
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) < 100:
                summaries.append(tokenizer.decode(chunk, skip_special_tokens=True))
                progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
                continue
            summaries.append(summarize_with_bart(tokenizer, model, chunk))
            progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
        progress.empty()
        return "\n\n".join(summaries)

    if model_choice in ("FLAN-T5 Baseline", "FLAN-T5 Finetuned"):
        tokenizer, model = get_model(model_choice, bart_path, t5_path)
        max_tokens = 512 if model_choice == "FLAN-T5 Baseline" else 1024
        chunks = split_into_chunks(tokenizer, text, max_tokens=max_tokens)
        summaries = []
        progress = st.progress(0, text=f"Summarizing chunks: 0/{len(chunks)}")
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) < 100:
                summaries.append(tokenizer.decode(chunk, skip_special_tokens=True))
                progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
                continue
            use_prompt = model_choice == "FLAN-T5 Baseline"
            summaries.append(summarize_with_t5(tokenizer, model, chunk, use_prompt=use_prompt))
            progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
        progress.empty()
        return "\n\n".join(summaries)

    if model_choice == "Hybrid (BART Finetuned -> FLAN-T5 Finetuned)":
        bart_tokenizer, bart_model = get_model("BART Finetuned", bart_path, t5_path)
        t5_tokenizer, t5_model = get_model("FLAN-T5 Finetuned", bart_path, t5_path)
        chunks = split_into_chunks(bart_tokenizer, text, max_tokens=1024)
        summaries = []
        progress = st.progress(0, text=f"Summarizing chunks: 0/{len(chunks)}")
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) < 100:
                summaries.append(bart_tokenizer.decode(chunk, skip_special_tokens=True))
                progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
                continue
            bart_summary = summarize_with_bart(bart_tokenizer, bart_model, chunk)
            summaries.append(summarize_with_t5(t5_tokenizer, t5_model, bart_summary, use_prompt=False))
            progress.progress(int((idx / len(chunks)) * 100), text=f"Summarizing chunks: {idx}/{len(chunks)}")
        progress.empty()
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
        if torch.cuda.is_available():
            st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
        else:
            st.warning("CUDA GPU not detected by PyTorch. App is running on CPU.")

        bart_model_dir = st.text_input(
            "BART finetuned model folder",
            value=str(DEFAULT_BART_FINETUNED_DIR),
        ).strip()
        t5_model_dir = st.text_input(
            "FLAN-T5 finetuned model folder",
            value=str(DEFAULT_T5_FINETUNED_DIR),
        ).strip()

        st.markdown(f"**BART finetuned path:** `{bart_model_dir}`")
        st.markdown(f"**FLAN-T5 finetuned path:** `{t5_model_dir}`")
        if st.button("Preload selected model"):
            with st.spinner("Loading model into memory..."):
                _ = (
                    get_model(model_choice, bart_model_dir, t5_model_dir)
                    if model_choice != "Hybrid (BART Finetuned -> FLAN-T5 Finetuned)"
                    else (
                        get_model("BART Finetuned", bart_model_dir, t5_model_dir),
                        get_model("FLAN-T5 Finetuned", bart_model_dir, t5_model_dir),
                    )
                )
            st.success("Model preloaded.")

        if not Path(bart_model_dir).exists() or not Path(t5_model_dir).exists():
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
                summary = run_pipeline(text, model_choice, bart_model_dir, t5_model_dir)
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")
                return

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
            st.caption(f"GPU memory currently allocated: {mem_gb:.2f} GB")

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
