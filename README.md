# Legal Bill Summarization

Transformer-based summarization pipeline for Indian legal bills using:
- `BART` (extractive-style summarization)
- `FLAN-T5` (abstractive summarization)
- Hybrid pipeline: `BART finetuned -> FLAN-T5 finetuned`

## What This Project Does
- Preprocesses bill PDFs and reference summaries into JSONL training data
- Fine-tunes BART and FLAN-T5 models
- Runs baseline, finetuned, and hybrid summarization
- Evaluates summaries using ROUGE, BLEU, and BERTScore
- Provides a Streamlit GUI for interactive summarization

## Project Structure
```text
legal-bill-summarization/
|-- app.py
|-- dataset_formating.py
|-- train_model_bart.py
|-- train_model_t5.py
|-- BART_baseline.py
|-- T5_baseline.py
|-- bart_finetuned.py
|-- t5_finetuned.py
|-- FINAL_FINETUNED.py
|-- val_rouge.py
|-- requirements.txt
|-- .gitignore
|-- bart_model/          # local finetuned model folder (not pushed)
|-- flant5_model/        # local finetuned model folder (not pushed)
```

## Setup
1. Clone repository
```bash
git clone https://github.com/kiranv739/legal-bill-summarization.git
cd legal-bill-summarization
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Artifact Management (Best Practice)
Keep large files out of GitHub:
- Code and documentation: GitHub
- Models and full datasets: external storage + links in README

See detailed guidance in `data/README.md`.

### Recommended Approach
1. Host finetuned model zip files externally (Drive/S3/Hugging Face/Kaggle).
2. Keep `bart_model/` and `flant5_model/` local (already ignored by `.gitignore`).
3. Optionally host your full dataset externally and keep only a small sample in repo.

### Optional Model Download Helper
You can download model archives via environment-variable URLs:

```powershell
$env:BART_MODEL_URL="https://your-link/bart_model.zip"
$env:T5_MODEL_URL="https://your-link/flant5_model.zip"
python scripts/download_models.py
```

The script extracts to:
- `bart_model/`
- `flant5_model/`

## Dataset Formatting
Put bill PDFs in `bills/` and summary `.txt` files in `summaries/`, then run:
```bash
python dataset_formating.py
```
This creates `dataset.jsonl` with `text` and `summary` fields.

## Training
Fine-tune BART:
```bash
python train_model_bart.py
```

Fine-tune FLAN-T5:
```bash
python train_model_t5.py
```

By default, models are saved locally to:
- `bart_model/`
- `flant5_model/`

## Inference Scripts
Baseline BART:
```bash
python BART_baseline.py
```

Baseline FLAN-T5:
```bash
python T5_baseline.py
```

Finetuned BART:
```bash
python bart_finetuned.py
```

Finetuned FLAN-T5:
```bash
python t5_finetuned.py
```

Hybrid (finetuned BART -> finetuned FLAN-T5):
```bash
python FINAL_FINETUNED.py
```

## GUI (Streamlit)
Run:
```bash
python -m streamlit run app.py
```

### GUI Features
- Upload PDF bill and summarize directly
- Choose:
  - BART Baseline
  - FLAN-T5 Baseline
  - BART Finetuned
  - FLAN-T5 Finetuned
  - Hybrid (BART Finetuned -> FLAN-T5 Finetuned)
- Configure finetuned model folder paths from the sidebar (not hardcoded)
- PDF extraction progress + chunk-level summarization progress
- GPU/device visibility in sidebar
- Download summary as `.txt`

## Evaluation
```bash
python val_rouge.py
```
Outputs ROUGE, BLEU, BERTScore and exports `summary_evaluation_metrics.csv`.

## Important Note on Model Folders
Model folders are intentionally not pushed to GitHub because they are large.
- Keep `bart_model/` and `flant5_model/` local
- Use the GUI sidebar to point to their paths
- Share only code via GitHub

## Recent Changes
- Added `app.py` Streamlit GUI
- Added `streamlit` to dependencies
- Fixed long-sequence tokenization issues for T5/BART chunking
- Fixed finetuned BART runtime error (`len() of a 0-d tensor`)
- Added non-hardcoded finetuned model path inputs in GUI
- Added preload option and better progress feedback
