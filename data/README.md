# Data and Model Artifacts

This project keeps large artifacts outside GitHub.

## Why
- Full datasets and finetuned model weights are usually too large for regular Git.
- External hosting keeps the repository lightweight and easier to clone.

## Recommended Hosting
- Public artifacts: Hugging Face Hub, Kaggle, Zenodo
- Private artifacts: Google Drive, S3, private cloud bucket

## Expected Local Paths
- Dataset file: `dataset.jsonl` in repository root
- Finetuned BART folder: `bart_model/`
- Finetuned FLAN-T5 folder: `flant5_model/`

## Suggested Dataset Layout (if sharing raw data)
```text
data/
|-- raw/
|   |-- bills/
|   |-- summaries/
|-- processed/
|   |-- dataset.jsonl
```

## Quick Start for Collaborators
1. Download dataset and model artifacts from your published links.
2. Place them into the expected local paths shown above.
3. Run:
   - `python -m streamlit run app.py`
   - or training/inference scripts as needed.

## Notes
- Do not commit private or licensed data unless sharing is permitted.
- Keep only a very small sample in GitHub if you want reproducible demos.
