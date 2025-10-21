Author: Kiran Varma Indukuri
Affiliation: PES University, Bangalore

1. Overview

This project implements a transformer-based text summarization pipeline for Indian legal bills, leveraging both fine-tuned and hybrid models.
It uses BART and FLAN-T5 architectures for extractive and abstractive summarization respectively, with a hybrid pipeline that combines their strengths for improved factuality and readability.

This repository is structured to let you:

Preprocess and clean raw bill PDFs and summaries

Train transformer models on your dataset

Run inference using fine-tuned or hybrid models

Evaluate performance using ROUGE, BLEU, and BERTScore

2. Features

✅ Preprocessing of PDF bills & summary text files
✅ Fine-tuning BART and T5 on custom JSONL datasets
✅ Hybrid summarization combining extractive (BART) and abstractive (T5) models
✅ Evaluation with ROUGE, BLEU, and BERTScore
✅ Modular design (training / inference / evaluation folders)

Folder Structure
legal-bill-summarization/
│
├── README.md
├── requirements.txt
├── dataset_formating.py
│
├── training/
│   ├── train_model_bart.py
│   ├── train_model_t5.py
│
├── summarization/
│   ├── bart_finetuned.py
│   ├── t5_finetuned.py
│   ├── hybrid_finetuned.py
│   ├── BART_baseline.py
│   ├── T5_baseline.py
│
├── evaluation/
│   └── val_rouge.py
│
├── sample_data/         # update this directory in the code 
│   ├── bills/           # sample PDFs
│   ├── summaries/       # reference summaries
│   └── dataset.jsonl
│
├── .gitignore
└── LICENSE

3. Setup
1️⃣ Clone the repository
git clone https://github.com/kiranv739/legal-bill-summarization.git
cd legal-bill-summarization

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Prepare your dataset

Place your bill PDFs in sample_data/bills/
and reference summaries in text file's in sample_data/summaries/, then run:

python dataset_formating.py


This generates a clean dataset.jsonl file with "text" and "summary" fields.

4. Training
▶️ Fine-tune BART
python training/train_model_bart.py

▶️ Fine-tune FLAN-T5
python training/train_model_t5.py

Fine-tuned models will be saved in:
/bart_model/
/flant5_model/

5. Summarization (Inference)

Use one of the fine-tuned models or the hybrid model:

python summarization/bart_finetuned.py
python summarization/t5_finetuned.py
python summarization/hybrid_finetuned.py


Each script reads input text (bill) and outputs generated summaries.

6. Evaluation
You can compute ROUGE, BLEU, and BERTScore with:
python evaluation/val_rouge.py

