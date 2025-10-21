import os
import csv
from tqdm import tqdm
from evaluate import load
from bert_score import score as bert_score

# === Folder Configuration ===
summary_dirs = {
    "BART": "summaries_bart",
    "T5": "summaries_flant5",
    # "BART+T5": "summaries_bart_t5",
    "FLAN-T5-FINETUNED": "summaries_flant5_finetuned",
    "BART-FINETUNED": "summaries_bart_finetuned",
    "FINAL-FINETUNED": "summaries_final_finetuned"
}
reference_dir = "reference"

# === Load Metrics ===
rouge = load("rouge")
bleu = load("bleu")

# === Results Storage ===
results = {}

# === Evaluate Each Model's Summaries ===
for model_name, model_dir in summary_dirs.items():
    print(f"\n🔍 Evaluating summaries from: {model_dir}")
    system_summaries = []
    reference_summaries = []

    # Match file by name
    filenames = sorted(os.listdir(reference_dir))
    for fname in tqdm(filenames):
        ref_path = os.path.join(reference_dir, fname)
        hyp_path = os.path.join(model_dir, fname)

        if not os.path.exists(hyp_path):
            print(f"❌ Missing prediction: {hyp_path}")
            continue

        with open(ref_path, "r", encoding="utf-8") as ref_file:
            reference = ref_file.read().strip()

        with open(hyp_path, "r", encoding="utf-8") as hyp_file:
            hypothesis = hyp_file.read().strip()

        reference_summaries.append(reference)
        system_summaries.append(hypothesis)

    if not system_summaries:
        print(f"⚠️ No summaries found for {model_name}")
        continue

    # === Compute ROUGE ===
    rouge_result = rouge.compute(
        predictions=system_summaries,
        references=reference_summaries,
        use_stemmer=True
    )

    # === Compute BLEU ===
    bleu_result = bleu.compute(
        predictions=system_summaries,
        references=[[ref] for ref in reference_summaries]
    )

    # === Compute BERTScore ===
    P, R, F1 = bert_score(system_summaries, reference_summaries, lang="en", verbose=False)

    # === Save Results ===
    results[model_name] = {
        "ROUGE-1": rouge_result["rouge1"],
        "ROUGE-L": rouge_result["rougeL"],
        "BLEU": bleu_result["bleu"],
        "BERTScore-F1": F1.mean().item()
    }

# === Print Final Report ===
print("\n📊 Summary Evaluation Report")
for model, scores in results.items():
    print(f"\n🧠 {model}")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# === Export to CSV ===
csv_file = "summary_evaluation_metrics.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "ROUGE-1", "ROUGE-L", "BLEU", "BERTScore-F1"])
    for model, metrics in results.items():
        writer.writerow([
            model,
            round(metrics["ROUGE-1"], 4),
            round(metrics["ROUGE-L"], 4),
            round(metrics["BLEU"], 4),
            round(metrics["BERTScore-F1"], 4)
        ])

print(f"\n📁 CSV report saved to: {csv_file}")
