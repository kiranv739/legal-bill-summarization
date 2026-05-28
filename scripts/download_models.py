import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def main() -> None:
    bart_url = os.getenv("BART_MODEL_URL", "").strip()
    t5_url = os.getenv("T5_MODEL_URL", "").strip()

    if not bart_url and not t5_url:
        print("No URLs provided.")
        print("Set BART_MODEL_URL and/or T5_MODEL_URL environment variables.")
        print("Example (PowerShell):")
        print('$env:BART_MODEL_URL="https://your-link/bart_model.zip"')
        print('$env:T5_MODEL_URL="https://your-link/flant5_model.zip"')
        print("python scripts/download_models.py")
        return

    downloads_dir = PROJECT_ROOT / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    targets = [
        (bart_url, "bart_model.zip", PROJECT_ROOT / "bart_model"),
        (t5_url, "flant5_model.zip", PROJECT_ROOT / "flant5_model"),
    ]

    for url, archive_name, output_dir in targets:
        if not url:
            continue

        archive_path = downloads_dir / archive_name
        print(f"Downloading: {url}")
        download_file(url, archive_path)
        print(f"Extracting to: {output_dir}")
        extract_zip(archive_path, output_dir)
        print("Done.\n")

    print("Model download/extract workflow completed.")


if __name__ == "__main__":
    main()
