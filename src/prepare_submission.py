import zipfile

from data_utils import PROJECT_ROOT

ZIP_PATH = PROJECT_ROOT.parent / "wine-quality-prediction_submission.zip"
EXCLUDE_DIRS = {"venv", "__pycache__", ".git", ".vscode", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc"}

REQUIRED = [
    "data/WineQT.csv", "data/WineQT_clean.csv", "requirements.txt", "README.md",
    "app/streamlit_app.py", "src/predictor.py", "src/data_utils.py",
    "models/tuned_best_model.joblib", "models/feature_meta.json",
    "reports/final_report.md", "reports/model_comparison.csv",
]


def main():
    missing = [r for r in REQUIRED if not (PROJECT_ROOT / r).exists()]
    if missing:
        print("Missing files, fix before zipping:")
        for m in missing:
            print(f"  - {m}")
        return

    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PROJECT_ROOT.rglob("*")):
            rel = p.relative_to(PROJECT_ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if p.is_file() and p.suffix not in EXCLUDE_SUFFIXES:
                zf.write(p, f"wine-quality-prediction/{rel.as_posix()}")
                count += 1
    size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"Created: {ZIP_PATH}")
    print(f"Files included: {count} | Size: {size_mb:.2f} MB")
    print("venv folder was excluded on purpose.")


if __name__ == "__main__":
    main()