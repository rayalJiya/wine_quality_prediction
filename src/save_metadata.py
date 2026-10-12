import json

from data_utils import FEATURES, PROJECT_ROOT, get_train_test_split

META_PATH = PROJECT_ROOT / "models" / "feature_meta.json"


def main():
    X_train, _, _, _ = get_train_test_split()  # training data only
    meta = {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "median": float(X_train[col].median()),
        }
        for col in FEATURES
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {META_PATH}")
    for col, m in meta.items():
        print(f"{col:22s} min={m['min']:<8.4g} median={m['median']:<8.4g} max={m['max']:<8.4g}")


if __name__ == "__main__":
    main()