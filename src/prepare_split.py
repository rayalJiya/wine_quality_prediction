import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_utils import (
    FEATURES, TARGET, RANDOM_STATE, TEST_SIZE,
    load_clean_data, get_features_and_target, get_train_test_split,
)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # TASK 6: feature / target separation
    section("TASK 6: FEATURES AND TARGET")
    df = load_clean_data()
    X, y = get_features_and_target(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
    print(f"Target: {TARGET}")
    print(f"'quality' in X: {TARGET in X.columns}")
    print(f"'Id' in X: {'Id' in X.columns}")

    # TASK 7: stratified split
    section(f"TASK 7: STRATIFIED SPLIT (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    dist = pd.DataFrame({
        "full_count": y.value_counts().sort_index(),
        "train_count": y_train.value_counts().sort_index(),
        "test_count": y_test.value_counts().sort_index(),
        "full_%": (y.value_counts(normalize=True).sort_index() * 100).round(2),
        "train_%": (y_train.value_counts(normalize=True).sort_index() * 100).round(2),
        "test_%": (y_test.value_counts(normalize=True).sort_index() * 100).round(2),
    })
    print("\nClass distribution (stratification check):")
    print(dist.to_string())

    # TASK 8: scaling without leakage (demonstration)
    section("TASK 8: STANDARD SCALING (fit on TRAIN only)")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    print("Feature ranges BEFORE scaling (train):")
    print(X_train.agg(["min", "max"]).T.round(3).to_string())

    check = pd.DataFrame({
        "train_mean": X_train_scaled.mean().round(3),
        "train_std": X_train_scaled.std(ddof=0).round(3),
        "test_mean": X_test_scaled.mean().round(3),
        "test_std": X_test_scaled.std(ddof=0).round(3),
    })
    print("\nAFTER scaling:")
    print(check.to_string())
    print("\nNote: train mean ~0 and std ~1 by construction.")
    print("Test mean/std are NOT exactly 0/1. That is correct: the scaler only")
    print("learned from train data, so no test information leaked.")

    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()