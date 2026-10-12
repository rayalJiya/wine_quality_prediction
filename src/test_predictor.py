from predictor import out_of_range_features, predict_quality

# First row of the dataset (actual quality = 5)
sample = {
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}

result = predict_quality(sample)
print(f"Predicted Wine Quality: {result['predicted_quality']}")
print(f"Quality Category: {result['category']}")
print(f"Prediction Confidence: {result['confidence'] * 100:.0f}%")
print("\nAll class probabilities:")
for c, p in result["probabilities"].items():
    print(f"  Quality {c}: {p * 100:.1f}%")
print(f"\nOut-of-range features: {out_of_range_features(sample)}")