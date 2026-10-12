# Wine Quality Prediction: Final Analysis and Conclusion

Generated on 2026-10-11 from the result files in `reports/`.

## Results summary

### Model comparison (test set, 204 samples)

| model | train_accuracy | test_accuracy | precision_weighted | recall_weighted | f1_weighted | precision_macro | recall_macro | f1_macro |
|---|---|---|---|---|---|---|---|---|
| SVM | 0.6769 | 0.6029 | 0.5672 | 0.6029 | 0.5811 | 0.2949 | 0.2866 | 0.2869 |
| Logistic Regression | 0.6057 | 0.5980 | 0.5742 | 0.5980 | 0.5772 | 0.3620 | 0.2915 | 0.3043 |
| Random Forest | 1.0000 | 0.5833 | 0.5589 | 0.5833 | 0.5699 | 0.2841 | 0.2894 | 0.2861 |
| KNN | 0.6781 | 0.5588 | 0.5416 | 0.5588 | 0.5498 | 0.2772 | 0.2793 | 0.2780 |

### Original vs tuned model (SVM)

| model | train_accuracy | test_accuracy | test_f1_weighted | test_f1_macro |
|---|---|---|---|---|
| Original | 0.6769 | 0.6029 | 0.5811 | 0.2869 |
| Tuned | 0.6978 | 0.5882 | 0.5678 | 0.2769 |

---

# TASK 17: Final Model Analysis

**1. Most influential feature.** Random Forest importance ranks **alcohol** first. Permutation importance on the test set ranks **alcohol** first. By Pearson correlation the strongest link with quality is **alcohol** (r = 0.49). The three least important features are fixed acidity, residual sugar, free sulfur dioxide.

**2. Best algorithm.** **SVM** performed best (weighted F1 = 0.581, test accuracy = 60.3%). The weakest was **KNN** (weighted F1 = 0.550).

**3. Why the best model won.** Wine quality depends on non-linear effects and interactions between chemical properties, and a tree ensemble can capture both without needing scaled inputs. Models with a simple or distance-based boundary struggle when the quality classes overlap this heavily. Note that the test set is small, so a gap of one or two percentage points between models can be noise.

**4. Did scaling help?**

- **Logistic Regression**: test accuracy 56.4% (unscaled) vs 59.8% (scaled), so scaling improved.
- **SVM**: test accuracy 52.4% (unscaled) vs 60.3% (scaled), so scaling improved.
- **KNN**: test accuracy 47.1% (unscaled) vs 55.9% (scaled), so scaling improved.

Random Forest was not scaled because trees split on thresholds and are not affected by feature ranges.

**5. Did tuning help?** GridSearchCV (5-fold stratified CV, scaler inside the pipeline, test set untouched) chose `{'model__C': 100, 'model__class_weight': None, 'model__gamma': 0.01}` with a cross-validation accuracy of 60.2% (std 0.031). On the test set, accuracy went from 60.3% to 58.8%, so the result did not improve (slightly lower). One test sample equals 0.49 percentage points, so small changes should not be over-interpreted. In the manual experiments the parameter with the largest effect on CV accuracy was **model__class_weight**, and the combined "improved" model stayed essentially the same on the test set. The train accuracy of the original model was 67.7% against 60.3% on test, which shows the overfitting gap. Higher training accuracy alone does not mean a better model.

**6. Difficult quality classes.** Classes with under 30% recall for SVM: **3, 4, 8**. Per-class recall: 3: 0.0%, 4: 0.0%, 5: 74.7%, 6: 59.8%, 7: 37.5%, 8: 0.0%. The most frequent confusion is actual 6 predicted as 5 (28 samples). Of 81 wrong predictions, 74 (91.4%) were off by exactly one level, so the model mostly confuses neighbouring scores.

**7. Dataset limitations.**
- Small: 1018 rows after cleaning (125 duplicates removed from 1143).
- Strong class imbalance: quality 3: 6, quality 4: 33, quality 5: 433, quality 6: 409, quality 7: 122, quality 8: 15. Classes 3 and 8 have too few samples for reliable precision or recall.
- Quality is a subjective human score, so identical wines can receive different labels (label noise).
- Only chemical properties are included. Grape variety, region, brand and aroma are missing.
- Only red wine samples, so results may not transfer to other wine types.
- The test set is small, so metrics have wide uncertainty.

---

# TASK 18: Final Conclusion

**Dataset understanding.** The dataset has 1143 wine samples with 11 chemical properties, an `Id` column and a target `quality` (scores 3 to 8). It is treated as multi-class classification.

**Data cleaning.** `Id` was dropped. There were no missing values. 125 duplicate rows (identical features and quality) were removed, leaving 1018 rows. No negative or physically invalid values were found. Outliers were kept because high sulfur dioxide or sugar values are plausible for wine.

**Exploratory Data Analysis.** Quality is concentrated in classes 5 and 6. Alcohol, volatile acidity and sulphates show clear patterns against quality, and several features contain outliers.

**Feature engineering and scaling.** All 11 features were used. StandardScaler was fitted on training data only, inside a Pipeline, for Logistic Regression, SVM and KNN, to avoid leakage.

**Models used.** Logistic Regression, Random Forest, SVM (RBF) and KNN, on a stratified 80/20 split with random_state=42.

**Model evaluation.** Accuracy, weighted and macro precision, recall and F1, per-class reports and confusion matrices were used, because accuracy alone hides poor results on rare classes.

**Best-performing model.** SVM, with 60.3% test accuracy and weighted F1 of 0.581.

**Hyperparameter tuning.** Cross-validated GridSearchCV gave a CV accuracy of 60.2%. Final test accuracy: 58.8% (original: 60.3%).

**Wine quality prediction system.** A Streamlit app takes the 11 chemical values and shows the predicted score, the category (3-4 Poor, 5 Average, 6 Good, 7-8 Excellent) and the confidence, which is the model's predicted probability for the chosen class. It warns on low confidence and on out-of-range inputs.

**Important findings.** alcohol is the most important feature by model importance. The model is reliable mainly for classes 5 and 6, and errors are mostly between neighbouring scores. Rare classes are hard to predict because of very few samples.

**Real-world applications.** Quick screening of wine batches in a winery, supporting quality control before expert tasting, and pricing or grading support. It should assist human tasters, not replace them.

**What I learned.** *(Write this part in your own words. Suggested points: preventing leakage with Pipelines and a held-out test set, why accuracy can mislead on imbalanced data, why tuning gives small gains on noisy labels, and how to build a Streamlit app around a saved model.)*
