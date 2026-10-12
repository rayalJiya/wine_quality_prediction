# Wine Quality Prediction System

End-to-end machine learning project that predicts red wine quality (scores 3-8)
from 11 chemical properties, with a Streamlit web app.

## Project structure
```
data/        WineQT.csv (raw), WineQT_clean.csv (cleaned)
src/         data cleaning, EDA, training, tuning, predictor
app/         Streamlit application
models/      saved models and feature metadata
reports/     result tables, figures, final_report.md
```

## Setup (Windows, PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline (in order)
```powershell
python src\clean_data.py
python src\eda.py
python src\correlation.py
python src\prepare_split.py
python src\train_models.py
python src\confusion_analysis.py
python src\feature_importance.py
python src\improve_model.py
python src\tune_model.py
python src\save_metadata.py
python src\generate_report.py
```

## Run the app
```powershell
streamlit run app\streamlit_app.py
```

## Method summary
- Target: `quality`; `Id` dropped; duplicates removed (1143 to 1018 rows)
- Stratified 80/20 split, random_state=42
- Scaling (StandardScaler) inside a Pipeline, so no data leakage
- Models: Logistic Regression, Random Forest, SVM, KNN
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Tuning: GridSearchCV with 5-fold stratified cross-validation
- Quality categories: 3-4 Poor, 5 Average, 6 Good, 7-8 Excellent

## Results and analysis
See `reports/final_report.md`.

## Limitations
Small dataset, strong class imbalance (classes 3 and 8 are very rare),
subjective labels, red wine only.