# Smart Music Streaming Subscriber Churn Prediction & Retention System

A college mini-project that trains a churn classifier on the selected music-streaming subscriber dataset and presents analytics plus individual risk predictions in a dark Streamlit dashboard.

## Project structure

```text
smart_music_churn_system/
├── data/
│   ├── train.csv                 # add the selected dataset here
│   └── test.csv                  # optional; currently not required for training
├── models/                       # trained model bundle is saved here
├── app.py                        # FastAPI prediction API
├── dashboard.py                  # Streamlit dashboard
├── train.py                      # preprocessing, model comparison, evaluation
└── requirements.txt
```

## 1. Add the dataset

Copy the selected `train.csv` into `data/train.csv`. It must contain a binary `churned` column where 1 means churned and 0 means stayed. `test.csv` can also be copied to `data/test.csv`, but it is not required for training.

The training script excludes `customer_id` and `signup_date` from model features. The date is intentionally excluded until a valid, defensible account-tenure feature is defined.

## 2. Set up the environment

Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
myenv\Scripts\activate.bat
```

## 3. Train and evaluate

```bash
python train.py
```

The script compares Logistic Regression and Random Forest on a stratified 80/20 holdout split, prints precision/recall/F1 and ROC-AUC, then refits the model with the higher validation ROC-AUC on all labelled training rows. It saves `models/churn_model.pkl`.

Do not report metrics from the full-data refit as holdout metrics. The metadata stores the selected model and holdout ROC-AUC.

## 4. Run the API

Open terminal 1:

```bash
uvicorn app:app --reload
```

API docs: http://127.0.0.1:8000/docs

## 5. Run the dashboard

Open terminal 2, activate the same environment, then:

```bash
streamlit run dashboard.py
```

Streamlit normally opens at http://localhost:8501.

## Important limitations

- The project predicts patterns associated with churn in this dataset; it does not prove why a particular subscriber leaves.
- Retention suggestions are rule-based and should be treated as decision support, not guaranteed interventions.
- The provided test file has no known target labels, so accuracy on it cannot be measured unless labels become available.
- The dashboard's risk bands (under 35%, 35–70%, 70% and above) are illustrative operating thresholds, not validated business cut-offs.
- Dataset provenance should be documented honestly in the synopsis.
