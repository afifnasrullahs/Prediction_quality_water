# Water Quality Monitoring App

A Streamlit dashboard to inspect sensor datasets, review model performance, and simulate real-time water-quality predictions augmented with Groq-powered recommendations.

## Features
- **Data & Info**: summary statistics, label distributions, and sample rows for Panel A & B datasets.
- **Model & Evaluasi**: fixed benchmarking table per panel (Logistic Regression, Random Forest, XGBoost) plus confusion matrices from `Model/*.png`.
- **Simulasi Prediksi**: interactive form accepting `flow1`, `flow2`, `turbidity`, `tds`, `ph`. The best overall model (`Model/best_model_overall.pkl`) generates quality labels and triggers the Groq recommendation service.

## Project Structure
```
├─ app/
│  ├─ core/
│  │  ├─ config.py            # Paths, feature config, benchmark metadata
│  │  ├─ data_loader.py       # Lazy dataset access + summaries
│  │  ├─ model_manager.py     # Model registry, evaluation, prediction helpers
│  │  ├─ recommendation.py    # Groq Chat Completions wrapper
│  │  └─ container.py         # Wiring dependencies for the Streamlit app
├─ Dataset/                   # Raw & processed CSV (ignored in git)
├─ Model/                     # Trained artifacts (ignored in git)
├─ streamlit_app.py           # Main Streamlit entry point
├─ requirements.txt
└─ README.md
```

## Local Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` next to the project root with:
```
GROQ_API_KEY=your_api_key_here
```

## Running
```bash
venv\Scripts\activate
streamlit run streamlit_app.py
```
Open the URL shown in the terminal (usually http://localhost:8501).

## Deployment Tips
- Push the code (without large datasets/models) to GitHub.
- On Streamlit Community Cloud, point to `streamlit_app.py`, set `GROQ_API_KEY` in the app secrets, and optionally upload lightweight sample datasets if needed.

## Notes
- CSV datasets and `.pkl` model artifacts exceed GitHub’s 100 MB cap, so they’re excluded via `.gitignore`. Keep them locally or store in object storage for production use.
- Recommendation errors with message “Groq API sedang mengalami gangguan…” indicate temporary 5xx responses from Groq; retry later. 

