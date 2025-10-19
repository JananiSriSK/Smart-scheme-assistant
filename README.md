# Smart Scheme Assistant

The **Smart Scheme Assistant** is a system designed to recommend suitable government schemes to users based on their personal information and eligibility. It combines **machine learning** with **retrieval-augmented generation (RAG)** to provide accurate, context-aware recommendations.

🔗 **Live Demo**: [Smart Scheme Assistant](https://smart-scheme-assistant.streamlit.app/)

---

## Features

* **User Profile Input**: Collects user details (age, gender, caste, employment status, disability, education, etc.) in a simple form.
* **Machine Learning Model**: Multi-label classifier (XGBoost) trained on government schemes dataset to predict eligible schemes.
* **Gemini API + FAISS (RAG)**: Enhances recommendations with context retrieval from scheme descriptions, benefits, and eligibility details.
* **Unified Recommendation**: Combines ML predictions with RAG fallback to ensure relevant results even when ML predictions are insufficient.
* **Interactive Web App**: Built using **Streamlit**, deployed for easy access.

---

## Tech Stack

* **Frontend & Deployment**: Streamlit
* **Machine Learning**: Scikit-learn, XGBoost
* **Vector Database**: FAISS
* **Language Model**: Gemini API
* **Data Handling**: Pandas, NumPy
* **Scraping & Processing**: Selenium, Playwright

---

## Project Workflow

1. **Data Collection**

   * Scraped 3700+ government schemes using Selenium & Playwright.
   * Collected scheme details: *Scheme Name, State, Ministry, Eligibility, Benefits, Documents Required, Application Process*.

2. **Data Cleaning & Feature Engineering**

   * Removed duplicates, standardized formats, and engineered eligibility features (e.g., `is_student`, `is_disabled`, `belongs_to_minority`, etc.).
   * Conducted exploratory data analysis (EDA) to study non-linear patterns.

3. **Model Building**

   * Experimented with linear and non-linear models.
   * Chose **XGBoost multi-label classifier** as it outperformed linear models on non-linear dataset patterns.
   * Saved trained model along with feature and target mappings.

4. **RAG System with Gemini**

   * Indexed scheme details using **FAISS**.
   * Used **Gemini API** to extract structured features and for retrieval-augmented recommendations.

5. **Integration**

   * Built a unified recommendation pipeline (`chatbot_utils.py`).
   * ML model makes primary predictions; RAG provides fallback and extra context.

6. **Deployment**

   * Packaged into a **Streamlit app** (`app.py`).
   * Deployed at: [smart-scheme-assistant.streamlit.app](https://smart-scheme-assistant.streamlit.app/)

---

## File Structure

```
.
├── .devcontainer/                  # Dev container setup
├── datasets/                       # Dataset storage
├── scheme_vector_index_gemini/     # FAISS vector index built with Gemini embeddings
├── app.py                          # Streamlit app entry point
├── chatbot_utils.py                # ML + RAG unified logic
├── feature_columns.pkl             # Saved feature columns
├── target_columns.pkl              # Saved target labels
├── trained_model_xgb.pkl           # Trained ML model
├── step1_Scraping_scheme_links.ipynb
├── step2_playwright-scheme_info_extraction.ipynb
├── step3_EDA.ipynb
├── step4_scheme_feature_extract_api.ipynb
├── step5_feature_eng&model_building.ipynb
├── step6_build_index.ipynb
├── requirements.txt
└── README.md
```

---

## Setup & Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JananiSriSK/Smart-Scheme-Assistant.git
   cd Smart-Scheme-Assistant
   ```
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the app in your browser (`localhost:8501` when running locally).
2. Enter your profile details (age, gender, caste, employment, etc.).
3. Receive personalized scheme recommendations:

   * ML predictions (based on eligibility features).
   * RAG retrieval for additional context and fallback.

---

## Future Enhancements

* Improve model performance for rare eligibility labels.
* Add support for regional language queries.
* Enhance chatbot capabilities for interactive conversations.
* Expand dataset with more recent government schemes.


