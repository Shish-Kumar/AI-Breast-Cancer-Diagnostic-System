# 🎗️ AI Breast Cancer Diagnostic System

A professional, end-to-end Machine Learning web application designed to assist in the early detection of Breast Cancer using biopsy measurements. This system uses clinical-grade diagnostics powered by a Logistic Regression model trained on the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Features
- **High Accuracy**: Real-time diagnostic prediction with **97.37% accuracy**.
- **User-Friendly UI**: Premium medical-themed dashboard built with Streamlit.
- **Explainable AI**: Integrated tooltips explaining the medical significance of each biopsy parameter.
- **Robust Pipeline**: Automated feature selection (RFECV) and data scaling for maximum reliability.

##  Project Structure
```text
├── data/                    # Dataset source (csv)
├── models/                  # Trained models, Scalers, and Feature lists (pkl)
├── notebooks/               # Jupyter Notebook for EDA & Model Training
├── app.py                   # Streamlit Web Application
├── requirements.txt         # Project Dependencies
└── .gitignore               # Files ignored by git
```

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnostic.git
   cd breast-cancer-diagnostic
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🧬 Scientific Overview
The system analyzes 12 critical biopsy parameters (out of the original 30) that were scientifically selected for their predictive power. Features include:
- **Radius Mean**: Average distance from center to tumor boundary.
- **Texture Mean**: Surface roughness indicator.
- **Concavity**: Severity of depressions in the contour.
- ... and more.

##  Disclaimer
*This tool is for informational and educational purposes only. It is not intended to replace professional medical diagnosis or treatment. Always consult with a qualified healthcare provider for medical concerns.*

---
Created with ❤️ by Shish Kumar Kushwah
