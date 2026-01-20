# ⚽ Champions League 2025-2026 Score Predictor

This project utilizes **Poisson Regression** and machine learning techniques to analyze match data from the 2025-2026 UEFA Champions League season. It predicts the most likely match scores and calculates win probabilities for any given pair of teams.

## 🚀 Features

- **Statistical Modeling:** Uses the Poisson Distribution to model goal scoring as a rare event.
- **xG Calculation:** Computes "Expected Goals" (xG) for both home and away teams based on historical attack and defense strengths.
- **Win Probabilities:** Displays precise percentage chances for a Home Win, Draw, or Away Win.
- **Interactive Interface:** A clean, user-friendly web UI built with **Streamlit**.

## 🧠 How It Works

The model evaluates each team's performance based on:
1.  **Attack Strength:** How many goals a team scores compared to the league average.
2.  **Defense Weakness:** How many goals a team concedes compared to the league average.
3.  **Home Advantage:** The statistical benefit of playing at home.

Using these metrics, it generates a matrix of all possible scorelines (0-0 to 5-5) to find the outcome with the highest probability.

## 🛠️ Tech Stack

- **Python:** Core logic and data processing.
- **Streamlit:** Web application framework.
- **Pandas:** Data manipulation and cleaning.
- **Statsmodels:** Implementation of the Poisson Generalized Linear Model (GLM).
- **SciPy:** Probability mass function calculations.

## 📦 Installation & Usage

To run this project locally on your machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/mdnine9/CL-Score-Predictor.git](https://github.com/mdnine9/CL-Score-Predictor.git)
   cd CL-Score-Predictor

2. Install requirements:
pip install -r requirements.txt

3. Run the App
streamlit run app.py

The data used in this project is sourced from Kaggle: Champions League Matches 2025-2026

