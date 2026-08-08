# Airbnb Rental Arbitrage Analytics — New York City

**Academic Analytics Project | Purdue University**  
*Portfolio version enhanced following instructor feedback to improve categorical-variable treatment and model interpretation.*

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white)

---

## Overview

This project applies multiple linear regression and interactive analytics to evaluate Airbnb rental arbitrage opportunities in New York City.

The business scenario focuses on helping a prospective rental-arbitrage investor understand which listing characteristics are associated with higher nightly prices, identify listings above a $120 benchmark, and estimate the potential nightly price of a prospective property before signing a lease.

The project combines predictive modeling, model evaluation, business interpretation, interactive visualization, and a nightly-price simulator.

---

## Business Questions

The analysis addresses the following questions:

1. Which New York City boroughs are associated with higher nightly prices?
2. Are customer reviews associated with higher rental prices?
3. Do minimum-night requirements influence nightly prices?
4. Does listing availability affect nightly prices?
5. How accurately can listing characteristics be used to estimate nightly price?

---

## Data

The analysis uses an Airbnb dataset containing **4,892 New York City listings**.

Variables used include:

- Borough
- Minimum nights
- Number of reviews
- Reviews per month
- Availability during the year
- Nightly price

**Target Variable:** Nightly Price

The listing ID was excluded from modeling because it functions only as an identifier.

---

## Analytical Approach

A **Multiple Linear Regression** model was developed using an 80/20 training and test split.

Model performance was evaluated using:

- R²
- Adjusted R²
- RMSE
- MAE
- Training vs. test performance comparison

For this enhanced portfolio version, borough is treated as a **categorical variable using one-hot encoding**, with Manhattan serving as the reference category.

---

## Model Performance

| Metric | Training | Test |
|---|---:|---:|
| R² | 0.9812 | 0.9860 |
| Adjusted R² | 0.9811 | 0.9859 |
| RMSE | 7.1807 | 5.9183 |
| MAE | 0.7507 | 0.7565 |

The close training and test Adjusted R² values indicate that the model generalizes consistently to unseen data and does not show evidence of substantial overfitting.

---

## Selected Model Insights

Holding the other predictors constant, the enhanced model estimates that:

- **Brooklyn** listings are approximately **$47.54 lower per night** than comparable Manhattan listings.
- **Queens** listings are approximately **$89.62 lower per night** than comparable Manhattan listings.
- **Staten Island** listings are approximately **$134.66 lower per night** than comparable Manhattan listings.
- **Bronx** listings are approximately **$171.59 lower per night** than comparable Manhattan listings.
- One additional required minimum night is associated with an estimated **$0.18 increase** in nightly price.
- One additional review is associated with an estimated **$0.15 increase** in nightly price.
- One additional review per month is associated with an estimated **$0.20 increase** in nightly price.
- One additional available day per year is associated with an estimated **$0.16 increase** in nightly price.

These relationships represent associations identified in the historical data and should not be interpreted as causal effects.

---

## Interactive Application

The Gradio application includes:

- Executive KPI summary
- Interactive borough and listing filters
- Nightly price analysis by borough
- Reviews vs. nightly price analysis
- Minimum-night analysis
- Availability analysis
- Percentage of listings above the $120 benchmark
- Training and test model-performance metrics
- Business-friendly coefficient interpretation
- Interactive nightly-price simulator
- Reset controls and error handling

---

## Portfolio Enhancement

The original academic assignment represented borough using numeric codes from 1 through 5.

Instructor feedback noted that although this limitation was disclosed, borough is fundamentally a **categorical variable rather than a continuous numeric measure**.

For this portfolio version, the model was enhanced by applying categorical encoding to borough. This allows each borough to be compared directly with a reference borough rather than assuming an equal numeric distance between borough codes.

The revised specification also produced slightly stronger out-of-sample model performance.

---

## Problem–Data–Insights–Deployment Framework

| Component | Application |
|---|---|
| **Problem** | Evaluate listing characteristics associated with higher Airbnb nightly prices and support rental-arbitrage decisions. |
| **Data** | NYC Airbnb listing characteristics including borough, reviews, minimum nights, availability, and price. |
| **Insights** | Regression analysis identifies relationships between listing characteristics, location, and expected nightly price. |
| **Deployment** | Interactive Gradio dashboard and price simulator translate analytical results into a decision-support application. |

---

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- Plotly
- Gradio
- Google Colab
- Multiple Linear Regression

---

## Repository Contents

- `Airbnb_Rental_Arbitrage.ipynb` — complete Google Colab-compatible analysis and application
- `airbnb_arbitrage_app.py` — standalone Python application
- `requirements.txt` — required Python packages
- `README.md` — project documentation

---

## Important Note

Model estimates are based on historical Airbnb listing patterns and should support, not replace, business judgment.
