# Explainable Predictive Process Monitoring Framework

## Overview

This project implements an Explainable Predictive Process Monitoring (PPM) framework developed as part of a Master's thesis in Business Informatics at Technische Hochschule Brandenburg.

The framework predicts

- Remaining processing time
- Delay risk

for running ticket and request processes and provides explainable predictions together with decision support recommendations.

---

## Features

- Remaining Time Prediction (XGBoost)
- Delay Risk Prediction (XGBoost + Isotonic Calibration)
- Explainability using SHAP
- Interactive Streamlit dashboard
- Decision support recommendations
- Ticket monitoring dashboard

---

## Technologies

- Python
- Streamlit
- XGBoost
- SHAP
- Scikit-Learn
- Plotly
- Pandas

---

## Project Structure

```
app_v4.py
gauges.py
handlungsempfehlung.py
ppm_utils.py
requirements.txt

data/
    tickets_it_demo.csv
    cache/
        models_classical.joblib
        isotonic_calibrators.joblib
        config.pkl
```

---

## Installation

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app_v4.py
```

---

## Research Context

This prototype was developed for the Master's thesis:

**Predictive Process Monitoring in kleinen und mittleren Unternehmen – Entwicklung und Evaluation eines praxistauglichen datengetriebenen Artefakts zur Vorhersage von Durchlaufzeiten und Verzögerungsrisiken in Ticket- und Request-Prozessen**

Technische Hochschule Brandenburg

## License

This repository is intended for academic and research purposes.
