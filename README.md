# Titanic Survival Predictor

A machine learning model that predicts whether a Titanic passenger survived based on their details.

## About
Built using Python and Scikit-Learn. Trained on the real Titanic dataset of 891 passengers and achieves 78.3% accuracy — consistent with industry benchmarks for this dataset.

## How it works
1. Loads the Titanic dataset directly from the web
2. Selects key features: passenger class, sex, age, family size, and fare
3. Trains a Random Forest classifier on the data
4. Predicts survival and ranks which features mattered most

## Key Findings
- Age (29.1%) and Sex (27.2%) were the strongest predictors of survival
- Women and higher-paying passengers had significantly better survival rates
- The model confirmed the historical "women and children first" pattern in the data

## Tech Stack
- Python
- Scikit-Learn
- Pandas

## Results
- Accuracy: 78.3%
- Dataset size: 891 passengers