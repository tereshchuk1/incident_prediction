# Factory Sensor Incident Prediction

## Task

Given a stream of factory sensor readings, predict whether an equipment incident will occur in the near future. The goal is to fire an alert early enough to allow preventive action.

## What Was Done

Generated synthetic sensor data (vibration, temperature, pressure) with randomly injected incident windows. Built a sliding-window binary classification pipeline: the model looks at the last W days of sensor history and predicts whether an incident will happen in the next H days.

Feature engineering includes STL trend decomposition and cyclic calendar encoding to capture seasonal patterns. Three models were trained and compared — Random Forest, SVM, and LSTM — all with chronological train/test split to prevent data leakage.

Alert thresholds were tuned per model using the Precision-Recall curve rather than the default 0.5, since the cost of missing an incident differs from the cost of a false alarm.

## Metrics

Models are evaluated on Precision, Recall, F1, ROC-AUC, and **Detection Lag** — how many steps early or late the model fires relative to the actual incident start. Negative lag means the model predicted before the incident began, which is the ideal outcome for a warning system.

## Results

| Model | Precision | Recall | F1 | ROC-AUC | Avg Detection Lag |
|-------|-----------|--------|----|---------|-------------------|
| Random Forest | 0.833 | 0.526 | **0.645** | 0.761 | +3.0 days |
| SVM | 0.875 | 0.368 | 0.519 | 0.683 | +3.0 days |
| LSTM | 0.889 | 0.421 | 0.571 | 0.709 | +3.0 days |

Random Forest achieved the best F1 score. All three models showed reasonable precision but struggled with recall due to the low frequency of incidents in the data.

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels tensorflow matplotlib
jupyter notebook time_series_test.ipynb
```
