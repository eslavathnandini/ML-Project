# Time Series Forecasting - Data Size Impact Study

This repository contains a comprehensive empirical study evaluating the impact of training dataset size on the predictive accuracy of classical machine learning algorithms versus deep neural networks. 

Specifically, this project compares the data efficiency and data hunger of:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Regression (SVR)**
- **Long Short-Term Memory (LSTM) Networks**

The evaluation is conducted on the daily closing prices of **Tata Consultancy Services (TCS.NS)** from 2018 to 2024.

## Project Structure
- `generate_notebook.py`: A script that procedurally generates the core Jupyter Notebook containing the data pipeline, model configurations, and analysis logic.
- `time_series_forecasting_research.ipynb`: The self-contained interactive playground demonstrating the sliding-window scaling preprocessing and the 100-to-1174 size subset experimentation.
- `extract_assets.py`: A headless extraction tool that rigorously tests the models, dumps the validation error (MSE and MAE loss functions) into `results_table.csv`, and generates the visual matplotlib charts utilized for publication.
- `paper.tex` and `paper.pdf`: A fully typeset 13-page academic format (IEEEtran single-column) paper, detailing the mathematical foundations, literature background, and formal discussion on time series forecasting constraints based on the outputs of the underlying scripts.

## Key Findings
- **Data Efficiency:** Classical models (SVR and KNN) excel in heavily constrained, small chronological datasets ($N = 100, 300$). SVR demonstrates massive tolerance margins for generalization, preventing catastrophic failure seen in deep sequences handling limited historical inputs without strict regularization.
- **Data Hunger:** The LSTM neural architecture fundamentally underperforms at small sample tiers, but exponentially outpaces all non-parametric metrics once the historical exposure exceeds $N = 1000$, proving neural data hunger scales into optimal performance given big data conditions.

## Visual Assets
- `raw_stock_data.png`: Exploratory mapping of the TCS.NS training sets.
- `learning_curves.png`: Tracking predictive performance and relative bias plateaus against dataset scalability.
- `predictions_vs_actuals.png`: The consolidated sequence-to-scalar extrapolations on the untouched 20% test baseline.
- Individual `<model>_predictions.png` charts isolate algorithm reaction velocities to sharp market inversions.

## Research Context
Author: **Eslavath Nandini**  
Course: **Intro to Machine Learning EE5180**
