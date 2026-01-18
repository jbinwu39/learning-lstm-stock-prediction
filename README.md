# Learning LSTM Stock Prediction 

This repository is a learning-focused implementation of an LSTM (Long Short-Term Memory) neural network for time-series forecasting using historical stock prices.

The goal of this project is to understand the end-to-end workflow for sequence modeling in deep learning, including data collection, preprocessing (scaling + windowing), model training, and evaluating predictions against real market prices.

---

## Project Overview

This project trains a 2-layer LSTM model to predict the **next closing price** based on a rolling window of the previous **60 trading days** of AAPL closing prices.

**Core steps:**
1. Download historical AAPL price data from Yahoo Finance  
2. Extract the **Close** price series  
3. Scale values using **MinMaxScaler**  
4. Convert the time series into supervised learning sequences (lookback window = 60)  
5. Train an LSTM model in Keras/TensorFlow  
6. Predict on the test set and evaluate performance using standard regression metrics  
7. Plot predictions vs actual closing prices  

---

## Dataset

- **Source:** Yahoo Finance (via `pandas_datareader` + `yfinance`)
- **Ticker:** `AAPL`
- **Date range:** `2010-01-01` to `2024-01-01`
- **Feature used:** `Close` price only

---

## Model Architecture

The notebook uses the following architecture:

- **LSTM(50 units, return_sequences=True)**
- **LSTM(50 units, return_sequences=False)**
- **Dense(25)**
- **Dense(1)** (final predicted price)

This structure is a common baseline for sequence regression tasks and is well-suited for learning how stacked LSTMs compress time-dependent information into a single prediction.

---

## Training Setup

- **Train/Test split:** 80% train, 20% test (time-ordered, no shuffling)
- **Lookback window:** 60 days
- **Scaling:** MinMax scaling
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Batch size:** 1
- **Epochs:** 2

---

## Evaluation Metrics

Model performance is measured using:

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of determination)

A prediction plot is also generated showing:
- training data (historical)
- actual test prices
- predicted test prices

---

## Results

The notebook outputs:
- a prediction curve overlaying actual closing prices
- numerical evaluation metrics (MSE/RMSE/MAE/R²)

Note: stock prices are noisy and influenced by external factors not included in this model. The LSTM can capture trends in the time series, but short-term volatility may reduce point-by-point accuracy.

---

## How to Run

### Option 1: Run the notebook
Open and run:

- `notebooks/Apple LSTM_realsuagecase.ipynb`

### Option 2: Install dependencies locally
Create a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
