# **📈 Quantitative Trading System: Regime-Filtered EMA Strategy**

## **📌 Project Overview**

This project implements an end-to-end **Quantitative Trading System** for NIFTY 50 derivatives. It integrates advanced Data Engineering, **Hidden Markov Models (HMM)** for regime detection, and Machine Learning classifiers to filter trade signals.

The system moves beyond simple technical analysis by identifying market regimes (Uptrend, Downtrend, Sideways) and using AI to predict the probability of trade success.

### **🎯 Key Objectives**

* **Data Engineering:** Processed 1 year of 5-minute interval data for NIFTY Spot, Futures, and Options.  
* **Regime Detection:** Classified market states using Gaussian HMM on Option Greeks (Delta, Gamma, Vega).  
* **Strategy Execution:** Implemented a **5/15 EMA Crossover** strategy, filtered by the detected regime.  
* **ML Enhancement:** Optimized performance using **XGBoost** and **Deep Neural Networks (LSTM/DNN)**.  
* **Outlier Analysis:** Identified "Jackpot" trades (\>3 Sigma) to understand high-performance anomalies.

## **📊 Performance Summary**

The system was backtested on 1 year of intraday data. The Machine Learning enhancement significantly reduced risk while maximizing precision.

| Strategy | Win Rate | Total Profit | Risk Profile |
| :---- | :---- | :---- | :---- |
| **Baseline (EMA Only)** | 26% | \~20,063 Points | **High Risk** (Trend Following) |
| **XGBoost Filter** | 31% | \~109 Points | **Moderate** (Noise Reduction) |
| **Neural Network** | **100%** | \~342 Points | **Ultra-Conservative** (Sniper Mode) |

**💡 Key Insight:** Trades identified as statistical outliers (Z-Score \> 3\) generated **10.3x more profit** than average trades. These "Jackpot" trades correlated strongly with expanding EMA gaps and specific volatility regimes.

## **📂 Repository Structure**

├── data/                  \# Processed Data (Spot, Fut, Opt, Regimes)  
├── models/                \# Saved Models (HMM, XGBoost, Keras)  
├── notebooks/             \# Jupyter Notebooks (Step-by-Step Analysis)  
│   ├── 1\_Data\_Acquisition.ipynb  
│   ├── 2\_Data\_Cleaning.ipynb  
│   ├── 3\_Feature\_Engineering.ipynb  
│   ├── 4\_Regime\_Detection.ipynb  
│   ├── 5\_Baseline\_Strategy.ipynb  
│   ├── 6\_ML\_Models.ipynb  
│   └── 7\_Outlier\_Analysis.ipynb  
├── plots/                 \# Visualizations (Regime Charts, Equity Curves)  
├── results/               \# Trade Logs, Metrics Reports  
├── src/                   \# Core Python Modules  
│   ├── data\_utils.py      \# Cleaning & Merging Logic  
│   ├── features.py        \# Greeks & Technical Indicators  
│   ├── regime.py          \# HMM Implementation  
│   ├── strategy.py        \# Signal Generation Engine  
│   └── ml\_models.py       \# ML Training & Prediction  
├── requirements.txt       \# Project Dependencies  
└── README.md              \# Project Documentation

## **🚀 How to Run**

### **1\. Prerequisites & Installation**

Ensure you have Python 3.10 or higher installed. Then, install the required dependencies:

pip install \-r requirements.txt

### **2\. Execution Methods**

You can run the project in two ways: either by executing the Python modules directly or by running the Jupyter Notebooks interactively.

#### **Option A: Run via Python Scripts (Recommended)**

Execute the following commands in your terminal in this specific order:

Step 1: Data Pipeline  
Clean the raw data and generate technical features (Greeks, EMAs).  
python src/data\_utils.py  
python src/features.py

Step 2: Regime Detection  
Train the Hidden Markov Model (HMM) to classify market states.  
python src/regime.py

Step 3: Strategy & Backtest  
Run the baseline strategy and the Machine Learning enhancement models.  
python src/strategy.py  
python src/ml\_models.py

#### **Option B: Run via Jupyter Notebooks**

Navigate to the notebooks/ directory and run the notebooks in numerical order:

1. 1\_Data\_Acquisition.ipynb  
2. 2\_Data\_Cleaning.ipynb  
3. 3\_Feature\_Engineering.ipynb  
4. 4\_Regime\_Detection.ipynb  
5. 5\_Baseline\_Strategy.ipynb  
6. 6\_ML\_Models.ipynb  
7. 7\_Outlier\_Analysis.ipynb

## **👤 Author**

* **Name:** Rachit Goyal  
* **Project Date:** January 2026  
* **Tools:** Python, Pandas, Scikit-Learn, TensorFlow, HMMlearn