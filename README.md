# GMF (Guide Me in Finance) Investments

## 📌 Project Overview
GMF Investments is a data-driven financial analysis project focused on Tesla (TSLA) stock. It includes time series forecasting, risk analysis, and portfolio optimization to guide investment decisions.

**🚀 Deployed Dashboard:** [https://gmfinvest.streamlit.app/](https://gmfinvest.streamlit.app/)


## 📊 Project Workflow
The project follows a structured workflow:

1. **Exploratory Data Analysis (EDA)** - Clean and analyze Tesla stock data.
2. **Task 1: Data Preparation** - Load, preprocess, and split the dataset.
3. **Task 2: Time Series Forecasting** - Predict Tesla stock prices using SARIMAX.
4. **Task 3: Forecast Future Market Trends** - Analyze trends, volatility, and market risks.
5. **Task 4: Portfolio Optimization** - Construct an optimal portfolio with TSLA, BND, and SPY.

---

## 📂 Folder Structure
```
GMF-Investments/
│── data/                 # Raw and processed datasets
│── models/               # Trained models
│── notebooks/            # Jupyter Notebooks for analysi     
│── README.md             # Project documentation
│── requirements.txt      # Dependencies for the project
```

---

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vlazacck/GMF-Investments-git
   cd GMF-Investments
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Jupyter Notebook to explore the analysis:
   ```bash
   jupyter notebook
   ```

---

## 📈 Key Results
### ✅ **Time Series Forecasting (Task 2 & 3)**
- **Best Model**: SARIMAX(2,1,4)x(0,1,1,12)
- **Forecast Horizon**: 6-12 months
- **Performance Metrics:**
  - MAE: 93.03
  - RMSE: 113.62
- **Trend Analysis:** Moderate upward trend with periods of volatility.

### ✅ **Portfolio Optimization (Task 4)**
- **Assets Considered:** TSLA, BND, SPY
- **Optimal Weights:**
  - TSLA: 76.61%
  - BND: 23.37%
  - SPY: 0.02%
- **Risk-Return Analysis:**
  - Annualized Return: 3.88%
  - Annualized Risk: 0.47%
  - Sharpe Ratio: 4.02

---

## 📌 Future Enhancements
🔹 Expand forecasts to multiple stocks for diversified strategies.  
🔹 Implement deep learning models (LSTM, Transformer-based models).  
🔹 Develop an interactive dashboard for real-time insights.  

---

## 🤝 Contributing
1. Fork the repo and create a new branch.
2. Commit your changes with a descriptive message.
3. Submit a pull request for review.

---

## 🛠 Technologies Used
- **Python** (pandas, numpy, statsmodels, scikit-learn, matplotlib, seaborn)
- **Jupyter Notebook** for data analysis and visualization
- **SARIMAX** for time series forecasting
- **Monte Carlo Simulations** for portfolio optimization
- **Git & GitHub** for version control

---


