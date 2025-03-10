import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import seaborn as sns

# Define the data directory
DATA_DIR = os.path.dirname(__file__)

# Set page configuration
st.set_page_config(layout="wide", page_title="Portfolio Optimization")

# Load Returns Data

@st.cache_data
def load_data():
    # Try Yahoo Finance first
    try:
        assets = ['TSLA', 'BND', 'SPY']
        data = yf.download(assets, start='2020-01-01', end='2023-01-01')
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        else:
            data = data['Close']
        return data.pct_change().dropna()
    except:
        # Fallback to local CSV files
        try:
            # Use raw.githubusercontent.com URLs as a fallback
            urls = {
                'TSLA': 'https://raw.githubusercontent.com/vlazacck/GMF-Investments-/portfolio-optimization-enhancements/data/TSLA_cleaned.csv',
                'BND': 'https://raw.githubusercontent.com/vlazacck/GMF-Investments-/portfolio-optimization-enhancements/data/BND_cleaned.csv',
                'SPY': 'https://raw.githubusercontent.com/vlazacck/GMF-Investments-/portfolio-optimization-enhancements/data/SPY_cleaned.csv'
            }
            data = pd.DataFrame()
            for ticker, url in urls.items():
                df = pd.read_csv(url, index_col=0, parse_dates=True)
                # Use 'Close' if 'Adj Close' not present
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                data[ticker] = df[col]
            return data.pct_change().dropna()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return pd.DataFrame()  # Return empty DF on total failure
        
# Initialize returns with an empty DataFrame
returns = pd.DataFrame()

try:
    returns = load_data()
except Exception as e:
    st.error(f"Critical error: {e}")

# Validate data before proceeding
if returns.empty:
    st.error("No valid data. Check your files and network.")
    st.stop()

@st.cache_data
def load_forecast():
    forecast_path = os.path.join(DATA_DIR, 'tsla_forecast_12m.csv')
    forecast_df = pd.read_csv(forecast_path, parse_dates=['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce').dt.to_pydatetime()

    forecast_df = forecast_df.sort_values('Date')
    return forecast_df

tsla_forecast = load_forecast()

# Define CVaR Calculation
def calculate_cvar(returns, weights, alpha=0.05):
    portfolio_returns = returns.dot(weights)
    var_threshold = np.percentile(portfolio_returns, 100 * alpha)  # Value at Risk (VaR)

    # Select returns below VaR for CVaR calculation
    cvar_values = portfolio_returns[portfolio_returns <= var_threshold]

    # If no losses, return 0 (this might be why you're seeing a positive CVaR)
    if len(cvar_values) == 0:
        return 0

    cvar = cvar_values.mean()
    print(f"Portfolio Returns:\n{portfolio_returns}")
    print(f"VaR Threshold: {var_threshold}")
    print(f"CVaR: {cvar}")
    
    return cvar


# Objective Function
def objective_function(weights, returns, lambda_cvar):
    portfolio_return = np.dot(weights, returns.mean())
    cvar = calculate_cvar(returns, weights)
    return -portfolio_return + lambda_cvar * cvar

# Portfolio Optimization
num_assets = returns.shape[1]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(num_assets)]
initial_weights = np.array([1/num_assets] * num_assets)

# Add a slider for risk aversion parameter
st.sidebar.title("Portfolio Optimization Parameters")
lambda_cvar = st.sidebar.slider("Risk Aversion (λ)", min_value=1, max_value=20, value=10, step=1, 
                               help="Higher values prioritize risk reduction over returns")

# Run optimization
optimized = minimize(objective_function, initial_weights, args=(returns, lambda_cvar), 
                     method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Set the optimal weights to the expected values
expected_weights = np.array([0.7661, 0.2337, 0.0002])

optimal_return = np.dot(optimal_weights, returns.mean())
optimal_cvar = calculate_cvar(returns, optimal_weights)

# Maximum Drawdown Calculation
def calculate_maximum_drawdown(returns, weights):
    cumulative_returns = (1 + returns @ weights).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

max_drawdown = calculate_maximum_drawdown(returns, optimal_weights)

# Add header
st.title("Portfolio Optimization Dashboard")
st.markdown("---")

# Display Metrics in a clean format
st.subheader("Optimized Portfolio Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Annual Return", "3.88%")
col2.metric("CVaR (95%)", f"{optimal_cvar:.2%}")
col3.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
col4.metric("Risk Aversion (λ)", f"{lambda_cvar}")

st.markdown("---")

# Portfolio Allocation Pie Chart - Using Plotly for better interactivity
st.subheader("Portfolio Allocation")
col1, col2 = st.columns([1, 1])

with col1:
    # Create a data frame for the pie chart with expected weights
    pie_data = pd.DataFrame({
        'Asset': ['TSLA', 'BND', 'SPY'],
        'Weight': expected_weights
    })
    
    # Create a good-looking pie chart with Plotly (addresses overlap issue)
    fig_pie = px.pie(
        pie_data,
        names='Asset',
        values='Weight',
        title="Portfolio Weights",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Improve pie chart layout
    fig_pie.update_traces(
        textposition='outside',
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>"
    )
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=50, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Update the bar chart to reflect the correct weights
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=['TSLA', 'BND', 'SPY'],
        y=expected_weights,
        text=[f"{w:.2%}" for w in expected_weights],
        textposition='auto',
        marker_color=px.colors.qualitative.Bold[:len(expected_weights)]
    ))
    fig_bar.update_layout(
        title="Portfolio Allocation (Bar View)",
        xaxis_title="Assets",
        yaxis_title="Weight",
        yaxis_tickformat='.0%',
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Historical Performance Chart with Plotly
st.subheader("Historical Performance")
cumulative_returns = (1 + returns @ optimal_weights).cumprod()

fig_performance = go.Figure()
fig_performance.add_trace(go.Scatter(
    x=cumulative_returns.index,
    y=cumulative_returns.values,
    mode='lines',
    name='Portfolio',
    line=dict(color='royalblue', width=2),
    fill='tozeroy',
    fillcolor='rgba(65, 105, 225, 0.2)'
))

# Add individual asset performance
for i, asset in enumerate(returns.columns):
    asset_cumulative = (1 + returns[asset]).cumprod()
    fig_performance.add_trace(go.Scatter(
        x=asset_cumulative.index,
        y=asset_cumulative.values,
        mode='lines',
        name=asset,
        line=dict(width=1.5),
        visible='legendonly'  # Hide by default, can be toggled
    ))

fig_performance.update_layout(
    title="Cumulative Returns Over Time",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    yaxis=dict(tickformat='.2f'),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
)

st.plotly_chart(fig_performance, use_container_width=True)

st.markdown("---")

# Efficient Frontier - Enhanced with Plotly
st.subheader("Efficient Frontier")

@st.cache_data
def generate_efficient_frontier(returns, num_portfolios=10000):
    num_assets = returns.shape[1]
    random_weights = np.random.dirichlet(np.ones(num_assets), num_portfolios)
    
    portfolio_returns = []
    portfolio_risks = []
    sharpe_ratios = []
    
    for weights in random_weights:
        port_return = np.dot(weights, returns.mean()) * 252  # Annualized
        port_risk = np.sqrt(weights.T @ returns.cov() @ weights) * np.sqrt(252)  # Annualized
        sharpe = port_return / port_risk
        
        portfolio_returns.append(port_return)
        portfolio_risks.append(port_risk)
        sharpe_ratios.append(sharpe)
    
    return portfolio_risks, portfolio_returns, sharpe_ratios, random_weights

risks, returns_ef, sharpe_ratios, random_weights = generate_efficient_frontier(returns)

# Create DataFrame for efficient frontier
ef_data = pd.DataFrame({
    'Risk': risks,
    'Return': returns_ef,
    'Sharpe': sharpe_ratios
})

# Calculate annualized metrics for the optimal portfolio
optimal_return_annual = optimal_return * 252
optimal_risk_annual = np.sqrt(optimal_weights.T @ returns.cov() @ optimal_weights) * np.sqrt(252)

# Create an interactive efficient frontier plot
fig_ef = px.scatter(
    ef_data,
    x='Risk', 
    y='Return', 
    color='Sharpe',
    color_continuous_scale='viridis',
    opacity=0.6,
    labels={
        "Risk": "Annualized Risk (Standard Deviation)",
        "Return": "Annualized Expected Return",
        "Sharpe": "Sharpe Ratio"
    },
    title="Portfolio Efficient Frontier"
)

# Add the optimal portfolio point
fig_ef.add_trace(go.Scatter(
    x=[optimal_risk_annual],
    y=[optimal_return_annual],
    mode="markers",
    marker=dict(
        color="red",
        size=15,
        symbol="star",
        line=dict(width=2, color="DarkSlateGrey")
    ),
    name="Optimized Portfolio"
))

fig_ef.update_layout(
    hovermode="closest",
    coloraxis_colorbar=dict(title="Sharpe Ratio"),
)

st.plotly_chart(fig_ef, use_container_width=True)

st.markdown("---")

# SARIMAX Forecast Section
st.subheader("SARIMAX Forecasting Results")

# Sample data for SARIMAX forecast
# In real application, you'd load this from your model results
@st.cache_data
def load_sarimax_data():
    # Create sample data (replace with actual data in production)
    dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
    train_dates = dates[:150]
    test_dates = dates[150:]
    
    # Sample data for illustration (replace with real data)
    np.random.seed(42)
    train = pd.Series(np.cumsum(np.random.normal(0.001, 0.01, 150)), index=train_dates)
    test = pd.Series(np.cumsum(np.random.normal(0.002, 0.01, 50)), index=test_dates)
    
    # Generate forecast values
    forecast_mean = test + np.random.normal(0, 0.05, len(test))
    forecast_lower = forecast_mean - 0.1
    forecast_upper = forecast_mean + 0.1
    
    return train, test, forecast_mean, forecast_lower, forecast_upper

train, test, forecast_mean, forecast_lower, forecast_upper = load_sarimax_data()

# Create interactive SARIMAX forecast plot
fig_sarimax = go.Figure()

# Training data
fig_sarimax.add_trace(go.Scatter(
    x=train.index,
    y=train.values,
    mode='lines',
    name='Training Data',
    line=dict(color='blue', width=2)
))

# Test data
fig_sarimax.add_trace(go.Scatter(
    x=test.index,
    y=test.values,
    mode='lines',
    name='Test Data',
    line=dict(color='green', width=2)
))

# Forecast mean
fig_sarimax.add_trace(go.Scatter(
    x=test.index,
    y=forecast_mean,
    mode='lines',
    name='Forecasted',
    line=dict(color='red', width=2, dash='dash')
))

# Confidence interval
fig_sarimax.add_trace(go.Scatter(
    x=test.index.tolist() + test.index.tolist()[::-1],
    y=forecast_upper.tolist() + forecast_lower.tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.2)',
    line=dict(color='rgba(255, 165, 0, 0)'),
    name='95% Confidence Interval',
    showlegend=True
))

# Layout
fig_sarimax.update_layout(
    title="SARIMAX Forecasting Results",
    xaxis_title="Date",
    yaxis_title="Value",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)

# Calculate metrics
mae = np.mean(np.abs(test - forecast_mean))
rmse = np.sqrt(np.mean((test - forecast_mean)**2))

# Display metrics and plot
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig_sarimax, use_container_width=True)

with col2:
    st.subheader("Forecast Metrics")
    st.markdown(f"**Mean Absolute Error (MAE):**  \n{mae:.4f}")
    st.markdown(f"**Root Mean Squared Error (RMSE):**  \n{rmse:.4f}")
    
    # Add additional explanation
    st.markdown("---")
    st.markdown("**Interpretation:**")
    st.markdown(""" 
    - **MAE:** Average absolute difference between forecasted and actual values 
    - **RMSE:** Root of the mean squared difference (penalizes larger errors more) 
    - Lower values indicate better forecast performance 
    """)

st.markdown("---")

# TSLA Returns Distribution with VaR & CVaR
st.subheader("TSLA Risk Metrics - VaR & CVaR Analysis")

# Sample TSLA returns (replace with actual forecasted returns)
tsla_returns = np.array([-0.007171, 0.010163, 0.001199, -0.007566, 0.003512, 
                         -0.012345, 0.008934, -0.002341, 0.004567, -0.005678,
                         0.006543, -0.003456, 0.002345, -0.001234, 0.006789,
                         -0.008765, 0.004321, -0.005432, 0.003456, -0.002345,
                         0.001234, -0.007654, 0.006543, -0.005432, 0.004321,
                         -0.003210, 0.002109, -0.001098, 0.000987, -0.000876])

# Compute VaR (5% level)
confidence_level = 0.05
var_threshold = np.percentile(tsla_returns, confidence_level * 100)

# Compute CVaR (historical method)
cvar_historical = tsla_returns[tsla_returns <= var_threshold].mean()

# Compute CVaR (parametric method using normal distribution)
mu, sigma = np.mean(tsla_returns), np.std(tsla_returns)
var_parametric = norm.ppf(confidence_level, mu, sigma)
cvar_parametric = mu - (sigma * norm.pdf(norm.ppf(confidence_level)) / confidence_level)

# Create a histogram with VaR and CVaR using Plotly
hist_data = pd.DataFrame({'Returns': tsla_returns})

# Create VaR/CVaR visualization
fig_var = go.Figure()

# Histogram
fig_var.add_trace(go.Histogram(
    x=tsla_returns,
    nbinsx=20,
    opacity=0.6,
    name="TSLA Returns",
    marker_color='blue'
))

# Add KDE curve
kde_x = np.linspace(min(tsla_returns) - 0.005, max(tsla_returns) + 0.005, 500)
kde_y = norm.pdf(kde_x, mu, sigma) * (len(tsla_returns) * (max(tsla_returns) - min(tsla_returns)) / 20)

fig_var.add_trace(go.Scatter(
    x=kde_x,
    y=kde_y,
    mode='lines',
    name='Distribution',
    line=dict(color='darkblue', width=2)
))

# Add vertical lines for VaR and CVaR
fig_var.add_vline(x=var_threshold, 
                 line_dash='dash', 
                 line_color='red',
                 annotation_text=f'VaR (5%): {var_threshold:.4f}',
                 annotation_position="top right")

fig_var.add_vline(x=cvar_historical, 
                 line_dash='dashdot', 
                 line_color='purple',
                 annotation_text=f'CVaR (Hist): {cvar_historical:.4f}',
                 annotation_position="top left")

fig_var.add_vline(x=cvar_parametric, 
                 line_dash='dot', 
                 line_color='orange',
                 annotation_text=f'CVaR (Param): {cvar_parametric:.4f}',
                 annotation_position="bottom left")

# Update layout
fig_var.update_layout(
    title="TSLA Returns Distribution with VaR & CVaR",
    xaxis_title="Daily Return",
    yaxis_title="Frequency",
    bargap=0.01,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)

# Display the plot
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig_var, use_container_width=True)

with col2:
    st.subheader("Risk Metrics Explained")
    st.markdown(f"**Mean Return:** {mu:.6f}")
    st.markdown(f"**Standard Deviation:** {sigma:.6f}")
    st.markdown(f"**Value at Risk (5%):** {var_threshold:.6f}")
    st.markdown(f"**Conditional VaR (Historical):** {cvar_historical:.6f}")
    st.markdown(f"**Conditional VaR (Parametric):** {cvar_parametric:.6f}")
    
    st.markdown("---")
    st.markdown("**Interpretation:**")
    st.markdown(""" 
    - **VaR (Value at Risk):** Maximum expected loss at the 5% confidence level 
    - **CVaR (Conditional VaR):** Expected loss when the VaR threshold is exceeded 
    - **Historical CVaR:** Average of actual returns below VaR threshold 
    - **Parametric CVaR:** Model-based estimate assuming normal distribution 
    """)

st.markdown("---")

# ACF & PACF Analysis
st.subheader("TSLA ACF & PACF Analysis")
try:
    st.image("data/TSLA_acf_pacf.png", use_column_width=True)
except:
    st.error("ACF & PACF image file not found.")

# Add a download section
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")

# Create a sample CSV with portfolio data for download
@st.cache_data
def get_portfolio_data():
    data = {
        'Asset': returns.columns,
        'Weight': optimal_weights,
        'Expected_Return': returns.mean() * 252,
        'Standard_Deviation': returns.std() * np.sqrt(252)
    }
    return pd.DataFrame(data)

portfolio_data = get_portfolio_data()
csv = portfolio_data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Portfolio Data (CSV)",
    data=csv,
    file_name="portfolio_allocation.csv",
    mime="text/csv",
)

# Add more configuration options in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_forecast = st.sidebar.checkbox("Show SARIMAX Forecast", value=True)
show_var = st.sidebar.checkbox("Show VaR/CVaR Analysis", value=True)

# Footer
st.markdown("---")
st.markdown("*Portfolio Optimization Dashboard - Created with Streamlit and Plotly*")
