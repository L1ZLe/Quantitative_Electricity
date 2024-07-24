import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_predictions(model_name, data):
    st.write(f"Plotting predictions for {model_name}")
    # Placeholder code to plot predictions
    plt.plot(data)
    plt.title(f"Predictions vs Actuals for {model_name}")
    st.pyplot()

def plot_equity_curve(strategy_name, data):
    st.write(f"Plotting equity curve for {strategy_name}")
    # Placeholder code to plot equity curve
    plt.plot(data)
    plt.title(f"Equity Curve for {strategy_name}")
    st.pyplot()

def plot_trades(trades):
    st.write("Plotting individual trades")
    # Placeholder code to plot trades
    plt.plot(trades['Date'], trades['Profit/Loss'])
    plt.title("Individual Trades")
    st.pyplot()

def waiting_statement():
    st.write("Running backtest, please wait...")
    
def final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx):
    final_balance = starting_amount * (1 + total_roi)
    
    st.write(f"Final balance starting with ${starting_amount} and buying/selling 1 MWh of electricity: ${final_balance:.2f}")
    st.write(f"Total ROI: {total_roi*100:.2f}%")

    st.write("Price Plot with Position Indicator (Zoomed In):")
    dates = data.loc[start_idx:end_idx, 'Trade Date'].values
    prices = data.loc[start_idx:end_idx, 'Electricity: Wtd Avg Price $/MWh'].values
    positions = data.loc[start_idx:end_idx, 'Position'].values
    plt.figure(figsize=(15, 7))
    for i in range(1, len(dates)):
        if positions[i] == 1:
            plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='green')
        elif positions[i] == -1:
            plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='red')
    plt.title('Price Plot with Position Indicator (Zoomed In)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt)

def display_inputs(model_name):
    models_input_requirements = {
        "price_ARIMA_model.pkl": {"type": "price", "num_inputs": 1, "date_needed": True},
        "price_gru_model.h5": {"type": "price", "num_inputs": 1},
        "price_lstm_model.h5": {"type": "price", "num_inputs": 1},
        "sign_gru_model.keras": {"type": "direction", "num_inputs": 14},
        "sign_LSTM_model.keras": {"type": "direction", "num_inputs": 14},
        "################": {"type": "direction", "num_inputs": 3, "features": ["Temperature", "Gas Price", "Electricity Load"]},
        "sign_randomForest_model.pkl": {"type": "direction", "num_inputs": 1, "features": ["Today's Return"]},
        "sign_linearRegression_model.pkl": {"type": "direction", "num_inputs": 1, "features": ["Today's Return"]}
    }
    requirements = models_input_requirements.get(model_name, {})
    num_inputs = requirements.get("num_inputs", 0)
    features = requirements.get("features", [])

    inputs = []
    if features:
        for feature in features:
            value = st.number_input(f"Enter {feature}", value=0.0)
            inputs.append(value)
    else:
        for i in range(num_inputs):
            value = st.number_input(f"Day {i + 1} data", value=0.0)
            inputs.append(value)
    
    return np.array(inputs).reshape(1, -1) if len(inputs) > 0 else None
