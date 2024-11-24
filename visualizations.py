import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

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
        "price_randomForest_model.pkl": {"type": "direction", "num_inputs": 18, "features": ['Day', 'Month', 'Year', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday_Monday', 'Weekday_Sunday', 'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'return', 'Electricity: Daily Volume MWh % Change', 'Natural Gas: Henry Hub Natural Gas Spot Price % Change', 'pjm_load sum in MW % Change', 'temperature mean in C % Change']},
        "sign_randomForest_model.pkl": {"type": "direction", "num_inputs": 14, "features": ['Day', 'Month', 'Year', 'Electricity: Wtd Avg Price $/MWh',
            'Electricity: Daily Volume MWh',
            'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)',
            'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US',
            'Weekday', 'return', 'Electricity: Daily Volume MWh % Change',
            'Natural Gas: Henry Hub Natural Gas Spot Price % Change',
            'pjm_load sum in MW % Change', 'temperature mean in C % Change']},
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
