import numpy as np
import pandas as pd
import streamlit as st
from visualizations import final_balance_plotting, waiting_statement

def calculate_percentiles(data, window_size, percentile_20, percentile_80):
    data['Percentile_20'] = data['Electricity: Wtd Avg Price $/MWh'].rolling(window=window_size).apply(lambda x: np.percentile(x, percentile_20), raw=True)
    data['Percentile_80'] = data['Electricity: Wtd Avg Price $/MWh'].rolling(window=window_size).apply(lambda x: np.percentile(x, percentile_80), raw=True)
    return data

def run_percentile_strategy(starting_amount, data):
    st.write("Adjust Parameters:")
    window_size = st.slider("Window Size for Percentile-based Strategy", 1, 30, 14)
    percentile_20 = st.slider("Lower Percentile (Buy Signal)", 0, 50, 20)
    percentile_80 = st.slider("Upper Percentile (Sell Signal)", 50, 100, 80)

    start_date = st.date_input("Start Date for Plot", data['Trade Date'].min())
    end_date = st.date_input("End Date for Plot", data['Trade Date'].max())

    start_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(start_date)].index[0])
    end_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(end_date)].index[0])
    
    if st.button("Run Backtest"):
        waiting_statement()
        data = calculate_percentiles(data.iloc[start_idx:end_idx], window_size, percentile_20, percentile_80)

        data['Signal'] = 0
        data['Position'] = 0

        for i in range(window_size, len(data)):
            if data['Electricity: Wtd Avg Price $/MWh'].iloc[i] <= data['Percentile_20'].iloc[i]:
                data['Signal'].iloc[i] = 1  # Buy signal
            elif data['Electricity: Wtd Avg Price $/MWh'].iloc[i] >= data['Percentile_80'].iloc[i]:
                data['Signal'].iloc[i] = -1  # Sell signal
        
        data['Position'] = data['Signal'].replace(to_replace=0, method='ffill')
        data.fillna(method='ffill', inplace=True)

        total_roi = calculate_ROI(data)
        final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx)
        return data
    

def run_BOS_strategy(starting_amount, data):
    start_date = st.date_input("Start Date for Plot", data['Trade Date'].min())
    end_date = st.date_input("End Date for Plot", data['Trade Date'].max())

    start_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(start_date)].index[0])
    end_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(end_date)].index[0])
    
    if st.button("Run Backtest"):
        waiting_statement()
        data = BOS_logic(data.iloc[start_idx:end_idx], starting_amount)
        total_roi = calculate_ROI(data)
        final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx)
        return data

def BOS_logic(data, initial_capital):
    # Store the Trade Date column
    trade_dates = data['Trade Date'].values
    
    # Initialize parameters
    trend = True  # Assume an initial trend
    high = -np.inf
    low = np.inf
    close = data['Electricity: Wtd Avg Price $/MWh'].iloc[0]
    extrems_date = data.index[0]
    start_date = data.index[0]
    capital = initial_capital
    position = None

    results = []

    # Iterate through the data
    for current_date, current_row in data.iterrows():
        current_price = current_row['Electricity: Wtd Avg Price $/MWh']
        data_up_to_current_date = data.loc[:current_date, 'Electricity: Wtd Avg Price $/MWh']

        # Detect trend
        new_trend, relevant_data = detect_trend(data_up_to_current_date, extrems_date, trend, close)

        # Get latest high, low, and close
        high, low, close, start_date, extrems_date = get_latest_high_and_low(relevant_data, start_date, extrems_date, trend, new_trend, high, low, close)

        # Strategy: Buy or sell based on trend change (simulated)
        if position is None:
            if new_trend:
                position = 1
            else:
                position = -1
        elif new_trend and position == -1:
            capital += current_price
            position = 1
        elif not new_trend and position == 1:
            capital -= current_price
            position = -1

        # Store results for analysis
        results.append((current_date, current_price, trend, new_trend, high, low, close, capital, position))

        # Update trend
        trend = new_trend
    
    # Create a DataFrame to analyze results
    results = pd.DataFrame(results, columns=['Trade Date', 'Electricity: Wtd Avg Price $/MWh', 'Initial Trend', 'New Trend', 'High', 'Low', 'Close', 'Capital', 'Position']).set_index('Trade Date')
    
    # Attach the stored trade dates back to the DataFrame
    results['Trade Date'] = trade_dates
    return results

def detect_trend(data, extrems_date, trend, close_readfiles):
        latest_close = data.iloc[-1]
        if trend and latest_close < close_readfiles:
            trend = False
            data = data[data.index >= extrems_date]
        elif not trend and latest_close > close_readfiles:
            trend = True
            data = data[data.index >= extrems_date]
        return trend, data
def calculate_ROI(data):
    buy_price = None
    total_return = 0.0

    for i in range(1, len(data)):
        if data['Position'].iloc[i] == 1 and data['Position'].iloc[i - 1] == -1:
            buy_price = data['Electricity: Wtd Avg Price $/MWh'].iloc[i]
        elif data['Position'].iloc[i] == -1 and data['Position'].iloc[i - 1] == 1:
            if buy_price is not None:
                sell_price = data['Electricity: Wtd Avg Price $/MWh'].iloc[i]
                total_return += (sell_price - buy_price) / buy_price
                buy_price = None

    return total_return

def get_latest_high_and_low(data, start_date, extrems_date, initial_trend, new_trend, high, low, close):
        if initial_trend and not new_trend:
            low = np.inf
            close = np.inf
            high = None
            start_date = extrems_date
        elif not initial_trend and new_trend:
            high = -np.inf
            close = -np.inf
            low = None
            start_date = extrems_date

        if new_trend:
            for i in range(len(data)):
                if high <= data.iloc[i]:
                    high = data.iloc[i]
                    extrems_date = data.index[i]
                    temp = i
            for temp in range(temp, -1, -1):
                if close == data.iloc[temp]:
                    break
                if data.iloc[temp - 1] > data.iloc[temp] and data.iloc[temp + 1] > data.iloc[temp]:
                    close = data.iloc[temp]
                    break
        else:
            for i in range(len(data)):
                if low >= data.iloc[i]:
                    low = data.iloc[i]
                    extrems_date = data.index[i]
                    temp = i
            for temp in range(temp, -1, -1):
                if close == data.iloc[temp]:
                    break
                if data.iloc[temp - 1] < data.iloc[temp] and data.iloc[temp + 1] < data.iloc[temp]:
                    close = data.iloc[temp]
                    break

        if close in [None, np.inf, -np.inf]:
            close = data.iloc[0]

        return high, low, close, start_date, extrems_date

def strategy_description(strategy):
    descriptions = {
        "Break of Structure": (
            "Involves identifying a change in the market trend. Traders buy when the price breaks above a previous high, indicating a potential upward trend, and sell when the price breaks below a previous low, indicating a potential downward trend.\n"
            "Parameters required:\n"
            "- Price data (historical high prices).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "df['High'] = df['Electricity: Wtd Avg Price $/MWh']\n"
            "\n"
            "# Define thresholds\n"
            "high_threshold = df['High'].rolling(window=20).max()\n"
            "low_threshold = df['High'].rolling(window=20).min()\n"
            "\n"
            "# Buy and sell signals\n"
            "df['Buy'] = (df['High'] > high_threshold).astype(int)\n"
            "df['Sell'] = (df['High'] < low_threshold).astype(int)\n"
            "```"
        ),
        "Percentile Channel Breakout (Mean Reversion)": (
            "Involves buying when the price falls below a lower percentile and selling when it rises above an upper percentile of recent price data.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- Percentile thresholds (e.g., lower and upper percentiles).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "percentile_low = df['Electricity: Wtd Avg Price $/MWh'].quantile(0.1)\n"
            "percentile_high = df['Electricity: Wtd Avg Price $/MWh'].quantile(0.9)\n"
            "\n"
            "# Buy and sell signals\n"
            "df['Buy'] = (df['Electricity: Wtd Avg Price $/MWh'] < percentile_low).astype(int)\n"
            "df['Sell'] = (df['Electricity: Wtd Avg Price $/MWh'] > percentile_high).astype(int)\n"
            "```"
        ),
        "sign_linearRegression_model.pkl": (
            "Uses a linear regression model to predict future returns based on historical prices. The model predicts whether returns will be positive or negative, and positions are taken accordingly. The strategy aims to capitalize on predicted trends in the data.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "df['Returns'] = np.log(df['Electricity: Wtd Avg Price $/MWh']).diff()\n"
            "df['target'] = df['Returns'].shift(-1)\n"
            "df.dropna(inplace=True)\n"
            "\n"
            "# Train model\n"
            "X = df[['Electricity: Wtd Avg Price $/MWh']].values\n"
            "y = (df['target'] > 0).astype(int)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = model.predict(X_test)\n"
            "```"
        ),
        "sign_randomForest_model.pkl": (
            "Uses a Random Forest model to predict future returns based on various factors. The model predicts continuous return values, which are then used to set trading positions. Positive predictions lead to long positions. The strategy aims to leverage complex patterns in the data for better prediction accuracy.\n"
            "Parameters required:\n"
            "- ['Day', 'Month', 'Year', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday', 'return', 'Electricity: Daily Volume MWh % Change', 'Natural Gas: Henry Hub Natural Gas Spot Price % Change', 'pjm_load sum in MW % Change', 'temperature mean in C % Change']\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "df['Returns'] = np.log(df['Electricity: Wtd Avg Price $/MWh']).diff()\n"
            "df['target'] = df['Returns'].shift(-1)\n"
            "df.dropna(inplace=True)\n"
            "\n"
            "# Train model\n"
            "X = df[['Electricity: Wtd Avg Price $/MWh']].values\n"
            "y = df['target'].values\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)\n"
            "model = RandomForestRegressor()\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = model.predict(X_test)\n"
            "```"
        ),
        "sign_gru_model.keras": (
            "Uses a GRU (Gated Recurrent Unit) model to predict future returns based on sequences of historical prices. The model processes sequences of historical prices to predict whether future returns will be positive or negative. Positions are based on these predictions, with positive forecasts leading to long positions and negative forecasts leading to short positions. The strategy aims to capture temporal dependencies in the data for improved forecasting.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- Sequence length (e.g., 14 days).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "from sklearn.preprocessing import MinMaxScaler\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "scaler = MinMaxScaler(feature_range=(0, 1))\n"
            "data = scaler.fit_transform(df[['Electricity: Wtd Avg Price $/MWh']])\n"
            "\n"
            "# Define sequence length\n"
            "sequence_length = 14\n"
            "X_seq, y_seq = create_sequences(data, sequence_length)\n"
            "y_binary = (y_seq > 0).astype(int)\n"
            "\n"
            "# Split data\n"
            "X_train, X_test, y_train, y_test = train_test_split(X_seq, y_binary, test_size=0.2, random_state=1)\n"
            "\n"
            "# Define and train the GRU model\n"
            "model = tf.keras.models.Sequential([\n"
            "    tf.keras.layers.GRU(50, input_shape=(X_train.shape[1], X_train.shape[2])),\n"
            "    tf.keras.layers.Dense(1, activation='sigmoid')\n"
            "])\n"
            "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n"
            "model.fit(X_train, y_train, epochs=10, batch_size=32)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = (model.predict(X_test) > 0.5).astype(int)\n"
            "```"
        ),
        "sign_LSTM_model.keras": (
            "Uses an LSTM (Long Short-Term Memory) model to predict future returns based on sequences of historical prices. The model is designed to handle long-term dependencies and patterns in the data, making it well-suited for capturing trends over extended periods. Positions are set based on predictions, with positive forecasts leading to long positions and negative forecasts leading to short positions. The strategy aims to exploit temporal patterns for better market predictions.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- Sequence length (e.g., 14 days).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "scaler = StandardScaler()\n"
            "data = scaler.fit_transform(df[['Electricity: Wtd Avg Price $/MWh']])\n"
            "\n"
            "# Define sequence length\n"
            "seq_length = 14\n"
            "X_seq, y_seq = create_sequences(data, seq_length)\n"
            "\n"
            "# Define and train the LSTM model\n"
            "model = tf.keras.models.Sequential([\n"
            "    tf.keras.layers.LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2])),\n"
            "    tf.keras.layers.Dense(1)\n"
            "])\n"
            "model.compile(optimizer='adam', loss='mean_squared_error')\n"
            "model.fit(X_seq, y_seq, epochs=10, batch_size=32)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = model.predict(X_seq)\n"
            "```"
        ),
        "price_ARIMA_model.pkl": (
            "Uses an ARIMA (AutoRegressive Integrated Moving Average) model to forecast future prices based on historical data. The model is fit on the training data and makes iterative forecasts for the test period. Each forecast is updated with new observed values, and the strategy aims to predict future prices accurately by adjusting the model with real-time data.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- ARIMA order parameters (e.g., (p, d, q)).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "from statsmodels.tsa.arima_model import ARIMA\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "model = ARIMA(df['Electricity: Wtd Avg Price $/MWh'], order=(5, 1, 0))\n"
            "model_fit = model.fit(disp=0)\n"
            "\n"
            "# Forecast\n"
            "forecast = model_fit.forecast(steps=10)\n"
            "```"
        ),
        "price_gru_model.h5": (
            "Uses a GRU (Gated Recurrent Unit) model to predict future electricity prices based on sequences of historical prices. The model is trained on normalized data and makes predictions on future prices by capturing temporal dependencies in the price sequences. The strategy aims to improve forecasting accuracy by leveraging the GRU's ability to handle sequences of data.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- Sequence length (e.g., 1 day).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "from sklearn.preprocessing import MinMaxScaler\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "scaler = MinMaxScaler(feature_range=(0, 1))\n"
            "data = scaler.fit_transform(df[['Electricity: Wtd Avg Price $/MWh']])\n"
            "\n"
            "# Define sequence length\n"
            "sequence_length = 1\n"
            "X_seq, y_seq = create_sequences(data, sequence_length)\n"
            "\n"
            "# Define and train the GRU model\n"
            "model = tf.keras.models.Sequential([\n"
            "    tf.keras.layers.GRU(50, input_shape=(X_seq.shape[1], X_seq.shape[2])),\n"
            "    tf.keras.layers.Dense(1)\n"
            "])\n"
            "model.compile(optimizer='adam', loss='mean_squared_error')\n"
            "model.fit(X_seq, y_seq, epochs=10, batch_size=32)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = model.predict(X_seq)\n"
            "```"
        ),
        "price_lstm_model.h5": (
            "Uses an LSTM (Long Short-Term Memory) model to predict future electricity prices based on sequences of historical prices. The model is trained on standardized data and processes sequences of data to make predictions. The strategy aims to improve forecasting by capturing long-term dependencies and patterns in the data.\n"
            "Parameters required:\n"
            "- Price data (historical prices).\n"
            "- Sequence length (e.g., 14 days).\n"
            "\nExample Code:\n"
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "# Load and prepare data\n"
            "df = pd.read_csv('price_data.csv')\n"
            "scaler = StandardScaler()\n"
            "data = scaler.fit_transform(df[['Electricity: Wtd Avg Price $/MWh']])\n"
            "\n"
            "# Define sequence length\n"
            "seq_length = 14\n"
            "X_seq, y_seq = create_sequences(data, seq_length)\n"
            "\n"
            "# Define and train the LSTM model\n"
            "model = tf.keras.models.Sequential([\n"
            "    tf.keras.layers.LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2])),\n"
            "    tf.keras.layers.Dense(1)\n"
            "])\n"
            "model.compile(optimizer='adam', loss='mean_squared_error')\n"
            "model.fit(X_seq, y_seq, epochs=10, batch_size=32)\n"
            "\n"
            "# Make predictions\n"
            "y_pred = model.predict(X_seq)\n"
            "```"
        ),
        
        "price_randomForest_model.pkl": (
            "Uses a Random Forest model to predict future electricity prices based on various features such as historical prices, daily volume, natural gas prices, load, temperature, and weekday. The model is trained on a dataset with these features and aims to improve forecasting by leveraging the ensemble learning technique of Random Forests, which combines multiple decision trees to enhance prediction accuracy.\n"
            "Parameters required:\n"
            "- Historical price data.\n"
            "- Daily volume data.\n"
            "- Natural gas price data.\n"
            "- Load data.\n"
            "- Temperature data.\n"
            "- Weekday information.\n"
            "\nExample Code:\n"
            "```python\n"
            "import os\n"
            "import joblib\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error\n"
            "\n"
            "# Load and prepare data\n"
            "def load_dataset():\n"
            "    AllInOne_Data = pd.read_csv(r'datasets/Data_cleaned_Dataset.csv', parse_dates=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])\n"
            "    AllInOne_Data = AllInOne_Data.interpolate()\n"
            "    mean_non_zero = AllInOne_Data[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()\n"
            "    AllInOne_Data.loc[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero\n"
            "    return AllInOne_Data\n"
            "\n"
            "def prepare_data(df):\n"
            "    df_returns = df[['Trade Date', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday']]\n"
            "    df_returns.set_index(['Trade Date'], inplace=True)\n"
            "    df_returns.dropna(subset=['Electricity: Wtd Avg Price $/MWh'], inplace=True)\n"
            "    df_returns.interpolate(subset=['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'], inplace=True)\n"
            "    mean_non_zero = df_returns[df_returns['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()\n"
            "    df_returns.loc[df_returns['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero\n"
            "    df_returns['return'] = df_returns['Electricity: Wtd Avg Price $/MWh'].pct_change()\n"
            "    df_returns['target'] = df_returns['return'].shift(-1)\n"
            "    df_returns['Electricity: Daily Volume MWh % Change'] = df_returns['Electricity: Daily Volume MWh'].pct_change()\n"
            "    df_returns['Natural Gas: Henry Hub Natural Gas Spot Price % Change'] = df_returns['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'].pct_change()\n"
            "    df_returns['pjm_load sum in MW % Change'] = df_returns['pjm_load sum in MW (daily)'].pct_change()\n"
            "    df_returns['temperature mean in C % Change'] = df_returns['temperature mean in C (daily): US'].pct_change()\n"
            "    df_returns.dropna(inplace=True)\n"
            "    df_returns = pd.get_dummies(df_returns, columns=['Weekday'])\n"
            "    df_returns = df_returns[~((df_returns['Weekday_Friday'] == 1) | (df_returns['Weekday_Saturday'] == 1))]\n"
            "    df_returns.drop(columns=['Weekday_Friday', 'Weekday_Saturday'], inplace=True)\n"
            "    df_returns['direction'] = (df_returns['target'] > 0)\n"
            "    expected_feature_list = ['Day', 'Month', 'Year', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday_Monday', 'Weekday_Sunday', 'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'return', 'Electricity: Daily Volume MWh % Change', 'Natural Gas: Henry Hub Natural Gas Spot Price % Change', 'pjm_load sum in MW % Change', 'temperature mean in C % Change']\n"
            "    df_returns.insert(0, 'Day', df_returns.index.day)\n"
            "    df_returns.insert(1, 'Month', df_returns.index.month)\n"
            "    df_returns.insert(2, 'Year', df_returns.index.year)\n"
            "    X = df_returns[expected_feature_list]\n"
            "    y = df_returns['target'].dropna()\n"
            "    X = X.loc[y.index]\n"
            "    return X, y\n"
            "\n"
            "def load_models(model_name):\n"
            "    model_path = os.path.join('models', model_name)\n"
            "    return joblib.load(model_path)\n"
            "\n"
            "def predict_price_random_forest():\n"
            "    df = load_dataset()\n"
            "    X, y = prepare_data(df)\n"
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "    model = load_models('price_randomForest_model.pkl')\n"
            "    predictions = model.predict(X_test)\n"
            "    mae = mean_absolute_error(y_test, predictions)\n"
            "    rmse = np.sqrt(mean_squared_error(y_test, predictions))\n"
            "    mse = mean_squared_error(y_test, predictions)\n"
            "    return predictions, mae, rmse, mse\n"
            "\n"
            "# Example usage\n"
            "predictions, mae, rmse, mse = predict_price_random_forest()\n"
            "print(f'Predictions: {predictions}')\n"
            "print(f'Mean Absolute Error: {mae}')\n"
            "print(f'Root Mean Squared Error: {rmse}')\n"
            "print(f'Mean Squared Error: {mse}')\n"
            "```\n"
        )
    }
    st.write(descriptions.get(strategy, "Strategy not found"))


def trading_algo_roi_winrate(train_data, test_data, predictions):
    # Convert test_data to a Series (1-dimensional) before creating the DataFrame
    data = pd.DataFrame({'X_test': test_data.squeeze().shift(1), 'y_test': test_data.squeeze(), 'predictions': predictions})
    data['X_test'][0] = train_data['Electricity: Wtd Avg Price $/MWh'].iloc[-1]
    
    # Initialize positions
    data['position'] = 0
    for i in range(len(data)):
        if data['X_test'][i] < data['predictions'][i]:
            data['position'][i] = 1
        elif data['X_test'][i] > data['predictions'][i]:
            data['position'][i] = -1
        else:
            data['position'][i] = 0
    
    # Initialize the 'returns' and 'correct' columns
    data['returns'] = None
    data['correct'] = None
    
    # Main loop to calculate 'correct' values
    for i in range(len(data)):
        row_index = data.index[i]
        
        if data['position'][row_index] == 1:  # Long position
            if data['X_test'][row_index] < data['y_test'][row_index]:  # Market went up
                if data['y_test'][row_index] > data['predictions'][row_index]:
                    data.at[row_index, 'correct'] = 1  # Prediction lower than actual
                else:
                    data.at[row_index, 'correct'] = 0  # Prediction higher than actual
            else:
                data.at[row_index, 'correct'] = -1  # Market went down, wrong position
        
        elif data['position'][row_index] == -1:  # Short position
            if data['X_test'][row_index] > data['y_test'][row_index]:  # Market went down
                if data['y_test'][row_index] < data['predictions'][row_index]:
                    data.at[row_index, 'correct'] = 1  # Prediction higher than actual
                else:
                    data.at[row_index, 'correct'] = 0  # Prediction lower than actual
            else:
                data.at[row_index, 'correct'] = -1  # Market went up, wrong position
    
    # Calculate returns
    for index in data.index:
        if data.loc[index, 'correct'] == 1:
            data.loc[index, 'returns'] = abs((data.loc[index, 'predictions'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test'])
        elif data.loc[index, 'correct'] == 0:
            data.loc[index, 'returns'] = abs((data.loc[index, 'y_test'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test'])
        elif data.loc[index, 'correct'] == -1:
            data.loc[index, 'returns'] = abs((data.loc[index, 'y_test'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test']) * (-1)
    
    # Calculate win rate
    winrate = len(data[data['correct'].isin([1, 0])]) / len(data) * 100
    
    # Calculate ROI
    roi = data['returns'].sum() * 100
        
    return roi, winrate
