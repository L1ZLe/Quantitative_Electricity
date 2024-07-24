import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from model_module import calculate_metrics, load_models, prepare_input_for_prediction, load_dataset
from trading_strategies import run_percentile_strategy, run_BOS_strategy,strategy_description
from visualizations import plot_equity_curve, display_inputs


# Model names for price prediction
models_names_price = {
    "Select a model": "",
    "ARIMA": "price_ARIMA_model.pkl",
    "GRU": "price_gru_model.h5",
    "LSTM": "price_lstm_model.h5"
}

# Model names for direction prediction
models_names_direction = {
    "Select a model": "",
    "GRU": "sign_gru_model.keras",
    "LSTM": "sign_LSTM_model.keras",
    "Random Forest": "sign_randomForest_model.pkl",
    "Linear Regression": "sign_linearRegression_model.pkl"
}


def home():
    st.title("Electricity Trading Strategy Project")
    
    st.write("""
    Welcome to the Electricity Trading Strategy Project website! This platform showcases the work done in analyzing and developing trading strategies for the electricity market. Here's what you can explore:
    """)

    # Project Overview
    st.header("Project Overview")
    st.write("""
    This project aims to forecast electricity prices in the USA, specifically the PJM Interconnection, and develop trading strategies based on these forecasts and other known strategies. We employ advanced machine learning models such as SARIMA and GRU to predict price movements and create a robust trading strategy. The models will be evaluated based on their accuracy in predicting the direction of the next day's price and the accuracy of the predicted prices.
    """)

    # Key Features
    st.header("Key Features")
    st.write("""
    - **Model Overview**: Detailed explanations of the models used for price predictions.
    - **Data Exploration**: Interactive visualizations of historical electricity prices and other relevant data, such as natural gas prices, and how these variables impact electricity prices.
    - **Predictions**: Live predictions based on the latest data, allowing users to see next-day price forecasts based on the chosen model.
    - **Trading Strategy**: Comprehensive description of the trading logic and strategy implementation.
    - **Performance Metrics**: Evaluation of the trading strategy’s performance through various metrics such as Sharpe ratio, win rate, and ROI.
    - **Backtesting**: Tools to backtest the trading strategy on historical data to assess its viability.
    - **Risk Management**: Discussion on risk management techniques and tools to adjust strategy parameters.
    """)

    # Model Evaluation
    st.header("Model Evaluation")
    st.write("""
    We evaluate our models using two main criteria:
    - **Direction Accuracy**: The accuracy of predicting the direction of the next day's price movement.
    - **Price Accuracy**: The accuracy of the actual predicted prices compared to the real prices.
    """)

def data_exploration():
    from sklearn.preprocessing import MinMaxScaler
    import seaborn as sns

    # First dataset: Net electricity generation by source
    data_1 = pd.read_csv('datasets/Net_generation_United_States_all_sectors_monthly.csv')
    st.title("Data Exploration")
    
    # Display the raw data
    st.write("### Net electricity generation by source")
    st.write(data_1)

    # Convert 'Month' column to datetime format
    try:
        data_1['Month'] = pd.to_datetime(data_1['Month'], format='%b-%y', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the first dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_1 = data_1.dropna(subset=['Month'])

    # Select multiple sources to plot
    sources_1 = list(data_1.columns[1:])  # Assuming the first column is the date or time
    selected_sources_1 = st.multiselect("Select sources to plot from the first dataset", options=sources_1, default=sources_1)

    # Filter data for the selected sources
    plot_data_1 = data_1[['Month'] + selected_sources_1].dropna()

    # Plot the data
    fig_1, ax_1 = plt.subplots()
    for source in selected_sources_1:
        ax_1.plot(plot_data_1['Month'], plot_data_1[source], label=source)
    
    ax_1.set_title("Net Electricity Generation by Source")
    ax_1.set_xlabel("Month")
    ax_1.set_ylabel("Net Generation (thousand megawatthours)")
    ax_1.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_1)
    
    # Second dataset: Net generation by places
    data_2 = pd.read_csv(r"C:\Users\Sami El yaagoubi\Desktop\capstone\datasets\Net_generation_by places.csv")
    
    # Display the raw data
    st.write("### Net electricity generation by places")
    st.write(data_2)

    # Convert 'Month' column to datetime format
    try:
        data_2['Month'] = pd.to_datetime(data_2['Month'], format='%y-%b', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the second dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_2 = data_2.dropna(subset=['Month'])

    # Select multiple regions to plot
    regions = list(data_2.columns[1:])  # Assuming the first column is the date or time
    selected_regions = st.multiselect("Select regions to plot from the second dataset", options=regions, default=regions)

    # Filter data for the selected regions
    plot_data_2 = data_2[['Month'] + selected_regions].dropna()

    # Plot the data
    fig_2, ax_2 = plt.subplots()
    for region in selected_regions:
        ax_2.plot(plot_data_2['Month'], plot_data_2[region], label=region)
    
    ax_2.set_title("Net Electricity Generation by Places")
    ax_2.set_xlabel("Month")
    ax_2.set_ylabel("Net Generation (thousand megawatthours)")
    ax_2.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_2)

    file_path_3 = "datasets\Retail_sales_of_electricity_United_States_monthly.csv"
    data_3 = pd.read_csv(file_path_3)
    
    # Display the raw data
    st.write("### Retail sales of electricity")
    st.write(data_3)

    # Convert 'Month' column to datetime format
    try:
        data_3['Month'] = pd.to_datetime(data_3['Month'], format='%b-%y', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the third dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_3 = data_3.dropna(subset=['Month'])

    # Select multiple regions to plot
    types = list(data_3.columns[1:])  # Assuming the first column is the date or time
    selected_types = st.multiselect("Select types to plot from the third dataset", options=types, default=types)

    # Filter data for the selected regions
    plot_data_3 = data_3[['Month'] + selected_types].dropna()

    # Plot the data
    fig_3, ax_3 = plt.subplots()
    for type in selected_types:
        ax_3.plot(plot_data_3['Month'], plot_data_3[type], label=type)
    
    ax_3.set_title("Retail Sales of Electricity")
    ax_3.set_xlabel("Month")
    ax_3.set_ylabel("Sales (thousand megawatthours)")
    ax_3.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_3)

    st.subheader('Correlations between variables:')
    st.image("assets\Correlations between variables1.png")
    st.image("assets\Correlations between variables2.png")

    st.subheader("Relation between Electricity price and Temperature")
    st.image("assets\Relation between Electricity price and Temperature.png")

    st.subheader("Net_generated electricity in United States")
    st.image(r"assets\Net_generated electricity and Temperature.png")

    st.subheader("Average Electricity price by Month")
    st.image("assets\Average Electricity price by Month.png")

    st.subheader("Electricity seasonal decomposition")
    st.image("assets\Electricity seasonal decomposition.png")

    st.subheader("Natural Gas seasonal decomposition")
    st.image(r"assets\Natural Gas seasonal decomposition.png")

    st.title("Electricity and Natural Gas Data")
    st.write("Here is a preview of the dataset:")
    AllInOne_Data = load_dataset()
    st.dataframe(AllInOne_Data.head())
    # Date range slider
    st.write("Select the date range to display:")
    min_date = AllInOne_Data['Trade Date'].min().to_pydatetime()
    max_date = AllInOne_Data['Trade Date'].max().to_pydatetime()
    date_range = st.slider("Date", min_date, max_date, (min_date, max_date))
    # Filter data based on selected date range
    filtered_data = AllInOne_Data[(AllInOne_Data['Trade Date'] >= date_range[0]) & (AllInOne_Data['Trade Date'] <= date_range[1])]
    # Interactive graph for Electricity
    st.write("Electricity Prices Over Time:")
    fig_electricity = px.line(filtered_data, x='Trade Date', y='Electricity: Wtd Avg Price $/MWh', title='Electricity Prices Over Time')
    st.plotly_chart(fig_electricity)
    # Interactive graph for Natural Gas
    st.write("Natural Gas Prices Over Time:")
    fig_natural_gas = px.line(filtered_data, x='Trade Date', y='Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', title='Natural Gas Prices Over Time')
    st.plotly_chart(fig_natural_gas)

    
    st.write("### All in One Graph")
    # Drop rows with NaN values in the specified columns
    AllInOne_Data = AllInOne_Data.dropna(subset=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])
    # Get the list of types (assuming the first three columns are dates or time related)
    types = list(AllInOne_Data.columns[3:])
    # Multiselect widget for selecting types to plot
    selected_types = st.multiselect("Select variables to plot", options=types, default=[])
    # Prepare the data for plotting
    if selected_types:
        plot_AllInOne_Data = AllInOne_Data[['Trade Date'] + selected_types].dropna()
        # Normalize the selected columns
        scaler = MinMaxScaler()
        plot_AllInOne_Data[selected_types] = scaler.fit_transform(plot_AllInOne_Data[selected_types])
        # Plot the data
        fig_4, ax_4 = plt.subplots()
        for type in selected_types:
            ax_4.plot(plot_AllInOne_Data['Trade Date'], plot_AllInOne_Data[type], label=type)
        # Enhancements
        ax_4.set_title("Normalized Electricity Data Over Time")
        ax_4.set_xlabel("Trade Date")
        ax_4.set_ylabel("Normalized Value")
        ax_4.legend()
        plt.xticks(rotation=45)
        # Display the plot
        st.pyplot(fig_4)
    else:
        st.write("Please select at least one type to plot.")

    # New graph for moving average
    st.write("### Moving Average Graph")
    # User selects variable and moving average window
    selected_variable = st.selectbox("Select variable for moving average", options=types)
    moving_average_window = st.slider("Select moving average window (weeks)", min_value=20, max_value=100, value=20)
    # Prepare the data for the moving average plot
    if selected_variable:
        AllInOne_Data['Trade Date'] = pd.to_datetime(AllInOne_Data['Trade Date'])
        AllInOne_Data.set_index('Trade Date', inplace=True)
        plot_data_ma = AllInOne_Data[[selected_variable]].dropna()
        # Broadcast the weekly mean to all values in each week
        filled_df = plot_data_ma.copy()
        filled_df['Week'] = filled_df.index.strftime('%U')  # Add a new column to store the week number
        filled_df[selected_variable] = filled_df.groupby('Week')[selected_variable].transform(lambda x: x.fillna(x.mean()))
        # Calculate the rolling mean with the selected window
        filled_df[f'rolling_mean_{moving_average_window}'] = filled_df[selected_variable].rolling(window=moving_average_window).mean()
        # Plot the data
        fig_ma, ax_ma = plt.subplots()
        ax_ma.plot(filled_df.index, filled_df[selected_variable], label=selected_variable)
        ax_ma.plot(filled_df.index, filled_df[f'rolling_mean_{moving_average_window}'], label=f'Rolling Mean ({moving_average_window} weeks)', linestyle='--')
        # Enhancements
        ax_ma.set_title(f"{selected_variable} with {moving_average_window}-Week Rolling Mean")
        ax_ma.set_xlabel("Trade Date")
        ax_ma.set_ylabel(selected_variable)
        ax_ma.legend()
        plt.xticks(rotation=45)
        # Display the plot
        st.pyplot(fig_ma)
    else:
        st.write("Please select a variable to plot.")


            # New graph for non-linear relationships
    st.write("### Non-Linear Relationships Between Variables")
    # User selects two variables to plot
    variable_x = st.selectbox("Select variable for x-axis", options=types)
    variable_y = st.selectbox("Select variable for y-axis", options=types)
    # Prepare the data for the scatter plot
    if variable_x and variable_y:
        fig_non_linear, ax_non_linear = plt.subplots()
        sns.regplot(x=variable_x, y=variable_y, data=AllInOne_Data, ax=ax_non_linear, scatter_kws={'s':10}, line_kws={"color":"red"})
        # Enhancements
        ax_non_linear.set_title(f"Non-Linear Relationship between {variable_x} and {variable_y}")
        ax_non_linear.set_xlabel(variable_x)
        ax_non_linear.set_ylabel(variable_y)
        plt.xticks(rotation=45)
        # Display the plot
        st.pyplot(fig_non_linear)
    else:
        st.write("Please select variables to plot.")

def models_overview():
    st.title("Models Overview")
    
    st.write("""
    In this project, various models are employed to predict electricity prices and develop trading strategies. The models are categorized into those used for predicting the sign (direction) of price changes and those used for predicting the actual price. Below is a detailed description of each model used:
    """)

    # Predicting the Sign
    st.header("Predicting the Sign")
    st.subheader("Using Deep Learning:")
    st.write("""
    - **GRU Sign Detection**: 
      A Gated Recurrent Unit (GRU) model used to predict whether the next day's price will go up or down. 

      **Components**:
        - Update gate
        - Reset gate
        - Current memory content
        - Final memory at current time step

      **Special Features**:
        - Simplified architecture compared to LSTM
        - Faster training due to fewer parameters

      **Use Case in Predicting the Sign**:
        - To capture the sequential dependencies in electricity price changes and predict the direction.

      ![GRU Structure](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Gated_Recurrent_Unit%2C_base_type.svg/1920px-Gated_Recurrent_Unit%2C_base_type.svg.png)
    """)
    st.write("""
    - **LSTM Sign Detection**: 
      A Long Short-Term Memory (LSTM) model used for the same purpose as the GRU model.

      **Components**:
        - Forget gate
        - Input gate
        - Output gate
        - Cell state

      **Special Features**:
        - Ability to capture long-term dependencies
        - Effective in handling vanishing gradient problems

      **Use Case in Predicting the Sign**:
        - To utilize its memory capabilities for more accurate prediction of price direction over longer periods.

      ![LSTM Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
    """)

    st.subheader("Using Regression Models:")
    st.write("""
    - **Linear Regression**: 
      A basic regression model to predict the direction of the price change.

      **Components**:
        - Dependent variable
        - Independent variables
        - Coefficients
        - Intercept

      **Special Features**:
        - Simple implementation
        - Provides a baseline for comparison

      **Use Case in Predicting the Sign**:
        - To offer a straightforward approach to predicting price direction based on linear relationships.
        - Used the direction of lagged prices as a variable to predict the direction of the next day's price.
    """)
    st.image("assets\LinearRegression.png")
    st.write("""
    - **Random Forest**: 
      An ensemble learning method using multiple decision trees to improve prediction accuracy.

      **Components**:
        - Multiple decision trees
        - Bagging
        - Majority voting

      **Special Features**:
        - Reduces overfitting by averaging multiple decision trees
        - Handles large datasets with higher dimensionality

      **Use Case in Predicting the Sign**:
        - To enhance the accuracy of direction prediction by leveraging ensemble methods.
        - Used the direction of lagged prices as a variable to predict the direction of the next day's price.

      ![Random Forest](https://miro.medium.com/v2/resize:fit:1010/1*R3oJiyaQwyLUyLZL-scDpw.png)
    """)

    # Predicting the Price
    st.header("Predicting the Price")
    st.subheader("Naive Forecast:")
    st.write("""
    A simple model that uses the previous day's price as the forecast for the next day. 

    **Components**:
      - Previous day's price

    **Special Features**:
      - Minimal computation
      - Serves as a baseline for more complex models

    **Use Case in This Project**:
      - To provide a simple benchmark for evaluating the performance of other models.
    """)

    st.subheader("Random Forest:")
    st.write("""
    An ensemble learning method using multiple decision trees to predict the actual price.

    **Components**:
      - Multiple decision trees
      - Bagging
      - Aggregation of results

    **Special Features**:
      - Captures complex interactions between features
      - Provides robust and accurate predictions

    **Use Case in This Project**:
      - To predict actual electricity prices by capturing nonlinear relationships in the data.
      - Using features/variables such as natural gas prices and temperature to predict electricity prices.
    """)

    st.subheader("Machine Learning Models:")
    st.write("""
    - **ARIMA**: 
      An Autoregressive Integrated Moving Average model used for time series forecasting.

      **Components**:
        - **Autoregressive (AR) terms**: These represent past values of the forecast variable in a regression equation. AR terms model the dependency of the variable on its own past values, where "AR(p)" indicates "p" past values are used.
        - **Integrated (I) terms**: This reflects the differencing of raw observations in time series data to achieve stationarity. The "I(d)" term denotes the number of times differencing is applied to the series for achieving stationarity.
        - **Moving Average (MA) terms**: These involve the dependency between an observation and a residual error from a moving average model applied to lagged observations. MA terms in an "MA(q)" model use "q" past errors to forecast future values.
      
      **ACF and PACF in Time Series Modeling**:
        - **Autocorrelation Function (ACF)**: ACF measures the correlation between a time series and its lagged values. For AR models, ACF helps determine the order "p" by showing significant correlations at lags up to "p". For MA models, ACF drops off after lag "q", indicating the order of the moving average.
        - **Partial Autocorrelation Function (PACF)**: PACF measures the correlation between a time series and its lagged values while adjusting for the effects of intervening lags. PACF is useful in determining the order of AR models, as it shows direct effects of past lags on the current value, without the indirect effects of shorter lags.
        
      **Special Features**:
        - Effective for univariate time series
        - Captures various types of temporal patterns

      **Use Case in This Project**:
        - To model and forecast the electricity price time series based on past values and past forecast errors.

      ![ARIMA Model](https://pbs.twimg.com/media/GGR3W45akAAvsaf.png)
    """)
    st.image("assets/PACF_ACF.png",caption=" ACF and PACF plots of Electricity Price for ARIMA")
    st.write("""
    - **SARIMA**: 
      Seasonal ARIMA model to capture seasonality in the data.

      **Components**:
        - Seasonal autoregressive (SAR) terms
        - Seasonal integrated (SI) terms
        - Seasonal moving average (SMA) terms

      **Special Features**:
        - Incorporates seasonality into ARIMA
        - Models complex seasonal patterns

      **Use Case in This Project**:
        - To predict electricity prices by capturing both seasonal and non-seasonal patterns in the data.

      ![SARIMA Model](https://pbs.twimg.com/media/GBIbeegbkAAJpjz.png)
    """)
    st.write("""
    - **Auto ARIMA (SARIMA)**: 
      Automated selection of the best SARIMA model parameters.

      **Components**:
        - Automatic parameter tuning

      **Special Features**:
        - Simplifies model selection process
        - Enhances model accuracy by choosing optimal parameters

      **Use Case in This Project**:
        - To automate the process of finding the best SARIMA model for electricity price forecasting.

    """)
    st.write("""
    - **GARCH**: 
      Generalized Autoregressive Conditional Heteroskedasticity model to forecast volatility.

      **Components**:
        - Autoregressive terms for variance
        - Moving average terms for variance

      **Special Features**:
        - Models volatility clustering
        - Useful for financial time series

      **Use Case in This Project**:
        - To forecast the volatility of electricity prices, which can inform risk management strategies.

      ![GARCH Model](https://www.investopedia.com/thmb/NAR9L8kawoJ41JwUN-_gC6VEwLc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GARCH-9d737ade97834e6a92ebeae3b5543f22.png)
    """)


    st.subheader("Deep Learning Models:")
    st.write("""
    - **GRU**: 
      A Gated Recurrent Unit model for predicting the actual price.

      **Use Case in predicting the Price**:
        - To predict electricity prices by capturing temporal dependencies in the data.

    """)
    st.write("""
    - **LSTM**: 
      A Long Short-Term Memory model for price prediction.

      **Use Case in predicting the Price**:
        - To predict electricity prices by leveraging its long-term memory capabilities.

    """)
    st.write("""
    - **LSTM using Normalized Prices**: 
      LSTM model applied to normalized price data.

      **Components**:
        - Normalized input data

      **Special Features**:
        - Improved training efficiency
        - Enhanced model performance

      **Use Case in This Project**:
        - To achieve better model performance by normalizing the input price data.

    """)
    st.write("""
    - **LSTM Regression**: 
      LSTM model used in a regression setting for price prediction.

      **Components**:
        - Regression output layer

      **Special Features**:
        - Directly predicts continuous price values
        - Suitable for precise price forecasting

      **Use Case in This Project**:
        - To provide accurate price predictions by directly modeling the continuous price values.

    """)

    st.subheader("Model Creation, Training, and Prediction")
    st.write("""
    Here's a basic example of how a model is created, trained, and used for predictions:
    """)

    code = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample data
data = pd.read_csv('electricity_prices.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
    '''
    st.code(code, language='python')

def model_selection():
    AllInOne_Data = load_dataset()

    st.title("Model Selection for direction Prediction")
    st.write("Choose and compare different models for direction prediction.")

    direction_models = list(models_names_direction.keys())
    selected_direction_model = st.selectbox("Select a model", direction_models)

    # Show key metrics for the selected model
    if selected_direction_model and selected_direction_model != "Select a model":
        model_name = models_names_direction[selected_direction_model]
        strategy_description(model_name)
        st.write(f"Metrics for {selected_direction_model}:")
        metrics, descriptions = calculate_metrics(model_name, AllInOne_Data)
        st.write(pd.DataFrame({'Metric': descriptions.keys(), 'Description': descriptions.values()}))
        st.write(metrics)

    st.title("Model Selection for price Prediction")
    st.write("Choose and compare different models for price prediction.")

    price_models = list(models_names_price.keys())
    selected_price_model = st.selectbox("Select a model", price_models)

    # Show key metrics for the selected model
    if selected_price_model and selected_price_model != "Select a model":
        model_name = models_names_price[selected_price_model]
        strategy_description(model_name)
        metrics,descriptions = calculate_metrics(model_name, AllInOne_Data)
        st.write(f"Metrics for {selected_price_model}:")
        st.write(pd.DataFrame({'Metric': descriptions.keys(), 'Description': descriptions.values()}))
        st.write(metrics)

def predictions():
    import pmdarima as pm

    st.title("Electricity Prediction")

    # Step 1: Select prediction type
    prediction_type = st.selectbox("Select Prediction Type", ["", "Price", "Direction"])

    if prediction_type:
        if prediction_type == "Price":
            models_names = {
                "ARIMA": "price_ARIMA_model.pkl",
                "GRU": "price_gru_model.h5",
                "LSTM": "price_lstm_model.h5"
            }
        elif prediction_type == "Direction":
            models_names = {
                "GRU": "sign_gru_model.keras",
                "LSTM": "sign_LSTM_model.keras",
                "Random Forest": "sign_randomForest_model.pkl",
                "Linear Regression": "sign_linearRegression_model.pkl"
            }
        
        # Step 2: Select model based on prediction type
        model_name = st.selectbox("Select a Model", list(models_names.keys()))
        
        if model_name and model_name != "Select a model":
            model_file = models_names[model_name]
            model = load_models(model_file)
            
            # Step 3: Display inputs based on the model's requirements
            inputs = display_inputs(model_file)
            
            if inputs is not None:
                
                # Add a submit button
                if st.button("Submit"):
                    if model_name == "ARIMA":
                      inputs, scaler = prepare_input_for_prediction(inputs, model_file)
                      
                      # Automatically find the best ARIMA parameters
                      model = pm.auto_arima(inputs, seasonal=True, trace=True, error_action='ignore', suppress_warnings=True)
                      
                      # Fit the model on the entire dataset
                      model.fit(inputs)
                      
                      # Make the prediction with confidence intervals
                      forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)
                      prediction = forecast[0]
                      lower_bound, upper_bound = conf_int[0]
                      
                      st.write("The predicted price for the next day is: ${:.2f}".format(prediction))
                      st.write("95% confidence interval: ${:.2f} - ${:.2f}".format(lower_bound, upper_bound))
                    elif model_file == "price_gru_model.h5" or model_file == "price_lstm_model.h5":
                      inputs, scaler = prepare_input_for_prediction(inputs, model_file)

                      predicted_price_scaled = model.predict(inputs)
                      prediction = scaler.inverse_transform(predicted_price_scaled)

                      # # Assuming a normal distribution to estimate confidence intervals
                      # mean_prediction = prediction.mean()
                      # std_prediction = prediction.std()
                      # lower_bound = mean_prediction - 1.96 * std_prediction
                      # upper_bound = mean_prediction + 1.96 * std_prediction

                      predicted_value = prediction[0].item()
                      st.write(f"The predicted price for the next day is: ${predicted_value:.2f}")
                      #st.write(f"95% confidence interval: ${lower_bound:.2f} - ${upper_bound:.2f}")
                    else:
                      prediction = model.predict(inputs)[0]
                      
                      if prediction_type == "Price":
                          if model_name == "ARIMA":
                              st.write("The predicted price for the next day is: ${:.2f}".format(prediction))
                          else:
                              predicted_value = prediction[0].item()
                              st.write(f"The predicted price for the next day is: ${predicted_value:.2f}")
                      elif prediction_type == "Direction":
                          direction = "up" if prediction > 0 else "down"
                          st.write(f"The predicted direction for the next day is: {direction}")
            else:
                st.write("Please enter all required data for the selected model.")

def trading_strategies():
    AllInOne_Data = load_dataset()
    st.title("Trading Strategies")
    st.write("Explain and visualize trading strategies.")

    strategies = ["", ""]
    selected_strategy = st.selectbox("Select a strategy", strategies)

    if selected_strategy:
        st.write(f"Logic for {selected_strategy}:")
        st.write(strategy_description(selected_strategy))

        st.write("Flowchart/Diagrams:")
        # You can add code to visualize flowcharts or diagrams

        st.write("Performance Metrics:")
        #performance_metrics = backtest(selected_strategy, AllInOne_Data)
        #st.write(performance_metrics)

        st.write("Equity Curve:")
        plot_equity_curve(selected_strategy, AllInOne_Data)

def backtesting():
    AllInOne_Data = load_dataset()
    
    st.title("Backtesting")
    st.subheader("Interactive backtesting tool.")
    starting_amount = st.number_input("Starting Amount", value=1000, min_value=1000)
    strategies = ["Percentile Channel Breakout (Mean Reversion)", "Break of Structure"]
    selected_strategy = st.selectbox("Select a strategy for backtesting", strategies)
    
    strategy_description(selected_strategy)
    if selected_strategy == "Percentile Channel Breakout (Mean Reversion)":
        backtest_results = run_percentile_strategy(starting_amount, AllInOne_Data)
        
    elif selected_strategy == "Break of Structure":
        st.image("assets\BOS.png")
        backtest_results = run_BOS_strategy(starting_amount, AllInOne_Data)
        
    if backtest_results is not None:
        st.write("Backtest completed.")
        st.session_state['backtest_results'] = backtest_results
        st.session_state['strategy_name'] = selected_strategy
        export_results()

def export_results():
    if 'backtest_results' in st.session_state and 'strategy_name' in st.session_state:
        backtest_results = st.session_state['backtest_results']
        strategy_name = st.session_state['strategy_name']
        
        st.write(backtest_results)
        st.title("Export Results")
        st.write("Provide options to export backtesting results.")
        
        export_format = st.radio("Choose export format", ("CSV", "Excel"))

        if st.button("Export"):
            if export_format == "CSV":
                backtest_results.to_csv(f'{strategy_name}_backtest_results.csv')
                st.write("Results exported to CSV.")
            elif export_format == "Excel":
                backtest_results.to_excel(f'{strategy_name}_backtest_results.xlsx')
                st.write("Results exported to Excel.")
    else:
        st.title("No results available for export. Please run a backtest first.")

def contact():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from dotenv import find_dotenv, load_dotenv
    import os
    import openpyxl

    st.title("Contact Us")
    st.write("User feedback form.")
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    def send_email(name, email, message):
        # Email details
        
        from_email = os.getenv('EMAIL')
        from_password = os.getenv('EMAIL_PASSWORD')
        to_email = os.getenv('EMAIL')


        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = "New Contact Form Submission"

        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email, from_password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()
            return True
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return False

    st.write("Please fill out the form below to provide your feedback.")

    with st.form(key='contact_form'):
        name = st.text_input(label="Name")
        email = st.text_input(label="Email")
        message = st.text_area(label="Message")

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if send_email(name, email, message):
            st.success("Thank you for your feedback! Your message has been sent.")
        else:
            st.error("Failed to send your message. Please try again later.")

PAGES = {
    "Home": home,
    "Data Exploration": data_exploration,
    "Models Overview": models_overview,
    "Model Selection": model_selection,
    "Predictions": predictions,
    "Trading Strategies": trading_strategies,
    "Backtesting": backtesting,
    "Export Results": export_results,
    'Contact Us': contact
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
