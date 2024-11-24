import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error,classification_report,roc_auc_score
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from visualizations import waiting_statement


def load_dataset():
    AllInOne_Data = pd.read_csv(r"datasets/Data_cleaned_Dataset.csv", parse_dates=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])

# Interpolate missing data
    AllInOne_Data = AllInOne_Data.interpolate()

    # Replace zero values with the mean of non-zero values
    mean_non_zero = AllInOne_Data[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()
    AllInOne_Data.loc[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero

    return AllInOne_Data

def load_models(model_name):
    model_path = os.path.join('models', model_name)
    if model_name.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_name.endswith('.keras'):
        return tf.keras.models.load_model(model_path)
    elif model_name.endswith('.h5'):
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError('Model name must end with .pkl or .keras or .h5')

def predict(model_name, data):
    # Make predictions using a loaded model
    model = load_models(model_name)
    predictions = model.predict(data)
    confidence_intervals = (predictions - 0.1, predictions + 0.1)
    return predictions, confidence_intervals

def calculate_metrics(model_name, df):
    descriptions_classification = {
        "Accuracy": "Proportion of correctly predicted labels out of all labels in the dataset.",
        "ROI": "Return on investment based on the model's predictions in the dataset.",
        "Sharpe Ratio": "Performance of the trading strategy adjusted for risk.",
        "Precision": "Proportion of true positive predictions out of all positive predictions.",
        "Recall": "Proportion of true positive predictions out of all actual positive instances.",
        "F1 Score": "Harmonic mean of precision and recall.",
        "AUC-ROC": "Model’s ability to distinguish between the positive and negative classes.",
        "loss": "Mean loss between the predicted and actual labels in the dataset.",
        "MAE": "Mean absolute error between the predicted and actual labels in the dataset."
    }
    descriptions_others = {
        "R^2": "Proportion of variance in the target variable explained by the model in the dataset.",
        "Accuracy": "Percentage of times the model's predicted direction (sign) matches the actual direction in the dataset.",
        "ROI": "Total return on investment based on the model's predictions in the dataset.",
        "Sharpe Ratio": "Performance of the trading strategy adjusted for risk, calculated using the daily returns of the algorithm."
    }
    
    descriptions_RMSE_MAE_MSE = {
        "MAE": "Mean absolute error between the predicted and actual labels in the dataset.",
        "RMSE": "Root mean squared error between the predicted and actual labels in the dataset, but gives more weight to large errors.",
        "MSE": "Mean squared error between the predicted and actual labels in the dataset."
    }
    
    if(model_name=='sign_linearRegression_model.pkl'):
        
        df = df[['Electricity: Wtd Avg Price $/MWh']]
        df_returns = np.log(df).diff()
        df_returns['target'] = df_returns['Electricity: Wtd Avg Price $/MWh'].shift(-1)
        df_returns=df_returns.iloc[1:-1]

        test_size = int(len(df_returns) * 0.2)
        train_data = df_returns.iloc[:-test_size]
        test_data = df_returns.iloc[-test_size:]

        X_train = train_data['Electricity: Wtd Avg Price $/MWh'].to_numpy().reshape(-1, 1)
        X_test = test_data['Electricity: Wtd Avg Price $/MWh'].to_numpy().reshape(-1, 1)
        y_train = train_data['target'].to_numpy()
        y_test = test_data['target'].to_numpy()

        model = load_models(model_name)
        Ptrain = model.predict(X_train)
        Ptest = model.predict(X_test)

        df_returns['Position'] = 0

        # Ensure the length of Ptrain matches the corresponding slice
        train_start = len(df_returns) - len(Ptrain) - test_size
        df_returns.iloc[train_start:train_start+len(Ptrain), df_returns.columns.get_loc('Position')] = (Ptrain > 0)

        # Ensure the length of Ptest matches the corresponding slice
        test_start = len(df_returns) - test_size
        df_returns.iloc[test_start:test_start+len(Ptest), df_returns.columns.get_loc('Position')] = (Ptest > 0)

        df_returns['AlgoReturn'] = df_returns['Position'] * df_returns['target']
        daily_return = df_returns['AlgoReturn'].dropna()
        st.write("Table of Returns: ")
        st.write(df_returns)

        metrics = {
            "Training dataset R^2": model.score(X_train, y_train),
            "Testing dataset R^2": model.score(X_test, y_test),
            "Training dataset accuracy": np.mean(np.sign(Ptrain) == np.sign(y_train)) * 100,
            "Testing dataset accuracy": np.mean(np.sign(Ptest) == np.sign(y_test)) * 100,
            "Training dataset ROI": df_returns.iloc[train_start:train_start+len(Ptrain)]['AlgoReturn'].sum() * 100,
            "Testing dataset ROI": df_returns.iloc[test_start:test_start+len(Ptest)]['AlgoReturn'].sum() * 100,
            "Sharpe Ratio": daily_return.mean() / daily_return.std() * np.sqrt(252)
        }

        descriptions = descriptions_others
        
    elif(model_name=='sign_randomForest_model.pkl'):
        
        df_returns=df[['Trade Date','Electricity: Wtd Avg Price $/MWh','Electricity: Daily Volume MWh','Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)','pjm_load sum in MW (daily)','temperature mean in C (daily): US','Weekday']]
        # Prepare the data

        model = load_models(model_name)
        df_returns.set_index(['Trade Date'],inplace=True)
        df_returns.dropna(subset=['Electricity: Wtd Avg Price $/MWh'],inplace=True)
        df_returns.interpolate(subset=['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'],inplace=True)
        mean_non_zero = df_returns[df_returns['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()
        df_returns.loc[df_returns['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero
        df_returns['return']=df_returns["Electricity: Wtd Avg Price $/MWh"].pct_change().dropna()
        df_returns['target']=df_returns['return'].shift(-1)

        # Calculate percentage change for the features
        df_returns['Electricity: Daily Volume MWh % Change'] = df_returns['Electricity: Daily Volume MWh'].pct_change()
        df_returns['Natural Gas: Henry Hub Natural Gas Spot Price % Change'] = df_returns['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'].pct_change()
        df_returns['pjm_load sum in MW % Change'] = df_returns['pjm_load sum in MW (daily)'].pct_change()
        df_returns['temperature mean in C % Change'] = df_returns['temperature mean in C (daily): US'].pct_change()
        df_returns.dropna(inplace=True)
        weekday_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df_returns['Weekday'] = df_returns['Weekday'].map(weekday_mapping)
        df_returns['direction']=(df_returns['target']>0)

        X = df_returns.drop(columns=["target", "direction"])

        X.insert(0, "Day", df_returns.index.day)
        X.insert(1, "Month", df_returns.index.month)
        X.insert(2, "Year", df_returns.index.year)

        y = df_returns["direction"]

        # Assuming X and y are already defined
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        Ptrain = model.predict(X_train)
        Ptest = model.predict(X_test)

        # Add a new column 'Position' initialized to 0
        df_returns['Position'] = 0

        # Ensure the length of Ptrain matches the corresponding slice
        train_indices = X_train.index
        df_returns.loc[train_indices, 'Position'] = Ptrain

        # Ensure the length of Ptest matches the corresponding slice
        test_indices = X_test.index
        df_returns.loc[test_indices, 'Position'] = Ptest

        # Calculate 'AlgoReturn' as the product of 'Position' and 'target'
        df_returns['AlgoReturn'] = df_returns['Position'] * df_returns['target']

        # Calculate daily return and Sharpe ratio
        daily_return = df_returns['AlgoReturn'].dropna()
        sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)

        # Calculate R^2 scores and accuracies
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        train_accuracy = np.mean(Ptrain == y_train) * 100
        test_accuracy = np.mean(Ptest == y_test) * 100

        # Calculate ROI
        train_start = df_returns.index.get_loc(train_indices[0])
        test_start = df_returns.index.get_loc(test_indices[0])
        train_roi = df_returns['AlgoReturn'].iloc[train_start:train_start+len(Ptrain)].sum() * 100
        test_roi = df_returns['AlgoReturn'].iloc[test_start:test_start+len(Ptest)].sum() * 100
        
        st.write(f"The weekday mapping is: {weekday_mapping}")
        st.write("Table of Returns: ")
        st.write(df_returns.drop(columns=['direction']))
        st.write("Each Feature Importance:")
        st.image(r'assets/Feature importance RF_CLASSIFIER.png')
        
        metrics = {
            "Training dataset R^2": train_r2,
            "Testing dataset R^2": test_r2,
            "Training dataset accuracy": train_accuracy,
            "Testing dataset accuracy": test_accuracy,
            "Training dataset ROI": train_roi,
            "Testing dataset ROI": test_roi,
            "Sharpe Ratio": sharpe_ratio
        }
        descriptions = descriptions_others
    elif(model_name == 'sign_gru_model.keras'):
        model = load_models(model_name)

        df['Returns'] = df['Electricity: Wtd Avg Price $/MWh'].pct_change()
        df.dropna(inplace=True)
        # Set sequence length
        sequence_length = 14

        # Prepare the features and target
        X = df[['Electricity: Wtd Avg Price $/MWh']].values
        y = df['Returns'].values

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, sequence_length)

        # Convert the target to binary classification: 1 if return is positive, 0 if negative
        y_binary = (y_seq > 0).astype(int)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_binary, test_size=0.2, random_state=1)
        
        # Predict
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_prob = model.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract individual metrics
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']
        accuracy = report['accuracy']

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred_prob)

        # Predict
        Ptrain = model.predict(X_train)
        Ptest = model.predict(X_test)

        # Convert predictions to binary
        Ptrain_binary = (Ptrain > 0.5).astype(int)
        Ptest_binary = (Ptest > 0.5).astype(int)

        # Create a DataFrame to simulate returns
        df_returns = pd.DataFrame({
            'Electricity: Wtd Avg Price $/MWh': df[['Electricity: Wtd Avg Price $/MWh']].values[sequence_length:].squeeze(),
            'target': df['Returns'].values[sequence_length:],  # Ensure alignment with sequences
            'Position': np.nan
        })

        # Calculate positions
        df_returns['Position'] = 0
        df_returns.iloc[:len(Ptrain_binary), df_returns.columns.get_loc('Position')] = Ptrain_binary.flatten()
        df_returns.iloc[len(Ptrain_binary):, df_returns.columns.get_loc('Position')] = Ptest_binary.flatten()

        # Calculate AlgoReturn
        df_returns['AlgoReturn'] = df_returns['Position'] * df_returns['target']

        daily_return = df_returns['AlgoReturn'].dropna()
        sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)

        # Calculate other metrics
        training_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        testing_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        training_mae = mean_absolute_error(y_train, Ptrain_binary)
        testing_mae = mean_absolute_error(y_test, Ptest_binary)
        st.write("Table of Returns: ")
        st.write(df_returns)
        # Collect all metrics
        metrics = {
            "Training dataset accuracy": model.evaluate(X_train, y_train, verbose=0)[1],
            "Testing dataset accuracy": model.evaluate(X_test, y_test, verbose=0)[1],
            "Training dataset ROI": df_returns.iloc[:len(Ptrain_binary)]['AlgoReturn'].sum() * 100,
            "Testing dataset ROI": df_returns.iloc[len(Ptrain_binary):]['AlgoReturn'].sum() * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "AUC-ROC": auc_roc,
            "Training dataset loss": training_loss,
            "Testing dataset loss": testing_loss,
            "Training dataset MAE": training_mae,
            "Testing dataset MAE": testing_mae
        }
        # Print metrics and their descriptions in a table
        descriptions = descriptions_classification    # st.write("Past Predictions vs. Actual Prices")
    # plot_predictions(model_name, df)
    elif(model_name == 'sign_LSTM_model.keras'):
        model = load_models(model_name)

        df['Returns'] = df['Electricity: Wtd Avg Price $/MWh'].pct_change().shift(-1).fillna(0)
        df.dropna(inplace=True)

        # Set sequence length
        sequence_length = 14

        # Prepare the features and target
        X = df[['Electricity: Wtd Avg Price $/MWh']].values
        y = df['Returns'].values

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, sequence_length)

        # Convert the target to binary classification: 1 if return is positive, 0 if negative
        y_binary = (y_seq > 0).astype(int)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_binary, test_size=0.2, random_state=1)

        # Predict
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_prob = model.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract individual metrics
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']
        accuracy = report['accuracy']

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred_prob)

        # Predict
        Ptrain = model.predict(X_train)
        Ptest = model.predict(X_test)

        # Convert predictions to binary
        Ptrain_binary = (Ptrain > 0.5).astype(int)
        Ptest_binary = (Ptest > 0.5).astype(int)

        # Create a DataFrame to simulate returns
        df_returns = pd.DataFrame({
            'Electricity: Wtd Avg Price $/MWh': df[['Electricity: Wtd Avg Price $/MWh']].values[sequence_length:].squeeze(),
            'target': df['Returns'].values[sequence_length:],  # Ensure alignment with sequences
            'Position': np.nan
        })

        # Calculate positions
        df_returns['Position'] = 0
        df_returns.iloc[:len(Ptrain_binary), df_returns.columns.get_loc('Position')] = Ptrain_binary.flatten()
        df_returns.iloc[len(Ptrain_binary):, df_returns.columns.get_loc('Position')] = Ptest_binary.flatten()

        # Calculate AlgoReturn
        df_returns['AlgoReturn'] = df_returns['Position'] * df_returns['target']

        daily_return = df_returns['AlgoReturn'].dropna()
        sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)

        # Calculate other metrics
        training_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        testing_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        training_mae = mean_absolute_error(y_train, Ptrain_binary)
        testing_mae = mean_absolute_error(y_test, Ptest_binary)
        st.write("Table of Returns: ")
        st.write(df_returns)
        # Collect all metrics
        metrics = {
            "Training dataset accuracy": model.evaluate(X_train, y_train, verbose=0)[1],
            "Testing dataset accuracy": model.evaluate(X_test, y_test, verbose=0)[1],
            "Training dataset ROI": df_returns.iloc[:len(Ptrain_binary)]['AlgoReturn'].sum() * 100,
            "Testing dataset ROI": df_returns.iloc[len(Ptrain_binary):]['AlgoReturn'].sum() * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "AUC-ROC": auc_roc,
            "Training dataset loss": training_loss,
            "Testing dataset loss": testing_loss,
            "Training dataset MAE": training_mae,
            "Testing dataset MAE": testing_mae
        }
        # Print metrics and their descriptions in a table
        descriptions = descriptions_classification
    elif (model_name == 'price_randomForest_model.pkl'):
        waiting_statement()
        # Prepare the data
        
        # Prepare the data
        model = load_models(model_name)
        df_returns = df[['Trade Date', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday']]
        df_returns.set_index(['Trade Date'], inplace=True)
        df_returns.dropna(subset=['Electricity: Wtd Avg Price $/MWh'], inplace=True)
        df_returns.interpolate(subset=['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'], inplace=True)
        mean_non_zero = df_returns[df_returns['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()
        df_returns.loc[df_returns['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero
        df_returns['return'] = df_returns["Electricity: Wtd Avg Price $/MWh"].pct_change()
        df_returns['target'] = df_returns['return'].shift(-1)

        # Calculate percentage change for the features
        df_returns['Electricity: Daily Volume MWh % Change'] = df_returns['Electricity: Daily Volume MWh'].pct_change()
        df_returns['Natural Gas: Henry Hub Natural Gas Spot Price % Change'] = df_returns['Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'].pct_change()
        df_returns['pjm_load sum in MW % Change'] = df_returns['pjm_load sum in MW (daily)'].pct_change()
        df_returns['temperature mean in C % Change'] = df_returns['temperature mean in C (daily): US'].pct_change()
        df_returns.dropna(inplace=True)

        # One-hot encode the 'Weekday' column
        df_returns = pd.get_dummies(df_returns, columns=['Weekday'])

        # Drop rows where 'Weekday_Friday' or 'Weekday_Saturday' are true
        df_returns = df_returns[~((df_returns['Weekday_Friday'] == 1) | (df_returns['Weekday_Saturday'] == 1))]

        # Drop the 'Weekday_Friday' and 'Weekday_Saturday' columns
        df_returns.drop(columns=['Weekday_Friday', 'Weekday_Saturday'], inplace=True)

        df_returns['direction'] = (df_returns['target'] > 0)

        # Reorder columns to match the expected feature list
        expected_feature_list = ['Day', 'Month', 'Year', 'Electricity: Wtd Avg Price $/MWh', 'Electricity: Daily Volume MWh', 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', 'pjm_load sum in MW (daily)', 'temperature mean in C (daily): US', 'Weekday_Monday', 'Weekday_Sunday', 'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'return', 'Electricity: Daily Volume MWh % Change', 'Natural Gas: Henry Hub Natural Gas Spot Price % Change', 'pjm_load sum in MW % Change', 'temperature mean in C % Change']

        # Insert date-related columns
        df_returns.insert(0, "Day", df_returns.index.day)
        df_returns.insert(1, "Month", df_returns.index.month)
        df_returns.insert(2, "Year", df_returns.index.year)

        # Reorder columns to match the expected feature list
        X = df_returns[expected_feature_list]

        y = df_returns["target"].dropna()

        # Ensure X and y have the same length
        X = X.loc[y.index]

        # Assuming X and y are already defined
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictions = model.predict(X_test)

        analyze_predictions(X_test['return'], y_test.values, predictions)

        # Collect all metrics
        metrics = {
            "Mean Absolute Error": mean_absolute_error(y_test, predictions),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, predictions)),
            "Mean Squared Error": mean_squared_error(y_test, predictions)
        }
        descriptions = descriptions_RMSE_MAE_MSE

    elif (model_name == 'price_ARIMA_model.pkl'):
        waiting_statement()
        df = df[['Electricity: Wtd Avg Price $/MWh']]
        model = load_models(model_name)
        test_size = int(len(df) * 0.001)

        train_data = df[:-test_size]
        test_data = df[-test_size:]

        predictions = []

        for t in range(len(test_data)):
            # Forecast the next day
            model = ARIMA(train_data, order=(model.model_orders['ar'], model.model_orders['trend'], model.model_orders['ma'])).fit()
            forecast = model.get_forecast(steps=1)
            
            
            if not forecast.predicted_mean.empty:
                predicted_value = forecast.predicted_mean.iloc[0]
                predictions.append(predicted_value)
            else:
                st.write("Forecast is empty, unable to retrieve predicted value.")
            
            # Update training data with the new observed value
            train_data = pd.concat([train_data, test_data.iloc[t:t+1]])

        
        analyze_predictions(
            test_data.squeeze().shift(1).fillna(train_data['Electricity: Wtd Avg Price $/MWh'].iloc[-test_size-1]),
            test_data.squeeze().fillna(0),
            predictions
        )
        # Collect all metrics
        metrics = {
            "Mean Absolute Error": np.mean(np.abs(np.array(predictions) - test_data.values)),
            "Root Mean Squared Error": np.sqrt(np.mean((np.array(predictions) - test_data.values)**2)),
            "mse": mean_squared_error(test_data, predictions),
        }
        descriptions=descriptions_RMSE_MAE_MSE
    elif (model_name == 'price_gru_model.h5'):
        df = df[['Electricity: Wtd Avg Price $/MWh']]
        model = load_models(model_name)
        tf.random.set_seed(7)
        # Split into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Normalize the datasets
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        X_train, y_train = train[:-1], train[1:]
        X_test, y_test = test[:-1], test[1:]

        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        # Generate predictions from the best model
        predictions = model.predict(X_test)
        
        
        analyze_predictions(X_test.flatten(), y_test.flatten(), predictions.flatten())
        # Collect all metrics
        metrics = {
            "Mean Absolute Error": mean_absolute_error(y_test, predictions),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, predictions)),
            "Mean Squared Error": mean_squared_error(y_test, predictions)
        }
        descriptions=descriptions_RMSE_MAE_MSE
    elif (model_name == 'price_lstm_model.h5'):
        model = load_models(model_name)
        df = df[['Electricity: Wtd Avg Price $/MWh']]
        # Set seed for reproducibility
        tf.random.set_seed(7)

        # Split into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Standardize the datasets
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        # Define sequence length
        seq_length = 14
        X_test, y_test = create_sequences(test,test, seq_length)

        # Make predictions
        y_pred = model.predict(X_test)

        # Flatten the arrays and ensure they have the same length
        X_test_flat = X_test.flatten()
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        min_length = min(len(X_test_flat), len(y_test_flat), len(y_pred_flat))
        X_test_flat = X_test_flat[:min_length]
        y_test_flat = y_test_flat[:min_length]
        y_pred_flat = y_pred_flat[:min_length]

        # Call the analyze_predictions function
        analyze_predictions(X_test_flat, y_test_flat, y_pred_flat)
        metrics = {
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Mean Squared Error": mean_squared_error(y_test, y_pred)
        }
        descriptions=descriptions_RMSE_MAE_MSE

    return metrics, descriptions




def create_sequences(data, target, sequence_length):
            xs, ys = [], []
            for i in range(len(data) - sequence_length):
                x = data[i:i + sequence_length]
                y = target[i + sequence_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

def create_sequences_2(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def prepare_input_for_prediction(inputs, model_file):
    AllInOne_Data = load_dataset()[['Electricity: Wtd Avg Price $/MWh']]
    new_entry = pd.DataFrame({'Electricity: Wtd Avg Price $/MWh': [inputs[0][0]]})

    if model_file in ["price_lstm_model.h5", "price_gru_model.h5"]:
        AllInOne_Data = pd.concat([AllInOne_Data, new_entry], ignore_index=(model_file == "price_gru_model.h5"))
        scaler = load_models("scaler.pkl")
        AllInOne_Data['price_scaled'] = scaler.transform(AllInOne_Data[['Electricity: Wtd Avg Price $/MWh']])
        
        if model_file == "price_lstm_model.h5":
            latest_sequence = create_sequences_2(AllInOne_Data['price_scaled'].values, 1)
            X_test = latest_sequence[-1].reshape((1, 1, 1))
        else:  # for "price_gru_model.h5"
            X_test = AllInOne_Data['price_scaled'].values[-1].reshape((1, 1, 1))
        
        return X_test, scaler

    elif model_file == "price_ARIMA_model.pkl":
        data_size = int(len(AllInOne_Data) * 0.2)
        AllInOne_Data = pd.concat([AllInOne_Data.iloc[-data_size:], new_entry], ignore_index=True)
        X_test = AllInOne_Data['Electricity: Wtd Avg Price $/MWh'].values.flatten()
        
        return X_test, None
    elif model_file == "price_randomForest_model.pkl":
        
        new_entry = pd.DataFrame(inputs)
        
        return new_entry, None

def analyze_predictions(X_test, y_test, predictions):
    # Convert test_data to a Series (1-dimensional) before creating the DataFrame
    data = pd.DataFrame({
        'X_test': X_test,
        'y_test': y_test,
        'predictions': predictions
    })
    data['position'] = 0
    
    for i in range(len(data)):
        if data['X_test'].iloc[i] < data['predictions'].iloc[i]:
            data['position'].iloc[i] = 1
        elif data['X_test'].iloc[i] > data['predictions'].iloc[i]:
            data['position'].iloc[i] = -1
        else:
            data['position'].iloc[i] = 0

    # Initialize the 'returns' and 'correct' columns
    data['returns'] = None
    data['correct'] = None

    # Main loop to calculate 'correct' values
    for i in range(len(data)):
        row_index = data.index[i]  # Get the actual index label for the current row

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

    # Iterate over the actual index of the DataFrame for returns calculation
    for index in data.index:
        if data.loc[index, 'correct'] == 1:
            data.loc[index, 'returns'] = abs((data.loc[index, 'predictions'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test'])
        elif data.loc[index, 'correct'] == 0:
            data.loc[index, 'returns'] = abs((data.loc[index, 'y_test'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test'])
        elif data.loc[index, 'correct'] == -1:
            data.loc[index, 'returns'] = abs((data.loc[index, 'y_test'] - data.loc[index, 'X_test']) / data.loc[index, 'X_test']) * (-1)
    st.write(data)
    # Streamlit output
    st.write(f'The Win rate is: {len(data[data["correct"].isin([1,0])])/int(len(data))*100:.2f}%')
    st.write(f'The ROI is: {data["returns"].sum()*100:.2f}%')
