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


def load_dataset():
    AllInOne_Data = pd.read_csv(r"C:\Users\Sami El yaagoubi\Desktop\capstone\datasets\Data_cleaned_Dataset.csv", parse_dates=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])

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
        sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)

        metrics = {
            "Training dataset R^2": model.score(X_train, y_train),
            "Testing dataset R^2": model.score(X_test, y_test),
            "Training dataset accuracy": np.mean(np.sign(Ptrain) == np.sign(y_train)) * 100,
            "Testing dataset accuracy": np.mean(np.sign(Ptest) == np.sign(y_test)) * 100,
            "Training dataset ROI": df_returns.iloc[train_start:train_start+len(Ptrain)]['AlgoReturn'].sum() * 100,
            "Testing dataset ROI": df_returns.iloc[test_start:test_start+len(Ptest)]['AlgoReturn'].sum() * 100,
            "Sharpe Ratio": sharpe_ratio
        }
        descriptions = descriptions_others
        
    elif(model_name=='sign_randomForest_model.pkl'):
        df = df[['Electricity: Wtd Avg Price $/MWh']]
        df_returns = np.log(df).diff()
        df_returns['target'] = df_returns['Electricity: Wtd Avg Price $/MWh'].shift(-1)
        df_returns = df_returns.iloc[1:-1]

        test_size = int(len(df_returns) * 0.2)
        train_data = df_returns.iloc[:-test_size]
        test_data = df_returns.iloc[-test_size:]

        X_train = train_data['Electricity: Wtd Avg Price $/MWh'].to_numpy().reshape(-1, 1)
        X_test = test_data['Electricity: Wtd Avg Price $/MWh'].to_numpy().reshape(-1, 1)
        y_train = train_data['target'].to_numpy()
        y_test = test_data['target'].to_numpy()
        Ctrain = (y_train > 0)
        Ctest = (y_test > 0)

        model = load_models(model_name)
        Ptrain = model.predict(X_train)
        Ptest = model.predict(X_test)

        df_returns['Position'] = 0

        # Ensure the length of Ptrain matches the corresponding slice
        train_start = len(df_returns) - len(Ptrain) - test_size
        df_returns.iloc[train_start:train_start+len(Ptrain), df_returns.columns.get_loc('Position')] = Ptrain
        
        # Ensure the length of Ptest matches the corresponding slice
        test_start = len(df_returns) - test_size
        df_returns.iloc[test_start:test_start+len(Ptest), df_returns.columns.get_loc('Position')] = Ptest

        df_returns['AlgoReturn'] = df_returns['Position'] * df_returns['target']
        daily_return = df_returns['AlgoReturn'].dropna()
        sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)

        metrics = {
            "Training dataset R^2": model.score(X_train,Ctrain),
            "Testing dataset R^2": model.score(X_test,Ctest),
            "Training dataset accuracy": np.mean(Ptrain==Ctrain) * 100,
            "Testing dataset accuracy": np.mean(Ptest==Ctest) * 100,
            "Training dataset ROI": df_returns.iloc[train_start:train_start+len(Ptrain)]['AlgoReturn'].sum() * 100,
            "Testing dataset ROI": df_returns.iloc[test_start:test_start+len(Ptest)]['AlgoReturn'].sum() * 100,
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
    elif (model_name == 'price_ARIMA_model.pkl'):
        df=df[['Electricity: High Price $/MWh']]
        model = load_models(model_name)
        test_size = int(len(df) * 0.01)

        train_data = df[:-test_size]
        test_data = df[-test_size:]

        predictions = []

        for t in range(len(test_data)):
            # Forecast the next day
            model = ARIMA(train_data, order=(model.model_orders['ar'], model.model_orders['trend'], model.model_orders['ma'])).fit()
            forecast = model.forecast(steps=1)
            predictions.append(forecast)
            # Update training data with the new observed value
            train_data = pd.concat([train_data, test_data.iloc[t:t+1]])

        # Collect all metrics
        metrics = {
            "Mean Absolute Error": np.mean(np.abs(np.array(predictions) - test_data.values)),
            "Root Mean Squared Error": np.sqrt(np.mean((np.array(predictions) - test_data.values)**2)),
            "mse": mean_squared_error(test_data, predictions),
        }
        descriptions=descriptions_RMSE_MAE_MSE
    elif (model_name == 'price_gru_model.h5'):
        df = df[['Electricity: High Price $/MWh']]
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

        # Collect all metrics
        metrics = {
            "Mean Absolute Error": mean_absolute_error(y_test, predictions),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, predictions)),
            "Mean Squared Error": mean_squared_error(y_test, predictions)
        }
        descriptions=descriptions_RMSE_MAE_MSE
    elif (model_name == 'price_lstm_model.h5'):
        model = load_models(model_name)
        df = df[['Electricity: High Price $/MWh']]
        # Set seed for reproducibility
        tf.random.set_seed(7)

        # Assuming df is your DataFrame
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

def create_sequences(data, seq_length):
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
            latest_sequence = create_sequences(AllInOne_Data['price_scaled'].values, 1)
            X_test = latest_sequence[-1].reshape((1, 1, 1))
        else:  # for "price_gru_model.h5"
            X_test = AllInOne_Data['price_scaled'].values[-1].reshape((1, 1, 1))
        
        return X_test, scaler

    elif model_file == "price_ARIMA_model.pkl":
        data_size = int(len(AllInOne_Data) * 0.2)
        AllInOne_Data = pd.concat([AllInOne_Data.iloc[-data_size:], new_entry], ignore_index=True)
        X_test = AllInOne_Data['Electricity: Wtd Avg Price $/MWh'].values.flatten()
        
        return X_test, None
