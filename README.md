
<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">QUANTATIVE_ELECTRICITY</h1>
</p>
<p align="center">
    <em><code>► An advanced platform for forecasting electricity prices and developing trading strategies.</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/L1ZLe/Quantative_Electricity?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/L1ZLe/Quantative_Electricity?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/L1ZLe/Quantative_Electricity?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/L1ZLe/Quantative_Electricity?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

## 🔗 Quick Links

> - [📍 Overview](#-overview)
> - [📦 Features](#-features)
> - [📂 Repository Structure](#-repository-structure)
> - [🧩 Modules](#-modules)
> - [⚙️ Installation](#️-installation)
> - [🤖 Running Quantative_Electricity](#-running-Quantative_Electricity)
> - [🛠 Project Roadmap](#-project-roadmap)
> - [🤝 Contributing](#-contributing)
> - [📄 License](#-license)
> - [👏 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

<code>► Welcome to the Electricity Trading Strategy Project! This platform provides comprehensive tools and models to forecast electricity prices and develop trading strategies. We utilize advanced machine learning models including SARIMA and GRU to predict price movements and assess trading strategies. Explore live predictions, backtesting tools, and performance metrics to understand and improve trading strategies in the electricity market.</code>

---

## 📦 Features

► Explore a range of features designed to enhance your trading strategy:

- **Model Overview**: Detailed descriptions of the models used for forecasting electricity prices.
- **Data Exploration**: Interactive visualizations of historical electricity prices and influencing factors.
- **Predictions**: Live forecasts of next-day electricity prices using various models.
- **Trading Strategy**: Insights into the logic and implementation of trading strategies.
- **Performance Metrics**: Evaluation of strategies using metrics such as Sharpe ratio, ROI, and more.
- **Backtesting**: Assess the performance of strategies on historical data.
- **Risk Management**: Techniques and tools for adjusting strategy parameters.


---

## 📂 Repository Structure

```sh
└── Quantative_Electricity/
    ├── README.md
    ├── app.py
    ├── assets
    │   ├── ARIMA_predictions.png
    │   ├── Average Electricity price by Month.png
    │   ├── BOS.png
    │   ├── Correlations between variables1.png
    │   ├── Correlations between variables2.png
    │   ├── Electricity seasonal decomposition.png
    │   ├── LinearRegression.png
    │   ├── Natural Gas seasonal decomposition.png
    │   ├── Net_generated electricity and Temperature.png
    │   ├── PACF_ACF.png
    │   ├── RandomForest.png
    │   ├── Relation between Electricity price and Temperature.png
    │   ├── gru_predictions.png
    │   └── lstm_predictions.png
    ├── datasets
    │   ├── Data_cleaned_Dataset.csv
    │   ├── Net_generation_United_States_all_sectors_monthly.csv
    │   ├── Net_generation_by places.csv
    │   └── Retail_sales_of_electricity_United_States_monthly.csv
    ├── model_module.py
    ├── models
    │   ├── price_ARIMA_model.pkl
    │   ├── price_gru_model.h5
    │   ├── price_lstm_model.h5
    │   ├── scaler.pkl
    │   ├── sign_LSTM_model.keras
    │   ├── sign_gru_model.keras
    │   ├── sign_linearRegression_model.pkl
    │   └── sign_randomForest_model.pkl
    ├── requirements.txt
    ├── trading_strategies.py
    └── visualizations.py
```

---

## 🧩 Modules

<details closed><summary>.</summary>

| File                                                                                                       | Summary                         |
| ---                                                                                                        | ---                             |
| [model_module.py](https://github.com/L1ZLe/Quantative_Electricity/blob/master/model_module.py)             | <code>► Contains functions for model training and predictions.</code> |
| [visualizations.py](https://github.com/L1ZLe/Quantative_Electricity/blob/master/visualizations.py)         | <code>► Generates interactive visualizations for data exploration.</code> |
| [trading_strategies.py](https://github.com/L1ZLe/Quantative_Electricity/blob/master/trading_strategies.py) | <code>► Implements various trading strategies based on predictions.</code> |
| [app.py](https://github.com/L1ZLe/Quantative_Electricity/blob/master/app.py)                               | <code>► Main entry point for running the Streamlit application.</code> |

</details>

---

### ⚙️ Installation

1. Clone the Quantative_Electricity repository:

```sh
git clone https://github.com/L1ZLe/Quantative_Electricity
```

2. Change to the project directory:

```sh
cd Quantative_Electricity
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

### 🤖 Running Quantative_Electricity

Use the following command to run Quantative_Electricity:

```sh
streamlit run app.py
```


---

## 🛠 Project Roadmap

- [X] `► Initial setup and model development`
- [X] `► Implementation of trading strategies`
- [X] `► Enhance user interface and visualizations`
- [X] `► Deploy and monitor application`

---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Join the Discussions](https://github.com/L1ZLe/Quantative_Electricity/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/L1ZLe/Quantative_Electricity/issues)**: Submit bugs found or log feature requests for Quantative_Electricity.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/L1ZLe/Quantative_Electricity
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

## 📄 License

This project is protected under the [MIT License](https://choosealicense.com/licenses/mit/). For more details, refer to the [LICENSE](LICENSE) file.

---

## 👏 Acknowledgments

- **[Pandas](https://pandas.pydata.org/)**: For providing the data manipulation tools.
- **[TensorFlow](https://www.tensorflow.org/)**: For enabling deep learning models.
- **[Streamlit](https://streamlit.io/)**: For allowing easy deployment of the web application.
- **[eia.gov](https://www.eia.gov/)**: For providing electricity data.

---

<p align="center">
  <em>Thank you for exploring Quantative_Electricity. We hope you find it valuable for your trading strategy development!</em>
</p>
