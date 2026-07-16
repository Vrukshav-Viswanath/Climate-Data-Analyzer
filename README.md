# Climate Data Analyzer

A machine learning project that predicts next-day average temperature for Basel, Switzerland using linear regression. Trained on historical weather data (humidity, pressure, cloud cover, and solar radiation), it evaluates performance with Mean Absolute Error and visualizes predicted vs. actual temperatures with Python, pandas, and scikit-learn.

## How it works

The notebook (`Climate_Data_Analyzer.ipynb`):

1. Loads a year of Basel weather data
2. Trains a `LinearRegression` model on humidity, pressure, cloud cover, and global radiation to predict mean temperature
3. Evaluates performance with Mean Absolute Error (MAE)
4. Plots predicted vs. actual temperature
5. Runs a sanity check by predicting on a real held-out row

## Dataset

This project uses the [Weather Prediction Dataset](https://www.kaggle.com/datasets/thedevastator/weather-prediction), which includes daily weather features for several European cities, including Basel.

To run this notebook:

1. Download the dataset from the link above
2. Follow the directions below to extract one year of data
3. Place the CSV in `data/one_year_weather_dataset.csv`

Expected columns include `BASEL_humidity`, `BASEL_pressure`, `BASEL_cloud_cover`, `BASEL_global_radiation`, and `BASEL_temp_mean`.

### Extracting one year of data

The full Kaggle download spans multiple years, so it needs to be trimmed down before use. The dataset has a `DATE` column formatted as `YYYYMMDD`. Filter it down to a single year like this:

```python
import pandas as pd

df = pd.read_csv("full_weather_dataset.csv")

# Keep only one calendar year (adjust the year as needed)
one_year = df[(df["DATE"] >= 20000101) & (df["DATE"] <= 20001231)]

one_year.to_csv("data/one_year_weather_dataset.csv", index=False)
```

Check the actual `DATE` range in your download first (`df["DATE"].min()`, `df["DATE"].max()`) and pick a year that's fully covered.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Climate_Data_Analyzer.ipynb
```

## Tech stack

- Python
- pandas
- scikit-learn
- matplotlib

## Getting started

```bash
git clone https://github.com/Vrukshav-Viswanath/Climate-Data-Analyzer.git
cd Climate-Data-Analyzer
```

## Results

The model reports Mean Absolute Error (MAE) in °C, along with a scatter plot comparing predicted vs. actual temperatures for the held-out test set.
