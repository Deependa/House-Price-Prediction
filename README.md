# House Price Prediction - Machine Learning Model

 ## 1.Overview

This project aims to predict house prices using machine learning. The pipeline includes data preprocessing, model training, evaluation, and deployment using Flask as a REST API. Model training is conducted in Google Colab, while deployment is done locally or on a cloud service.




## 2. Features

### Data Preprocessing: Handling missing values, feature engineering, scaling, and encoding.

### Model Training: Regression models like Linear Regression, Random Forest, and XGBoost.

### Hyperparameter Optimization: Using GridSearchCV or RandomizedSearchCV.

### API Deployment: Serving predictions using Flask.

### Frontend: A simple HTML interface for user input and displaying predictions.



## 3. Dataset

The project uses the California Housing Dataset from Scikit-learn. Ensure you have the dataset loaded in Google Colab before proceeding with training.



## 4. Installation

### Clone the repository:

git clone <repo_link>
cd house-price-prediction

### Install dependencies:

pip install -r requirements.txt

### Model Training (Google Colab)

Open Google Colab and upload the notebook (house_price_train.ipynb).

### Run all cells to preprocess data and train the model.

Save the trained model as a .pkl file:

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

Download model.pkl and place it in your project directory.

## Running the Flask API

Ensure you have the trained model (model.pkl) and preprocessor (preprocessor.pkl) in the project directory.

## Start the Flask server:

python app.py

API Usage

Endpoint: /predict

Method: POST

## Input (JSON format):

{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}

## Response:

{
  "predicted_price": 256000.0
}

## Frontend Web Interface

Open index.html in a browser.

### Enter house details and submit to get the predicted price.

### Deployment (Optional)

For cloud deployment, use services like Render, AWS, or GCP.

## Files & Directories

### house_price_train.ipynb – Google Colab notebook for model training.

### app.py – Flask API for serving predictions.

### model.pkl – Trained model file.

### preprocessor.pkl – Data preprocessing pipeline.

### index.html – Simple web interface.

### requirements.txt – Required Python packages.

### README.md – Project documentation.

## Future Enhancements

Implement logging and error handling.


## Author

Deependar Singh


