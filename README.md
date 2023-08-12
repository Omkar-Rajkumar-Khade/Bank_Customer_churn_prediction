# Bank Customer Churn Prediction

This project aims to predict customer churn for ABC Multistate Bank. The dataset used for this project contains various features that may influence customer churn, such as credit score, age, tenure, balance, product usage, credit card status, active membership, estimated salary, and more. The target variable, "churn," indicates whether a customer has left the bank during a specific period (1 if churned, 0 if not).


## Getting Started
To run the project, follow these steps:

1) Clone the repository:
```
git clone https://github.com/your-username/bank-customer-churn-prediction.git
```
2) Install the required libraries: 
```
pip install pandas numpy scikit-learn tensorflow matplotlib streamlit
```
3) Open the Jupyter Notebook bank_churn_prediction.ipynb using Jupyter Notebook or any compatible environment.

4) Open the terminal or command prompt and navigate to the repository directory.

5) Run the Streamlit app: `streamlit run streamlit_app.py`

6) The app will open in your default web browser, allowing you to input feature values and see churn predictions.

Note: Please update the file paths if necessary and ensure that the required libraries are installed.

### Dataset
The dataset used for this project contains the following columns:
Dataset Download Link : https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

`customer_id`: Unused variable.

`credit_score`: Used as an input.

`country`: Unused variable.

`gender`: Unused variable.

`age`: Used as an input.

`tenure`: Used as an input.

`balance`: Used as an input.

`products_number` : Used as an input.

`credit_card`: Used as an input.

`active_member`: Used as an input.

`estimated_salary`: Used as an input.

`churn`: Target variable. 1 if the client has left the bank during some period or 0 if he/she has not.

## Project Overview
1. The project starts with loading the dataset into a pandas DataFrame.
2. Initial data exploration is performed by checking the DataFrame information and looking for duplicated rows.
3. The unnecessary column, customer_id, is dropped from the DataFrame.
4. Categorical variables (country and gender) are encoded using one-hot encoding with the pd.get_dummies() function.
5. The features (X) and the target variable (y) are separated in the DataFrame.
6. The data is split into training and testing sets using the train_test_split() function from scikit-learn.
7. The numerical features are standardized using the StandardScaler from scikit-learn.
8. A sequential model is created using TensorFlow and Keras.
9. The model consists of a hidden layer with 11 units and a ReLU activation function.
Another hidden layer with 11 units and a ReLU activation function is added.
An output layer with 1 unit and a sigmoid activation function is added.
10. The model is compiled with binary cross-entropy loss and Adam optimizer.
11. The model is trained using the scaled training data (X_trained_scaled) and training labels (y_train) for 100 epochs.
20% of the training dataset is used as validation data for verification purposes.
12. The weights of the hidden layers and the output layer are extracted and printed.
13. Predictions are made on the test data (X_test_scaled) using the trained model.
As the output is passed through a sigmoid function, the predictions are probabilities.
The predicted probabilities are converted to binary predictions (0 or 1) by using a threshold of 0.5.
14. The accuracy of the model is evaluated using the accuracy_score function from scikit-learn by comparing the predictions (y_pred) with the actual labels (y_test).
15. Loss and accuracy curves are plotted using the history object from model training.

## Repository Files
The repository contains the following files:

`dataset` folder contains the Bank Customer Churn Prediction.csv dataset used in the project.

`app.py` is the streamlit application file that defines the API endpoints and loads the saved model.

`models` is folder that contain the serialized machine learning models that is used for prediction.

`customer-churn-prediction-using-ann.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.

`customer_churn_prediction_using_ann 2.ipynb`: Model trained by this jupyter notebook is used in application 

`README.md`: Project documentation and instructions.



## Results and Discussion
The model achieves an accuracy of 0.8525 on the test data. 
This means that the model correctly predicts the customer churn status in approximately 85.25% of cases. 
The loss and accuracy curves plotted during model training can be examined to evaluate the model's performance and identify any overfitting or underfitting issues. The Streamlit app provides users with a simple interface to predict customer churn based on specific feature inputs. 

## Conclusion
In this project, we successfully built an Artificial Neural Network (ANN) to predict customer churn for ABC Multistate Bank. By training the model on the provided dataset and evaluating its performance, we achieved an accuracy of 0.8525 on the test data. The results demonstrate the potential of using machine learning techniques for customer churn prediction and provide insights for the bank to take proactive actions to retain customers. we have developed a Streamlit app that predicts customer churn for ABC Multistate Bank based on specific feature inputs

For more details and a step-by-step explanation of the project, please refer to the Jupyter Notebook customer-churn-prediction-using-ann.ipynb.

