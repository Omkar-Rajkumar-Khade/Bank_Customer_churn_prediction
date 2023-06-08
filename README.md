# Bank Customer Churn Prediction

This project aims to predict customer churn for ABC Multistate Bank. The dataset used for this project contains the following columns:

`customer_id`: Unused variable.

`credit_score`: Used as an input.

`country`: Used as an input.

`gender`: Used as an input.

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

`bank_churn_prediction.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.

`README.md`: Project documentation and instructions.

## Getting Started
To run the project, follow these steps:

```
Clone the repository: git clone https://github.com/your-username/bank-customer-churn-prediction.git
```
Install the required libraries: `pip install pandas numpy scikit-learn tensorflow matplotlib`
```
Open the Jupyter Notebook bank_churn_prediction.ipynb using Jupyter Notebook or any compatible environment.
```
```
Run the cells in the notebook to execute the code step by step.
```
Note: Make sure to update the file paths if necessary.

## Results and Discussion
The model achieves an accuracy of 0.8525 on the test data. 
This means that the model correctly predicts the customer churn status in approximately 85.25% of cases. 
The loss and accuracy curves plotted during model training can be examined to evaluate the model's performance and identify any overfitting or underfitting issues.

## Further Improvements
Here are a few suggestions for further improving the project:

Perform more in-depth exploratory data analysis to gain insights into the relationships between features and the target variable.
Try different preprocessing techniques, such as feature scaling or handling imbalanced classes, to see if they improve the model's performance.
Experiment with different neural network architectures, such as adding more hidden layers or adjusting the number of units in each layer.
Use more advanced optimization techniques, such as learning rate schedules or early stopping, to enhance the model's training process.
Explore other evaluation metrics, such as precision, recall, or F1 score, to get a more comprehensive understanding of the model's performance.

## Conclusion
In this project, we successfully built an Artificial Neural Network (ANN) to predict customer churn for ABC Multistate Bank. By training the model on the provided dataset and evaluating its performance, we achieved an accuracy of 0.8525 on the test data. The results demonstrate the potential of using machine learning techniques for customer churn prediction and provide insights for the bank to take proactive actions to retain customers.

For more details and a step-by-step explanation of the project, please refer to the Jupyter Notebook bank_churn_prediction.ipynb.

