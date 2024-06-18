import ast
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('creditcard.csv')

# Balance the data
normal = data[data['Class'] == 0].sample(500, random_state=42)
fraud  = data[data['Class'] == 1]
df = pd.concat([normal, fraud], axis=0, ignore_index=True)

# Separate Independent and Dependent
X = df.iloc[:, 0:-1]
y = df['Class']

# Splitted in Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building and Prediction
model = LogisticRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print(f' Train Accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f' Test Accuracy: {accuracy_score(y_test, y_test_pred)}')


# Streamlit Web-App

st.title('Credit Card Fraud Detection')
input_data = st.text_input('Enter the banking parameters to check the transaction is Fraud or Not:')
submit = st.button('Submit')


if submit:
      
      data = ast.literal_eval(input_data)
      data = np.array(data)
      sample = data.reshape(1, -1)
      pred = model.predict(sample)[0]

      if pred == 0:
            st.write('Normal Transaction')
      else:
            st.write('Fraud Transaction')
