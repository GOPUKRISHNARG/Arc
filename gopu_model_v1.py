#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


# In[4]:


# Load the dataset
data_path = 'wdbc.data'

# Read the data into a DataFrame
df = pd.read_csv(data_path, names=column_names)

# Display the first few rows of the dataset
df.head()


# In[5]:


# Drop the ID column
df.drop(columns=['ID'], inplace=True)

# Encode the target variable (M = 1, B = 0)
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# In[7]:


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


# In[ ]:





# In[ ]:




