import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('model_deployment/Telco-Customer-Churn.csv')

### drop the na values

df.dropna()

df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

df = pd.get_dummies(df[['tenure','MonthlyCharges','Contract','Churn']],drop_first=True)

X = df.drop('Churn',axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print('Accuracy:', round(accuracy_score(y_test, model.predict(X_test)),2))

#### Save Model ##pickle - Flask

joblib.dump(model,'churn_model.pkl')