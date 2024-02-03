from IPython.display import clear_output
%pip install gdown==4.5


clear_output()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data_df = pd.read_csv('bikers_data.csv')
data_y = data_df['Number of bikers'] # target
data_x = data_df.drop(['Number of bikers'], axis=1) # input features
data_df = data_df.drop(['Date', 'Unnamed: 0'], axis=1)
X = data_df.drop('Fremont Bridge Total', axis=1)
y = data_df['Fremont Bridge Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)
