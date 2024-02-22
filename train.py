import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data
df = pd.read_csv("expensedata.csv")  
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate expenses by date
expense_data = df.groupby('Date')['ExpenseAmount'].sum()
print(expense_data)

# Fit the SARIMA model
sarima_model = SARIMAX(expense_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fit_model = sarima_model.fit()

# Save the model
joblib.dump(fit_model, "sarima_model.pkl")