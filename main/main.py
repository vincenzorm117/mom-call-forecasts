import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

# create a dataframe with the number of calls you made to your mother in each month
# (assuming you have data for the past 12 months)
data = {'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'calls': [5, 8, 7, 10, 15, 12, 11, 14, 18, 20, 17, 22]}
df = pd.DataFrame(data)

# fit the ARIMA model
model = ARIMA(df['calls'], order=(1, 1, 1))
model_fit = model.fit(disp=False)

# forecast the next 3 months
forecast, stderr, conf_int = model_fit.forecast(steps=3)
print("Forecasted number of calls: ",forecast)
