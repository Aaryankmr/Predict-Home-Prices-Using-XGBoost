import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load your data (replace with your actual dataset)
data = pd.read_csv('Housing.csv')

# 2. Check your columns
# Assume data has: 'income', 'school_rating', 'num_hospitals_nearby', 'crime_rate', 'price'
features = ['income', 'school_rating', 'num_hospitals_nearby', 'crime_rate']
target = 'price'

X = data[features]
y = data[target]

# 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
