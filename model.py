# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load data
data = pd.read_csv("C:\documents\DevOpsdemo\endtoenddemo\AWS-CICD-Deploymennt-main\carprices.csv")

# Define features and target variable
features = ["Mileage", "Age(yrs)"]
target = "Sell Price($)"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Create and train Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save model to pickle file
pickle.dump(model, open("car_price_model.pkl", "wb"))

# Load model from pickle file
loaded_model = pickle.load(open("car_price_model.pkl", "rb"))

# **Define a sample new car data dictionary **
# You can replace this with any dictionary containing
# "Mileage" and "Age(yrs)" keys with corresponding values
new_car_data = {"Mileage": 50000, "Age(yrs)": 3}

# **Extract features from the dictionary into a list**
new_car_features = [new_car_data["Mileage"], new_car_data["Age(yrs)"]]

# **Predict price for the new car**
new_car_price = loaded_model.predict([new_car_features])[0]

print(f"Predicted price for new car: ${new_car_price:.2f}")
