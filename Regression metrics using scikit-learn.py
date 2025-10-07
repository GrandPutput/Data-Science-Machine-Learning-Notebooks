
# Train and evaluate a linear regression model on a sample of the diamonds dataset.
#
# This script demonstrates best practices for model training and evaluation using scikit-learn.
#
# Steps:
#   1. Prompt user for random seed for sampling.
#   2. Load and sample diamonds dataset from CSV.
#   3. Prepare features and target arrays.
#   4. Train LinearRegression model.
#   5. Predict and evaluate using MAE, MSE, RMSE, and R-squared.


# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn import metrics  # For regression metrics
from sklearn.linear_model import LinearRegression  # For linear regression model


def main():
	"""
	Main function to prompt for random seed, train linear regression, and evaluate metrics.
	Steps:
		1. Prompt user for random seed for sampling.
		2. Load and sample diamonds dataset from CSV.
		3. Prepare features and target arrays.
		4. Train LinearRegression model.
		5. Predict and evaluate using MAE, MSE, RMSE, and R-squared.
	"""
	# Prompt user for random seed for reproducible sampling
	rand = int(input("Enter random seed for sampling: "))

	# Load and sample the diamonds dataset
	diamonds = pd.read_csv('diamonds.csv').sample(n=500, random_state=rand)

	# Prepare features (carat, table) and target (price)
	X = diamonds[['carat', 'table']]
	y = diamonds['price']

	# Initialize and train the linear regression model
	model = LinearRegression()
	model.fit(X, y)

	# Predict on the training data
	y_pred = model.predict(X)

	# Evaluate using various regression metrics
	mae = metrics.mean_absolute_error(y, y_pred)
	print(f"MAE: {mae:.3f}")

	mse = metrics.mean_squared_error(y, y_pred)
	print(f"MSE: {mse:.3f}")

	rmse = np.sqrt(mse)
	print(f"RMSE: {rmse:.3f}")

	r2 = metrics.r2_score(y, y_pred)
	print(f"R-squared: {r2:.3f}")

if __name__ == "__main__":
	main()