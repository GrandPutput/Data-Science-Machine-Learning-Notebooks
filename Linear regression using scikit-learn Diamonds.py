diamonds = pd.read_csv('diamonds.csv')

# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.linear_model import LinearRegression  # For linear regression model
import warnings  # For suppressing warnings

warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

def load_data(filepath: str) -> pd.DataFrame:
	"""
	Load the diamonds dataset from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Loaded DataFrame.
	"""
	return pd.read_csv(filepath)

def get_user_input() -> list:
	"""
	Prompt user for carat and table values.
	Returns:
		list: List containing carat and table as floats.
	"""
	while True:
		try:
			carat = float(input("Enter carat value: "))
			table = float(input("Enter table value: "))
			return [carat, table]
		except ValueError:
			print("Invalid input. Please enter numeric values.")

def train_and_predict(X, y, Xnew):
	"""
	Train LinearRegression and predict price for new input.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		Xnew (list): New data for prediction.
	"""
	# Initialize and train the linear regression model
	model = LinearRegression()
	model.fit(X, y)
	# Print model intercept
	intercept = model.intercept_
	print('Intercept is', round(intercept, 3))
	# Print model coefficients
	coefficients = model.coef_
	print('Weights for carat and table features are', np.round(coefficients, 3))
	# Predict price for new input
	prediction = model.predict([Xnew])
	print('Predicted price is', np.round(prediction, 2))

def main():
	"""
	Main function to load data, get user input, train model, and predict price.
	Steps:
		1. Load diamonds dataset.
		2. Prompt user for carat and table values.
		3. Train linear regression model and print weights.
		4. Predict price for user input.
	"""
	# Load the diamonds dataset
	diamonds = load_data('diamonds.csv')
	# Prepare features and target
	X = diamonds[['carat', 'table']]
	y = diamonds['price']
	# Get user input for prediction
	Xnew = get_user_input()
	# Train model and predict
	train_and_predict(X, y, Xnew)

# Entry point for script execution
if __name__ == "__main__":
	main()