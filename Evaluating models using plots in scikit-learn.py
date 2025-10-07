diamonds = pd.read_csv('diamonds.csv').sample(n=50, random_state=42)

# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.inspection import PartialDependenceDisplay  # For partial dependence plots
from sklearn import metrics  # For evaluation metrics

def load_data(filepath: str, n: int = 50, random_state: int = 42) -> pd.DataFrame:
	"""
	Load and sample the diamonds dataset from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
		n (int): Number of samples to draw.
		random_state (int): Random seed for reproducibility.
	Returns:
		pd.DataFrame: Sampled DataFrame.
	"""
	return pd.read_csv(filepath).sample(n=n, random_state=random_state)

def get_features(df: pd.DataFrame) -> list:
	"""
	Prompt user to input two feature names and validate them.
	Args:
		df (pd.DataFrame): DataFrame to check feature names against.
	Returns:
		list: List of two valid feature names.
	"""
	print(f"Available features: {list(df.columns)}")
	while True:
		feature1 = input("Enter first feature: ").strip()
		feature2 = input("Enter second feature: ").strip()
		if feature1 in df.columns and feature2 in df.columns and feature1 != feature2:
			return [feature1, feature2]
		print("Invalid feature names or duplicate. Please try again.")

def train_and_evaluate(X, y, feature_names):
	"""
	Train linear regression, plot prediction error and partial dependence, print MAE.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		feature_names (list): Names of the features used.
	"""
	# Train linear regression model
	model = LinearRegression()
	model.fit(X, y)
	y_pred = model.predict(X)
	pred_error = y - y_pred

	# Plot prediction error
	plt.figure()
	plt.scatter(y_pred, pred_error)
	plt.xlabel('Predicted')
	plt.ylabel('Prediction error')
	plt.axhline(0, color='gray', linestyle='dashed')
	plt.title('Prediction Error Plot')
	plt.savefig('predictionError.png')
	plt.close()

	# Plot partial dependence
	PartialDependenceDisplay.from_estimator(model, X, features=[0, 1], feature_names=feature_names)
	plt.title('Partial Dependence Plot')
	plt.savefig('partial_dependence.png')
	plt.close()

	# Compute and print mean absolute error
	mae = metrics.mean_absolute_error(y, y_pred)
	print("MAE:", round(mae, 3))

def main():
	"""
	Main function to load data, get features, train model, plot, and evaluate.
	Steps:
		1. Load and sample diamonds dataset.
		2. Prompt user for two feature names.
		3. Train linear regression model.
		4. Plot prediction error and partial dependence.
		5. Print mean absolute error (MAE).
	"""
	# Load and sample the diamonds dataset
	diamonds = load_data('diamonds.csv', n=50, random_state=42)
	# Prompt user for two valid feature names
	features = get_features(diamonds)
	# Prepare feature matrix and target vector
	X = diamonds[features]
	y = diamonds['price']
	# Train model, plot, and evaluate
	train_and_evaluate(X, y, features)

# Entry point for script execution
if __name__ == "__main__":
	main()