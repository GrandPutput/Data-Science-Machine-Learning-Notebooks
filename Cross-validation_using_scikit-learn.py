taxis = pd.read_csv('taxis.csv')
y = taxis['fare']

# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split, cross_validate  # For splitting data and cross-validation
from sklearn.linear_model import LinearRegression  # For linear regression model

def load_data(filepath: str) -> pd.DataFrame:
	"""
	Load the taxis dataset from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Loaded DataFrame.
	"""
	return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
	"""
	Prepare features and target for modeling, and split into train, validation, and test sets.
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		tuple: Train, validation, and test features and targets.
	"""
	# Select features and target
	X = df[['passengers', 'distance']]
	y = df['fare']
	# Split into train and test sets (10% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	# Further split train into train and validation sets (2/9 validation)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2/9, random_state=42)
	return X_train, X_val, X_test, y_train, y_val, y_test

def perform_cross_validation(X_train, y_train):
	"""
	Perform cross-validation using Linear Regression and explained variance score.
	Args:
		X_train: Training features.
		y_train: Training target.
	Returns:
		dict: Cross-validation results.
	"""
	model = LinearRegression()
	# Perform 15-fold cross-validation
	cv_results = cross_validate(model, X_train, y_train, cv=15, scoring='explained_variance', return_train_score=False)
	return cv_results

def main():
	"""
	Main function to load data, preprocess, perform cross-validation, and print results.
	Steps:
		1. Load taxis dataset.
		2. Preprocess data (split into train, validation, test).
		3. Perform cross-validation on training set.
		4. Print test scores from cross-validation.
	"""
	# Load the taxis dataset
	taxis = load_data('taxis.csv')
	# Preprocess data
	X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(taxis)
	# Perform cross-validation
	cv_results = perform_cross_validation(X_train, y_train)
	# Print test scores from cross-validation
	print("Test score:", cv_results['test_score'])

# Entry point for script execution
if __name__ == "__main__":
	main()