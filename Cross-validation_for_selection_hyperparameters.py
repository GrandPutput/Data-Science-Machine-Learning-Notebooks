diamonds = pd.read_csv('diamonds.csv')

# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import ElasticNet  # For ElasticNet regression

def load_data(filepath: str) -> pd.DataFrame:
	"""
	Load the diamonds dataset from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Loaded DataFrame.
	"""
	return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
	"""
	Prepare features and target for modeling.
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		tuple: Scaled train/test features and targets.
	"""
	# Select features and target
	X = df[['carat', 'depth']]
	y = df['price']
	# Split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Standardize features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	return X_train_scaled, X_test_scaled, y_train, y_test

def tune_hyperparameters(X_train, y_train):
	"""
	Perform grid search cross-validation to tune ElasticNet alpha.
	Args:
		X_train: Scaled training features.
		y_train: Training target.
	Returns:
		GridSearchCV: Fitted grid search object.
	"""
	# Define ElasticNet model
	model = ElasticNet(random_state=0)
	# Define parameter grid for alpha
	param_grid = {'alpha': [0.1, 0.5, 0.9, 1.0]}
	# Set up grid search with 10-fold cross-validation
	grid_search = GridSearchCV(model, param_grid, cv=10)
	grid_search.fit(X_train, y_train)
	return grid_search

def main():
	"""
	Main function to load data, preprocess, tune hyperparameters, and print results.
	Steps:
		1. Load diamonds dataset.
		2. Preprocess data (split and scale).
		3. Tune ElasticNet alpha using cross-validation.
		4. Print mean test scores and best estimator.
	"""
	# Load the diamonds dataset
	diamonds = load_data('diamonds.csv')
	# Preprocess data
	X_train, X_test, y_train, y_test = preprocess_data(diamonds)
	# Tune hyperparameters
	grid_search = tune_hyperparameters(X_train, y_train)
	# Print results
	print('Mean testing scores:', grid_search.cv_results_['mean_test_score'])
	print('Best estimator:', grid_search.best_estimator_)

# Entry point for script execution
if __name__ == "__main__":
	main()