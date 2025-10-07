taxis = pd.read_csv("taxis.csv")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split, cross_validate, KFold  # For splitting and cross-validation
from sklearn.neighbors import KNeighborsRegressor  # For k-nearest neighbors regression
from sklearn.linear_model import LinearRegression  # For linear regression

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
	Prepare features and target, and split into train, validation, and test sets.
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		tuple: Train, validation, and test features and targets.
	"""
	X = df[['distance']]
	y = df['fare']
	# Split into train+val and test sets (10% test)
	X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	# Further split trainval into train and validation sets (0.1111*0.9 ≈ 0.1 validation)
	X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1111, random_state=42)
	return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_models(X_train, y_train, n_neighbors=3, n_splits=10, random_seed=42):
	"""
	Evaluate KNN and Linear Regression models using cross-validation.
	Args:
		X_train (pd.DataFrame): Training features.
		y_train (pd.Series): Training target.
		n_neighbors (int): Number of neighbors for KNN.
		n_splits (int): Number of folds for KFold.
		random_seed (int): Random seed for reproducibility.
	"""
	# Initialize models
	knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
	slr_model = LinearRegression()
	# Set up KFold cross-validation
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
	# Cross-validate KNN
	knn_results = cross_validate(knn_model, X_train, y_train, cv=kf, return_train_score=False)
	knn_scores = knn_results['test_score']
	print('k-nearest neighbor scores:', knn_scores.round(3))
	print('Mean:', knn_scores.mean().round(3))
	print('SD:', knn_scores.std().round(3))
	# Cross-validate Linear Regression
	slr_results = cross_validate(slr_model, X_train, y_train, cv=kf, return_train_score=False)
	slr_scores = slr_results['test_score']
	print('Simple linear regression scores:', slr_scores.round(3))
	print('Mean:', slr_scores.mean().round(3))
	print('SD:', slr_scores.std().round(3))

def main():
	"""
	Main function to load data, preprocess, evaluate models, and print results.
	Steps:
		1. Load taxis dataset.
		2. Preprocess data (split into train, validation, test).
		3. Evaluate KNN and Linear Regression models using cross-validation.
		4. Print cross-validation scores, means, and standard deviations.
	"""
	# Load the taxis dataset
	taxis = load_data("taxis.csv")
	# Preprocess data
	X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(taxis)
	# Evaluate models
	evaluate_models(X_train, y_train, n_neighbors=3, n_splits=10, random_seed=42)

# Entry point for script execution
if __name__ == "__main__":
	main()