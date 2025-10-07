
# Import required libraries
import os  # For file existence checking
import sys  # For exiting the script on error
import numpy as np  # For numerical operations (not directly used here, but often useful)
import pandas as pd  # For data manipulation
from sklearn import metrics  # For evaluation metrics
from sklearn.linear_model import LogisticRegression  # For logistic regression model


def load_data(filepath: str, random_state: int, n: int = 500) -> pd.DataFrame:
	"""
	Load and sample NBA data from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
		random_state (int): Random seed for reproducibility.
		n (int): Number of samples to draw.
	Returns:
		pd.DataFrame: Sampled DataFrame.
	"""
	# Check if the file exists before loading
	if not os.path.exists(filepath):
		print(f"Error: File '{filepath}' not found.")
		sys.exit(1)
	df = pd.read_csv(filepath)
	# If requested sample size is larger than available data, use full dataset
	if n > len(df):
		print(f"Warning: Requested sample size {n} is greater than data size {len(df)}. Using full dataset.")
		n = len(df)
	# Randomly sample n rows from the DataFrame
	return df.sample(n=n, random_state=random_state)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""
	Preprocess NBA data for classification.
	Converts the 'game_result' column to a binary 'win' column.
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		tuple: Features (X) and target (y).
	"""
	df = df.copy()
	# Convert 'game_result' to binary: 1 for win ('W'), 0 for loss ('L')
	df['win'] = df['game_result'].replace({'L': 0, 'W': 1})
	# Use 'elo_i' as the feature
	X = df[['elo_i']]
	# Use the new 'win' column as the target
	y = df['win']
	return X, y


def evaluate_model(y_true, y_pred) -> None:
	"""
	Print classification metrics for model evaluation.
	Args:
		y_true: True labels.
		y_pred: Predicted labels.
	"""
	# Compute and print the confusion matrix
	conf_matrix = metrics.confusion_matrix(y_true, y_pred)
	print("Confusion matrix:\n", conf_matrix)
	# Compute and print accuracy
	accuracy = metrics.accuracy_score(y_true, y_pred)
	print("Accuracy:", round(accuracy, 3))
	# Compute and print precision
	precision = metrics.precision_score(y_true, y_pred)
	print("Precision:", round(precision, 3))
	# Compute and print recall
	recall = metrics.recall_score(y_true, y_pred)
	print("Recall:", round(recall, 3))
	# Compute and print Cohen's kappa
	kappa = metrics.cohen_kappa_score(y_true, y_pred)
	print("Kappa:", round(kappa, 3))


def main():
	"""
	Main function to run logistic regression and print metrics.
	Steps:
		1. Get random state from user input.
		2. Load and sample NBA data.
		3. Preprocess data for classification.
		4. Train logistic regression model.
		5. Predict and evaluate model performance.
	"""
	# Prompt user for random state input
	try:
		rand = int(input("Enter random state (integer): "))
	except ValueError:
		print("Invalid input. Please enter an integer for random state.")
		sys.exit(1)

	data_path = "nbaallelo_log.csv"  # Path to the NBA data CSV file
	# Load and sample the data
	nba_df = load_data(data_path, random_state=rand)
	# Preprocess the data to get features and target
	X, y = preprocess_data(nba_df)

	# Initialize and train the logistic regression model
	model = LogisticRegression()
	model.fit(X, y)
	# Predict the target using the trained model
	y_pred = model.predict(X)
	# Evaluate and print model performance metrics
	evaluate_model(y, y_pred)


# Entry point for script execution
if __name__ == "__main__":
	main()