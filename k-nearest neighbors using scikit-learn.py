skySurvey = pd.read_csv('SDSS.csv')
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']
y = skySurvey['class']
skySurveyKnn = KNeighborsClassifier(3)
skySurveyKnn.fit(X_train, y_train)
y_pred = skySurveyKnn.predict(X_test)

# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.neighbors import KNeighborsClassifier  # For k-nearest neighbors classification
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import accuracy_score  # For accuracy metric

def load_data(filepath: str) -> pd.DataFrame:
	"""
	Load the sky survey dataset from a CSV file.
	Args:
		filepath (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Loaded DataFrame.
	"""
	return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
	"""
	Add 'u_g' feature and prepare features and target.
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		tuple: Features (X) and target (y).
	"""
	df = df.copy()
	# Create new feature 'u_g' as the difference between 'u' and 'g'
	df['u_g'] = df['u'] - df['g']
	X = df[['redshift', 'u_g']]
	y = df['class']
	return X, y

def train_and_evaluate(X, y, n_neighbors=3, test_size=0.3, random_seed=42):
	"""
	Train KNeighborsClassifier and evaluate accuracy.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		n_neighbors (int): Number of neighbors for KNN.
		test_size (float): Proportion of test set.
		random_seed (int): Random seed for reproducibility.
	"""
	# Set random seed for reproducibility
	np.random.seed(random_seed)
	# Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	# Initialize and train the KNN classifier
	knn = KNeighborsClassifier(n_neighbors)
	knn.fit(X_train, y_train)
	# Predict on test set
	y_pred = knn.predict(X_test)
	# Calculate accuracy
	score = accuracy_score(y_test, y_pred)
	print('Accuracy score is %.3f' % score)

def main():
	"""
	Main function to load data, preprocess, train model, and print accuracy.
	Steps:
		1. Load sky survey dataset.
		2. Preprocess data (add 'u_g', select features and target).
		3. Train KNN classifier and evaluate accuracy.
	"""
	# Load the sky survey dataset
	skySurvey = load_data('SDSS.csv')
	# Preprocess data
	X, y = preprocess_data(skySurvey)
	# Train model and evaluate
	train_and_evaluate(X, y, n_neighbors=3, test_size=0.3, random_seed=42)

# Entry point for script execution
if __name__ == "__main__":
	main()