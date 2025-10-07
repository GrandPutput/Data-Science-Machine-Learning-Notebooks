diamonds = pd.read_csv('diamonds.csv')
prediction = knn.predict(Xnew)
neighbors = knn.kneighbors(Xnew)

# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.neighbors import KNeighborsRegressor  # For k-nearest neighbors regression
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

def train_and_predict(X, y, Xnew, n_neighbors=12):
	"""
	Train KNeighborsRegressor and predict price for new input.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		Xnew (list): New data for prediction.
		n_neighbors (int): Number of neighbors to use.
	"""
	# Initialize and train the KNN regressor
	knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean')
	knn.fit(X, y)
	# Predict price for new input
	prediction = knn.predict([Xnew])
	print('Predicted price is', np.round(prediction, 2))
	# Get distances and indices of nearest neighbors
	neighbors = knn.kneighbors([Xnew])
	print('Distances and indices of the 12 nearest neighbors are', neighbors)

def main():
	"""
	Main function to load data, get user input, train model, and predict price.
	Steps:
		1. Load diamonds dataset.
		2. Prompt user for carat and table values.
		3. Train KNN regressor and predict price.
		4. Print prediction and neighbor info.
	"""
	# Load the diamonds dataset
	diamonds = load_data('diamonds.csv')
	# Prepare features and target
	X = diamonds[['carat', 'table']]
	y = diamonds['price']
	# Get user input for prediction
	Xnew = get_user_input()
	# Train model and predict
	train_and_predict(X, y, Xnew, n_neighbors=12)

# Entry point for script execution
if __name__ == "__main__":
	main()
