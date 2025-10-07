
# Train and evaluate a multilayer perceptron regressor on a sample of the diamonds dataset.
#
# This script demonstrates best practices for model training and evaluation using scikit-learn's MLPRegressor.
#
# Steps:
#   1. Load and sample diamonds dataset from CSV.
#   2. Prepare features and target arrays.
#   3. Split data into training and testing sets.
#   4. Train MLPRegressor model.
#   5. Output predictions and actual values.
#   6. Evaluate and print R^2 scores for train and test sets.


# Import required libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.neural_network import MLPRegressor  # For multilayer perceptron regressor


def main():
	"""
	Main function to load data, train MLP regressor, and evaluate performance.
	Steps:
		1. Load and sample diamonds dataset from CSV.
		2. Prepare features and target arrays.
		3. Split data into training and testing sets.
		4. Train MLPRegressor model.
		5. Output predictions and actual values.
		6. Evaluate and print R^2 scores for train and test sets.
	"""
	# Load and sample the dataset
	diamonds = pd.read_csv('diamonds.csv').sample(n=800, random_state=10)

	# Prepare features (drop categorical and target columns) and target
	X = diamonds.drop(['cut', 'color', 'clarity', 'price'], axis=1)
	y = diamonds['price']

	# Split the data into training and testing sets (70% train, 30% test)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=123
	)

	# Initialize and train the MLP regressor
	mlp = MLPRegressor(
		hidden_layer_sizes=(50, 50),
		activation='identity',
		max_iter=500,
		random_state=123
	)
	mlp.fit(X_train, y_train)

	# Output predictions for first 5 training samples
	print("Price predictions:", mlp.predict(X_train[:5]))
	print("Actual prices:\n", y_train.iloc[:5].to_frame().head(5))
	# Print R^2 scores for training and testing data
	print(f"Score for the training data: {mlp.score(X_train, y_train):.4f}")
	print(f"Score for the testing data: {mlp.score(X_test, y_test):.4f}")

if __name__ == "__main__":
	main()