
# Train and evaluate a decision tree regressor on the msleep_clean dataset.
#
# This script demonstrates best practices for data preprocessing, model training,
# and evaluation using scikit-learn's DecisionTreeRegressor.
#
# Steps:
#   1. Load msleep_clean dataset from CSV.
#   2. Prepare features and target arrays.
#   3. Split data into training and testing sets.
#   4. Train DecisionTreeRegressor model.
#   5. Evaluate and print R^2 score.
#   6. Print tree structure as text.


# Import required libraries
import pandas as pd  # For data manipulation
from sklearn.tree import DecisionTreeRegressor, export_text  # For decision tree regressor and tree export
from sklearn.model_selection import train_test_split  # For splitting data


def main():
	"""
	Main function to load data, train decision tree regressor, and evaluate performance.
	Steps:
		1. Load msleep_clean dataset from CSV.
		2. Prepare features and target arrays.
		3. Split data into training and testing sets.
		4. Train DecisionTreeRegressor model.
		5. Evaluate and print R^2 score.
		6. Print tree structure as text.
	"""
	# Load the msleep_clean dataset
	sleep = pd.read_csv('msleep_clean.csv')

	# Prepare features (awake, brainwt, bodywt) and target (sleep_rem)
	X = sleep[['awake', 'brainwt', 'bodywt']]
	y = sleep['sleep_rem']

	# Split the data into training and testing sets (default 75% train, 25% test)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, random_state=42
	)

	# Initialize and train the decision tree regressor
	model = DecisionTreeRegressor(max_depth=3, ccp_alpha=0.02, random_state=123)
	model.fit(X_train, y_train)

	# Evaluate the model using R^2 score on the test set
	r_squared = model.score(X_test, y_test)
	print(f"R-squared on test set: {r_squared:.4f}")

	# Print the tree structure as text
	tree_text = export_text(model, feature_names=list(X.columns))
	print(tree_text)

if __name__ == "__main__":
	main()