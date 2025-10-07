
# Train and evaluate a logistic regression model on the NBA dataset.
#
# This script demonstrates best practices for model training and evaluation using scikit-learn's LogisticRegression.
#
# Steps:
#   1. Load NBA dataset from CSV.
#   2. Create binary target variable ('win').
#   3. Prepare features and target arrays.
#   4. Train logistic regression model.
#   5. Output model coefficients and intercept.
#   6. Evaluate and print accuracy.


# Import required libraries
import pandas as pd  # For data manipulation
from sklearn.linear_model import LogisticRegression  # For logistic regression model


def main():
	"""
	Main function to load data, train logistic regression, and evaluate accuracy.
	Steps:
		1. Load NBA dataset from CSV.
		2. Create binary target variable ('win').
		3. Prepare features and target arrays.
		4. Train logistic regression model.
		5. Output model coefficients and intercept.
		6. Evaluate and print accuracy.
	"""
	# Load the dataset
	nba = pd.read_csv('nbaallelo_log.csv')

	# Create binary target: 1 if win ('W'), 0 if loss ('L')
	nba['win'] = (nba['game_result'] == 'W').astype(int)

	# Prepare features (elo_i) and target (win)
	X = nba[['elo_i']]
	y = nba['win']

	# Initialize and train the logistic regression model
	model = LogisticRegression(penalty='l2', solver='lbfgs')
	model.fit(X, y)

	# Output model coefficients and intercept
	print('w1 (coefficient for elo_i):', model.coef_)
	print('w0 (intercept):', model.intercept_)

	# Evaluate the model using accuracy
	score = model.score(X, y)
	print(f"Accuracy: {score:.3f}")

if __name__ == "__main__":
	main()