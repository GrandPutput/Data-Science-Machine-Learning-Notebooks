
# Train and evaluate a Naive Bayes classifier on the SDSS dataset.
#
# This script demonstrates best practices for feature engineering, model training,
# and evaluation using scikit-learn's GaussianNB.
#
# Steps:
#   1. Load SDSS dataset from CSV.
#   2. Feature engineering: create 'u_g' as difference between 'u' and 'g'.
#   3. Prepare features and target arrays.
#   4. Train GaussianNB classifier.
#   5. Evaluate and print accuracy.


# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.naive_bayes import GaussianNB  # For Naive Bayes classifier


def main():
	"""
	Main function to load data, engineer features, train Naive Bayes, and evaluate accuracy.
	Steps:
		1. Load SDSS dataset from CSV.
		2. Feature engineering: create 'u_g' as difference between 'u' and 'g'.
		3. Prepare features and target arrays.
		4. Train GaussianNB classifier.
		5. Evaluate and print accuracy.
	"""
	# Load the dataset
	sky_survey = pd.read_csv('SDSS.csv')

	# Feature engineering: create 'u_g' as difference between 'u' and 'g'
	sky_survey['u_g'] = sky_survey['u'] - sky_survey['g']

	# Prepare features (redshift, u_g) and target (class)
	X = sky_survey[['redshift', 'u_g']]
	y = sky_survey['class']

	# Initialize and train the Gaussian Naive Bayes classifier
	model = GaussianNB()
	model.fit(X, y)

	# Evaluate the model using accuracy
	score = model.score(X, y)
	print(f"Accuracy score is {score:.3f}")

if __name__ == "__main__":
	main()