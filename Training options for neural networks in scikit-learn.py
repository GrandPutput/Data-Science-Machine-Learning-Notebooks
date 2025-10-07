
# Train and evaluate a neural network regressor on a sample of the diamonds dataset.
#
# This script demonstrates best practices for data preprocessing, model training,
# and evaluation using scikit-learn's MLPRegressor.
#
# Steps:
#   1. Load and sample diamonds dataset from CSV.
#   2. Prepare features and target arrays.
#   3. Split data into training and testing sets.
#   4. Standardize features.
#   5. Train MLPRegressor model with custom options.
#   6. Evaluate and print R^2 scores for train and test sets.


# Import required libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.neural_network import MLPRegressor  # For neural network regressor
from sklearn.preprocessing import StandardScaler  # For feature scaling


def main():
    """
    Main function to load data, train MLP regressor, and evaluate performance.
    Steps:
        1. Load and sample diamonds dataset from CSV.
        2. Prepare features and target arrays.
        3. Split data into training and testing sets.
        4. Standardize features.
        5. Train MLPRegressor model with custom options.
        6. Evaluate and print R^2 scores for train and test sets.
    """
    # Load and sample the diamonds dataset
    diamonds = pd.read_csv('diamonds.csv')
    diamond_sample = diamonds.sample(1000, random_state=123)

    # Prepare features (drop categorical and target columns) and target
    X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
    y = diamond_sample['price']

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    # Standardize features for better neural network performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the MLP regressor with custom options
    mlp = MLPRegressor(
        random_state=42,
        hidden_layer_sizes=(50, 50, 50),
        learning_rate='adaptive',
        learning_rate_init=0.01,
        batch_size=100,
        max_iter=300
    )
    mlp.fit(X_train, y_train)

    # Evaluate the model using R^2 scores for train and test sets
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"Score for the training data: {train_score:.4f}")
    print(f"Score for the testing data: {test_score:.4f}")

if __name__ == "__main__":
    main()