
# Train and evaluate a neural network using Keras on a sample of the diamonds dataset.
#
# This script demonstrates best practices for data preprocessing, model training,
# and evaluation using Keras Sequential API.
#
# Steps:
#   1. Load and sample diamonds dataset from CSV.
#   2. Prepare features and target arrays.
#   3. Split data into training and testing sets.
#   4. Build and compile Keras Sequential model.
#   5. Train the model.
#   6. Output predictions and compare to actual values.


# Import required libraries
import os  # For environment variable settings
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
import keras  # For building and training neural networks


def main():
    """
    Main function to load data, build and train Keras model, and evaluate predictions.
    Steps:
        1. Load and sample diamonds dataset from CSV.
        2. Prepare features and target arrays.
        3. Split data into training and testing sets.
        4. Build and compile Keras Sequential model.
        5. Train the model.
        6. Output predictions and compare to actual values.
    """
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # Set random seed for reproducibility
    keras.utils.set_random_seed(812)

    # Load and sample the diamonds dataset
    df = pd.read_csv('diamonds.csv')
    diamond_sample = df.sample(1000, random_state=12)

    # Prepare features (drop categorical and target columns) and target
    X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
    y = diamond_sample['price']

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Build the Keras Sequential model
    model = keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='linear'),
        keras.layers.Dense(64, activation='linear'),
        keras.layers.Dense(1, activation='linear')
    ])

    # Print model summary
    print(model.summary(line_length=80))

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='Adam', loss='MeanSquaredError', metrics=['mse'])

    # Train the model for 5 epochs with batch size 100 and 10% validation split
    model.fit(X_train, y_train, batch_size=100, epochs=5, validation_split=0.1, verbose=0)

    # Predict and compare first 3 test samples
    predictions = model.predict(X_test[:3], verbose=0)
    print('Predictions:', predictions.round(3))
    print('Actual values:', y_test.iloc[:3].values)

if __name__ == "__main__":
    main()