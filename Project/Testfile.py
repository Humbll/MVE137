import numpy as np
import pandas as pd
from math import sqrt
import random
from matplotlib import pyplot as plt
import warnings

confidence = []


def knn_algorithm(traindata, testdata, k_neighbours=5):
    num_classes = len(np.unique(traindata[:, -1]))
    
    distances = np.sqrt(np.sum((traindata[:, :-1] - testdata) ** 2, axis=1))
    sorted_indices = np.argsort(distances)
    k_indices = sorted_indices[:k_neighbours]
    k_nearest_labels = traindata[k_indices, -1]
    
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    classification_result, = unique_labels[np.argmax(counts)]
    
    # Calculate confidence as the ratio of the most common class count to k_neighbors
    confidence = counts.max() / k_neighbours
    
    return classification_result, confidence



columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
           'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']
#Import the data file by uncommenting below and setting the path to the dataset
df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=columns)
df.head()


num_classes = df['class'].nunique()
print("Number of different cancer classes:", num_classes)

# Step 3: Plot a pie chart to visualize the class distribution
class_counts = df['class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Cancer Classes")
plt.show()

# Step 4: Find and replace missing values with the mode of each column
df = df.replace('?', pd.NA)
df = df.fillna(df.mode().iloc[0])

# Verify that missing values have been replaced
missing_values = df.isnull().sum().sum()
print("Total missing values after replacement:", missing_values)


# Drop obviously unwanted columns
unwanted_columns = ['id']  # Add the names of the unwanted columns to this list

# Use the drop method to remove the unwanted columns
df = df.drop(columns=unwanted_columns)

# Print the first few rows of the DataFrame to verify the columns have been dropped
print(df.head())



# Shuffle the dataset to randomize the order
shuffled_df = df.sample(frac=1, random_state=42)  # You can change the random_state for reproducibility

# Calculate the number of samples for the test set (20%)
test_size = int(0.2 * len(shuffled_df))

# Split the shuffled dataset into training and test sets
train_set = shuffled_df[test_size:]
test_set = shuffled_df[:test_size]

# Reset the index for the new DataFrames
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

# Print the sizes of the training and test sets
print("Training set size:", len(train_set))
print("Test set size:", len(test_set))




# Split the data into features (X) and labels (y)
X_train = train_set.iloc[:, 1:-1].values  # Features for training
y_train = train_set.iloc[:, -1].values    # Labels for training
X_test = test_set.iloc[:, 1:-1].values    # Features for testing
y_test = test_set.iloc[:, -1].values      # True labels for testing


def knn_algorithm(traindata, testdata, k_neighbours=5):
    return classification_result, confidence


k_neighbours = 5
# Initialize lists to store incorrect predictions and their corresponding confidences
incorrect_predictions = []
incorrect_confidences = []

# Iterate through the test set
for i in range(len(X_test)):
    test_instance = X_test[i]      # Features of the test instance
    true_label = y_test[i]         # True class label
    
    # Use the KNN classifier to make a prediction
    predicted_label, confidence = knn_algorithm(X_train, test_instance, k_neighbours)
    
    # Check if the prediction is incorrect
    if predicted_label != true_label:
        incorrect_predictions.append(i)
        incorrect_confidences.append(confidence)

# Print the confidence for the incorrect predictions
print("Confidence for incorrect predictions:")
for i, confidence in zip(incorrect_predictions, incorrect_confidences):
    print(f"Sample {i+1}: Confidence = {confidence:.2f}")

# Calculate and print the accuracy of the predictions
accuracy = 1 - (len(incorrect_predictions) / len(X_test))
print("\nAccuracy:", accuracy)