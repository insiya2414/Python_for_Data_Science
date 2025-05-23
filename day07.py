'''Q36) function to calculate the silhouette score for clustering results'''
#The Silhouette Score is a measure of how well each point lies within its cluster.
# Importing silhouette_score function from scikit-learn
from sklearn.metrics import silhouette_score

# Define a function to calculate the silhouette score
def calculate_silhouette_score(X, labels):
    """
    Calculates the silhouette score for the given dataset and cluster labels.

    Parameters:
    - X: array-like of shape (n_samples, n_features), the feature matrix.
    - labels: array-like of shape (n_samples,), the cluster labels assigned to each point.

    Returns:
    - score: float, the silhouette score (ranges from -1 to 1).
    """
    return silhouette_score(X, labels)

# Import a function to create synthetic clustering data
from sklearn.datasets import make_blobs

# Create a small synthetic dataset with 10 samples and 2 cluster centers
# - X will be a 2D array of coordinates
# - labels will be the cluster index assigned to each point
X, labels = make_blobs(n_samples=10,    # Total number of data points
                       centers=2,       # Number of clusters
                       random_state=0)  # Seed for reproducibility

# Call the function and print the silhouette score for this dataset
print("Silhouette Score:", calculate_silhouette_score(X, labels))


'''Q37) One-hot encoding of categorical variables in a dataset'''
# You're using pandas.get_dummies()[a function in pd library] to convert a categorical column (Color) into one-hot encoded columns — 
# this is a common technique in machine learning to convert non-numeric data into numeric format.
# Importing the pandas library
import pandas as pd

# Function to perform one-hot encoding on a specified column
def one_hot_encode(data, column):
    """
    Converts a categorical column into one-hot encoded columns.

    Parameters:
    - data: pd.DataFrame, the input DataFrame
    - column: str, the name of the categorical column to encode

    Returns:
    - pd.DataFrame with new one-hot encoded columns replacing the original column
    """
    return pd.get_dummies(data, columns=[column])

# Create a simple DataFrame with a single categorical column 'Color'
df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})

# Apply one-hot encoding on the 'Color' column
print(one_hot_encode(df, 'Color'))

# Output:   Color_Blue  Color_Green  Color_Red
# 0           0            0          1
# 1           1            0          0
# 2           0            1          0

'''Q38) Principal Component Analysis (PCA) to reduce dimensionality'''
# PCA (Principal Component Analysis) is a technique used to:
# Reduce the number of features (dimensions) in th dataset.
# While preserving as much information (variance) as possible.

# It's used when: we have too many features (e.g., hundreds).
# we want to simplify the data before visualization or feeding it into a machine learning model.

import numpy as np  # Import NumPy for numerical operations

# Define the PCA function
def pca(X, n_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features), the input dataset
    - n_components: int, the number of principal components to retain

    Returns:
    - Transformed data with reduced dimensions
    """

    # Step 1: Center the data (subtract the mean from each feature)
    # Now the data is centered at the origin (mean = 0), which is important for finding the direction of maximum variance.
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute the covariance matrix of the centered data
    # rowvar=False ensures columns are treated as features
    # The covariance matrix tells how features vary together
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix
    # Eigenvectors = directions (new axes)
    # Eigenvalues = how much variance (information) is in that direction
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Select the top 'n_components' eigenvectors (principal components)
    # Eigenvectors are in columns; we slice to select the first n_components
    principal_components = eigenvectors[:, :n_components]

    # Step 5: Project the centered data onto the principal components
    return X_centered.dot(principal_components)

# Sample 2D data
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Reduce from 2D to 1D
print(pca(X, 1))

# Output : [[-2.82842712]
#          [ 0.        ]
#          [ 2.82842712]]

'''Q39)  function to handle missing data using multiple imputation'''
# SimpleImputer is a tool from scikit-learn that allows us to fill in missing values using a simple strategy like:
# Mean, Median, Most Frequent, Constant value
from sklearn.impute import SimpleImputer
import numpy as np

def impute_missing_data(data):
    imputer = SimpleImputer(strategy = 'mean')
    # Fit the imputer on the data and transform it (replace NaN with the mean)
    return imputer.fit_transform(data) # fit_transform() is used to fit the imputer on the data and transform it.

# Sample 2D array with a missing value (np.nan)
data = np.array([[1,2],[np.nan,3],[7,6]])
print(impute_missing_data(data))

# Output :[[1. 2.]
#         [4. 3.]
#         [7. 6.]]

'''Q40) Group a dataset by a column and calculate the rolling average for another column'''
import pandas as pd

# Define function to compute rolling average
def rolling_average(df, group_col, target_col, window):
    # Group the DataFrame by the group_col (e.g., 'Group')
    # Then apply rolling(window) to the target_col (e.g., 'Value')
    # Finally, reset_index() to turn the output back into a DataFrame
    return df.groupby(group_col)[target_col].rolling(window=window).mean().reset_index()

# Sample data
data = {'Group': ['A', 'A', 'B', 'B'], 'Value': [10, 20, 30, 40]}
df = pd.DataFrame(data)

# Compute 2-point rolling average grouped by 'Group'
print(rolling_average(df, 'Group', 'Value', 2))

# Output:   Group  level_1  Value
#       0     A        0    NaN
#       1     A        1   15.0
#       2     B        2    NaN
#       3     B        3   35.0



