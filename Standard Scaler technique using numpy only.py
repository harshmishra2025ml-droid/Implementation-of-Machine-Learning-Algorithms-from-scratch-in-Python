
import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    # Calculate mean and standard deviation
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Avoid division by zero for constant columns
        self.std[self.std == 0] = 1

    # Transform the data
    def transform(self, X):
        return (X - self.mean) / self.std

    # Fit and transform together
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # Convert scaled data back to original data
    def inverse_transform(self, X):
        return (X * self.std) + self.mean


# Sample dataset
X = np.array([
    [20, 20000],
    [30, 30000],
    [40, 40000],
    [50, 50000],
    [60, 60000]
])

# Create object
scaler = StandardScaler()

# Apply scaling
X_scaled = scaler.fit_transform(X)

print("Mean:", scaler.mean)
print("Standard Deviation:", scaler.std)
print("Scaled Data:\n", X_scaled)

# Original data back
print("Original Data:\n", scaler.inverse_transform(X_scaled))
