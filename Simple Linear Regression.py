
import numpy as np
import matplotlib.pyplot as plt


# =====================================
# SIMPLE LINEAR REGRESSION FROM SCRATCH
# =====================================

class SimpleLinearRegression:

    def __init__(self):
        self.m = None
        self.c = None

    # Train the model
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        # Calculate mean
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate slope
        numerator = np.sum(
            (X - x_mean) * (y - y_mean)
        )

        denominator = np.sum(
            (X - x_mean) ** 2
        )

        if denominator == 0:
            raise ValueError("All X values are identical.")

        self.m = numerator / denominator

        # Calculate intercept
        self.c = y_mean - (self.m * x_mean)

    # Predict new values
    def predict(self, X):

        if self.m is None or self.c is None:
            raise ValueError("Train the model first!")

        X = np.array(X)

        return self.m * X + self.c

    # Calculate R2 score
    def score(self, X, y):

        y = np.array(y)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)

        ss_total = np.sum(
            (y - np.mean(y)) ** 2
        )

        if ss_total == 0:
            raise ValueError("R2 score is undefined.")

        return 1 - (ss_res / ss_total)


# =====================================
# MAIN PROGRAM
# =====================================

if __name__ == "__main__":

    # Dataset
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    # Create model
    model = SimpleLinearRegression()

    # Train model
    model.fit(X, y)

    # Display model parameters
    print("Slope:", model.m)
    print("Intercept:", model.c)

    print("\nRegression Equation:")
    print(f"y = {model.m:.2f}x + {model.c:.2f}")

    # Predictions
    predictions = model.predict([6, 7, 8])

    print("\nPredictions:")
    print(predictions)

    # Model evaluation
    r2 = model.score(X, y)

    print("\nR2 Score:", r2)

    # Visualization
    plt.scatter(
        X, y,
        color="blue",
        label="Actual Data"
    )

    plt.plot(
        X,
        model.predict(X),
        color="red",
        label="Regression Line"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Simple Linear Regression From Scratch")

    plt.legend()
    plt.grid(True)
    plt.show()
