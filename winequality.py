import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score


TEST_SPLIT_PROPORTION = 0.2


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python winequality.py <<filepath>>")

    # Parse data into evidence and labels
    evidence, labels = loadData(sys.argv[1])

    # Split data into training set and testing set (i.e. holdout cross-validation)
    xTrain, xTest, yTrain, yTest = train_test_split(
        evidence, labels, test_size=TEST_SPLIT_PROPORTION
    )

    # train model
    model = trainModel(xTrain, yTrain)
    
    # make predictions and evaluate accuracy
    predictions = model.predict(xTest)
    mae = calculateMeanAbsoluteError(yTest, predictions)
    r2 = calculateR2(yTest, predictions)

    print(f"MAE: {mae:.2f}")  # Mean Absolute Error; average L1 loss
    print(f"R^2: {r2:.2f}")  # R^2; the proportion of variance explained by the model


def trainModel(evidence, labels):
    model = LinearRegression()
    model.fit(evidence, labels)
    return model


def calculateMeanAbsoluteError(labels, predictions):
    return np.mean(np.abs(np.array(labels) - np.array(predictions)))


def calculateR2(labels, predictions):
    return r2_score(labels, predictions)


def loadData(fileName):
    evidence = []
    labels = []

    with open(fileName) as f:

        # The "csv" files are semicolon separated, not comma
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:

            # All of the evidence data is continuous, hence float type-casting
            evidence.append([
                float(row["fixed acidity"]),
                float(row["volatile acidity"]),
                float(row["citric acid"]),
                float(row["residual sugar"]),
                float(row["chlorides"]),
                float(row["free sulfur dioxide"]),
                float(row["total sulfur dioxide"]),
                float(row["density"]),
                float(row["pH"]),
                float(row["sulphates"]),
                float(row["alcohol"]),
            ])

            # Quality is a score beween 0 and 10
            labels.append(
                int(row["quality"])
            )
    
    return evidence, labels
    
            

if __name__ == "__main__":
    main()