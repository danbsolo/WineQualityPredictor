import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SPLIT_PROPORTION = 0.3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python winequality.py <<filepath>>")

    # Parse data into evidence and labels
    evidence, labels = loadData(sys.argv[1])

    # Split data into training set and testing set
    xTrain, xTest, yTrain, yTest = train_test_split(
        evidence, labels, test_size=TEST_SPLIT_PROPORTION
    )

    # train model
    model = trainModel(xTrain, yTrain)
    predictions = model.predict(xTest)

    print(f"Correct: {(yTest == predictions).sum()}")
    print(f"Incorrect: {(yTest != predictions).sum()}")


def trainModel(evidence, labels):
    # Change model here if necessary
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


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

            # Quality is a score beween 0 and 10. We'll make it integer just because it can
            labels.append(
                int(row["quality"])
            )
    
    return evidence, labels
    
            

if __name__ == "__main__":
    main()