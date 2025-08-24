import csv
import sys
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

# TODO: Do k-fold cross-validation, since this is a small dataset and we don't want to lose any data


TEST_SPLIT_PROPORTION = 0.3


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
    evaluate(yTest, predictions)

    # # gather and display performance stats
    # correctCount = (yTest == predictions).sum()
    # #incorrectCount = (yTest != predictions).sum()
    # totalCount = len(predictions)

    # print(f"Accuracy: {(correctCount / totalCount * 100):.2f}%")


def evaluate(labels, predictions):
    totalLoss = 0

    # Effectively use the L1 loss function
    for lab, pred in zip(labels, predictions):
        print(lab, pred)



def trainModel(evidence, labels):
    # Change model here if necessary
    model = Perceptron()
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

            # Quality is a score beween 0 and 10
            # While it's discrete, we're treating it as continuous for better measuring loss
            labels.append(
                float(row["quality"])
            )
    
    return evidence, labels
    
            

if __name__ == "__main__":
    main()