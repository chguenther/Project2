import pandas as pd
from sklearn.model_selection import train_test_split
from optimization_utilities import calc_accuracy, calc_classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def preprocess_data(df_earthquake):
    """
    Preprocess earthquake data as follows:
    1. Drop rows with NaN values.
    2. Add a target column for the 'mmi' class.
       This is derived from the existing 'mmi' column as follows:
        * mmi_class = 0 if mmi < 4,
        * mmi_class = 1 if 4 <= mmi < 5
        * mmi_class = 2 if mmi >= 5
    3. Drop columns 'id', 'time', 'place', 'felt', 'cdi', 'mmi', and 'significance'.
    4. Create X and y
    5. Split data into train and test data sets.
    """

    print("Preprocessing data ...")

    print("\tDropping rows with 'NaN' values:")
    num_rows_original = len(df_earthquake)
    # Drop rows with NaN and reset index
    df_cleaned = df_earthquake.dropna().reset_index(drop=True)
    rows_cleaned, cols_cleaned = df_cleaned.shape

    percent_dropped = (num_rows_original-rows_cleaned) / num_rows_original * 100
    print(f"\tDropped {percent_dropped:.2f}% of rows.")
    print(f"\tThere are {rows_cleaned} rows remaining.")
    print(f"\tThere are {cols_cleaned} columns remaining.")
    print("\t" + "-"*60)

    # Add a column for the 'mmi' class. This will be the target
    df_cleaned['mmi_class'] = [0 if mmi<4 else 1 if mmi>=4 and mmi<5 else 2 for mmi in df_cleaned['mmi']]

    # Drop columns 'id', 'time', 'place', 'felt', 'cdi', 'mmi', and 'significance'.
    # The columns 'id', 'time', and 'place' are not relevant for the model.
    # The columns 'felt', 'cdi', 'mmi', and 'significance' might introduce data leakage.
    print("\tDropping columns:")
    columns_to_drop = ['id', 'time', 'place', 'felt', 'cdi', 'mmi', 'significance']
    df_final = df_cleaned.drop(columns=columns_to_drop)
    rows_final, cols_final = df_final.shape
    print(f"\tNumber of rows remaining: {rows_final}.")
    print(f"\tNumber of columns remaining: {cols_final}.")
    print("\t" + "-"*60)

    # Define X and y
    print("\tCreating X and y:")
    X = df_final.drop(columns='mmi_class', axis=1)
    y = df_final['mmi_class']

    X_rows, X_cols = X.shape
    print(f"\tNumber of rows in X: {X_rows}.")
    print(f"\tNumber of columns in X: {X_cols}.")
    print(f"\tNumber of elements in y: {len(y)}.")
    print("="*100 + "\n")

    return(train_test_split(X, y))

def build_earthquake_model(df_earthquake):
    """
    Builds a model to predict the intensity of an earthquake using the 
    Modified Mercalli Intensity (MMI) Scale using the following steps:
    1. Preprocess the data,
    2. Split the data into train and test data sets,
    3. Scale the data using Standard Scaler,
    4. Fit a Random Forest Classifier model using optimized hyperparameters, and
    5. Calculate and print balanced accuracy scores.

    Returns the trained model.
    """
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df_earthquake)

    # Define a list of steps for the pipeline
    steps = [("Scale", StandardScaler()),
             ("Classifier", RandomForestClassifier(max_depth=6))]
    
    # Instantiate a pipeline
    pipeline = Pipeline(steps)

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Calculate balanced accuracy scores and classification report
    train_accuracy = calc_accuracy(X_train, y_train, pipeline)
    test_accuracy = calc_accuracy(X_test, y_test, pipeline)
    class_report = calc_classification_report(X_test, y_test, pipeline)

    # Print balanced accuracy scores of best classifier
    classifier = pipeline[1]
    print(f"Best Classifier: {classifier}:")
    print(f"Balanced Train Accuracy Score: {train_accuracy:.3f}.")
    print(f"Balanced Test Accuracy Score: {test_accuracy:.3f}.")
    print("-"*60)
    print("Classification Report:")
    print(class_report)

    return(pipeline)

def get_better_classifier(pipelines, df):
    """
    Accepts two pipelines and earthquake data.
    Uses two different classifiers (Random Forest Classifier and Multinomial Logistic
    Regression Classifier) to categorize the data, then evaluates which pipeline performs
    best in terms of the balanced test sccuracy score.
    Returns the pipeline that performs better with its balanced train and test accuracy
    scores.
    """

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Evaluating classifiers...")

    train_accuracies = []
    test_accuracies = []
    accuracies = []
    classifiers = []
    for pipeline in pipelines:
        # Fit the first pipeline
        pipeline.fit(X_train, y_train)

        # Calculate balanced accuracy scores
        train_accuracy = calc_accuracy(X_train, y_train, pipeline)
        test_accuracy = calc_accuracy(X_test, y_test, pipeline)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        accuracies.append((train_accuracy, test_accuracy))

        # Print balanced accuracy scores
        classifier = pipeline[1]
        classifiers.append(classifier)
        print(f"\t{classifier}:")
        print(f"\tBalanced Train Accuracy Score: {train_accuracy:.3f}.")
        print(f"\tBalanced Test Accuracy Score: {test_accuracy:.3f}.")
        print("\t" + "-"*60)

    # Return the pipeline that performs best according to the balanced test accuracy score.
    # Also, return the corresponding balanced accuracy scores.
    best_index = test_accuracies.index(max(test_accuracies))
    best_pipeline = pipelines[best_index]
    best_accuracy = accuracies[best_index]
    return(best_pipeline, best_accuracy)

def fine_tune_models(df_earthquake):
    """
    Builds a model to predict the intensity of an earthquake using the 
    Modified Mercalli Intensity (MMI) Scale using the following steps:
    1. Preprocess the data,
    2. Split the data into train and test data sets,
    3. Scale the data using Standard Scaler,
    4. Fit a Random Forest Classifier model using optimized hyperparameters, and
    5. Calculate and print balanced accuracy scores.

    Returns the trained model.
    """

    pipelines = []
    classifiers_to_test = [RandomForestClassifier(max_depth=5),
                           RandomForestClassifier(max_depth=6),
                           RandomForestClassifier(max_depth=7),
                           LogisticRegression(multi_class='multinomial')]

    for classifier in classifiers_to_test:
        steps = [("Scale", StandardScaler()),
                 ("Classifier", classifier)]
        pipelines.append(Pipeline(steps))

    best_pipeline, accuracy = get_better_classifier(pipelines,
                                                    df_earthquake)

    # Print balanced accuracy scores of best classifier
    best_classifier = best_pipeline[1]
    print(f"Best Classifier: {best_classifier}:")
    print(f"Balanced Train Accuracy Score: {accuracy[0]:.3f}.")
    print(f"Balanced Test Accuracy Score: {accuracy[1]:.3f}.")

    return(best_pipeline)

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")