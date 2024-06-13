import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

def calc_pvalues(X, y):
    """
    Calculate p-values for input X and y.
    Returns sorted list of p-values
    """

    # Instantiate and fit a linear regression model for p-value analysis
    lr = sm.OLS(y, X).fit()

    # Return the p-values of all columns sorted in ascending order
    return(lr.pvalues.sort_values())

def create_model(classifier, seed):
    """
    Create the correct model from the classifier specified on input.
    Output: Instantiated model
    """
    # Match the classifier and instantiate the correct model
    match classifier:
        case 'RFC':
            model = RandomForestClassifier(random_state=seed)
        case 'DT':
            model = DecisionTreeClassifier()
        case 'KNN':
            model = KNeighborsClassifier()
        case 'Logistic Regression':
            model = LogisticRegression(multi_class='multinomial')
        case 'SVM':
            model = SVC()
        case _:
            print(f"Classifier '{classifier}' is invalid.")
            print(f"Choose from one of 'RFC', 'KNN', 'Logistic Regression', or 'SVM'.")
    
    return(model)

def calc_accuracy(X, y, model):
    """
    Calculate balanced accuracy score for input data X, input target y, and
    input model.
    Requires input model to be instantiated and fit.
    Output: Balanced accuracy score. 
    """
    # Make predictions
    predict = model.predict(X)

    return(balanced_accuracy_score(y, predict))

def calc_classification_report(X, y, model):
    """
    Calculate classification report for input data X, input target y, and
    input model.
    Requires input model to be instantiated and fit.
    Output: Balanced accuracy score. 
    """
    # Make predictions
    predict = model.predict(X)

    return(classification_report(y, predict))

def evaluate_base_model(X, y, classifier, seed):
    """
    Split X and y into train and test data sets. Instantiate a model according
    to the classifier specified on input. Calculate balanced accuracy scores for
    train and test data.
    Output: Balance accuracy scores for train and test data sets.
    """
    # Split data into train and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Create and fit the appropriate model
    model = create_model(classifier, seed)
    model.fit(X_train, y_train)

    # Calculate train and test accuracies
    train_accuracy = calc_accuracy(X_train, y_train, model)
    test_accuracy = calc_accuracy(X_test, y_test, model)

    report = calc_classification_report(X_test, y_test, model)

    return(train_accuracy, test_accuracy, report)

def pvalues_optimization(X, y, classifier, seed):
    """
    Removes features one by one starting with the feature with the largest p-value and
    calculates the balanced train and test accuracy stores.
    Input: X - A Pandas DataFrame with the features.
           y - The target.
           classifier - The classifier to use
    Output: A Pandas DataFrame with the number of features removed,
            the balanced train accuracy, and the balanced test accuracy
    """

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Calculate p-values
    pvalues = calc_pvalues(X_train, y_train)

    # Create a list of p-value cutoffs that are equal to the p-values of each feature
    cutoffs = pvalues[pvalues > min(pvalues)].sort_values(ascending=False).to_list()
    # Create a dictionary to store the results of the feature removal
    removed_features = {'Features Removed': range(1, len(cutoffs)+1),
                        'Cutoff': cutoffs,
                        'Train Accuracy': [],
                        'Test Accuracy': []}

    for cutoff in cutoffs:
        # Remove features from train and test data
        X_train_reduced = X_train[pvalues[pvalues < cutoff].keys()]
        X_test_reduced = X_test[pvalues[pvalues < cutoff].keys()]

        # removed_features['Features Left'].append(len(X_train_reduced))

        # Instantiate and fit a model
        model = create_model(classifier, seed)
        model.fit(X_train_reduced, y_train)

        # Calculate accuracies
        train_accuracy = calc_accuracy(X_train_reduced, y_train, model)
        test_accuracy = calc_accuracy(X_test_reduced, y_test, model)

        # Record the accuracy for train and test data
        removed_features['Train Accuracy'].append(train_accuracy)
        removed_features['Test Accuracy'].append(test_accuracy)

    # Create a DataFrame for the results
    df_removed_features = pd.DataFrame(removed_features)

    return(df_removed_features)

def pca_optimization(X, y, classifier, seed):
    """
    For the data (X), target (y), and classifier given on input, iterates through the
    number of components from 2 to the number of features minus oneand performs PCA
    in each iteration. Calculates explained variance, balanced train accuracy score,
    and balanced test accuracy score in each step.
    Output: A Pandas DataFrame with number of components as index and explained
    variance, balanced train accuracy score, and balanced test accuracy score.
    """
    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Define a list used to vary the number of principal components to try from 2 to
    # the number of features minus 1.
    n_principal_components = range(2, len(X.columns))

    pca_optimization = {'Components': n_principal_components,
                        'Explained Variance': [],
                        'Train Accuracy': [],
                        'Test Accuracy': []}

    # Loop over the number of principal components to try
    for n_comps in n_principal_components:
     # Instantiate a PCA model
        pca = PCA(n_components=n_comps, random_state=seed)

        # Fit the PCA model to the train data
        pca.fit(X_train)

        # Transform train and test data
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Record the explained variance
        explained_variance = pca.explained_variance_ratio_.sum()
        pca_optimization['Explained Variance'].append(explained_variance)

        # Instantiate and fit a model with the PCA feature set
        model = create_model(classifier, seed)
        model.fit(X_train_pca, y_train)

        # Calculate accuracies
        train_accuracy = calc_accuracy(X_train_pca, y_train, model)
        test_accuracy = calc_accuracy(X_test_pca, y_test, model)

        # Record the accuracy for train and test data
        pca_optimization['Train Accuracy'].append(train_accuracy)
        pca_optimization['Test Accuracy'].append(test_accuracy)

    # Create a DataFrame
    df_pca_optimization = pd.DataFrame(pca_optimization).set_index('Components')

    return(df_pca_optimization)

def rfc_max_depth_tuning(X, y, seed):
    """
    Increase max_depths of a Random Forest Classifier model by one and calculate the
    balanced train and test accuracy scores for each. Stop when the balanced train
    accuracy reaches 1.0.
    Input:
    X - A dataframe of features
    y - A target
    seed - A random_state seed
    Output:
    A Pandas DataFrame with balanced train and test accuracies and their difference for
    each value of max_depth.
    """

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Define a dictionary for the results
    rfc_scores = {'Depth': [],
                  'Train Accuracy': [],
                  'Test Accuracy': [],
                  'Difference': []}
    
    train_accuracy = 0
    depth = 1
    while train_accuracy < 1.0:
        # Instantiate and fit an RFC model
        model = RandomForestClassifier(max_depth=depth, random_state=seed)
        model.fit(X_train, y_train)

        # Calculate accuracies
        train_accuracy = calc_accuracy(X_train, y_train, model)
        test_accuracy = calc_accuracy(X_test, y_test, model)

        rfc_scores['Depth'].append(depth)
        rfc_scores['Train Accuracy'].append(train_accuracy)
        rfc_scores['Test Accuracy'].append(test_accuracy)
        rfc_scores['Difference'].append(train_accuracy-test_accuracy)
        depth += 1

    return(pd.DataFrame(rfc_scores).set_index('Depth'))

def full_hp_tuning(X, y, params, classifier, seed):
    """
    Perform full hyperparameter tuning on a Random Forest Classifier Model.
    Input:
    X - A dataframe of features
    y - A target
    params - A dictionary of parameters to tune
    classifier - The classifier to tune
    seed - A random_state seed
    Output:
    Best fit parameters
    Balanced Train Accuracy Score for best fit parameters
    Balanced Test Accuracy Score for best fit parameters
    """

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Instantiate and fit an RFC model
    model = create_model(classifier, seed)

    # Instantiate the Randomized Search Estimator
    model_search = RandomizedSearchCV(model, params, random_state=seed, verbose=0)

    # Fit the Randomized Search Estimator on train data
    model_search.fit(X_train, y_train)

    best_fit = model_search.best_params_

    # Make predictions with the hypertuned model
    train_predict = model_search.predict(X_train)
    test_predict = model_search.predict(X_test)

    # Calculate and print the balanced accuracies
    train_accuracy = balanced_accuracy_score(y_train, train_predict)
    test_accuracy = balanced_accuracy_score(y_test, test_predict)

    # Return best hyperparameters
    return(best_fit, train_accuracy, test_accuracy)

def rfc_fine_tune_max_depth(X, y, depths, seed):
    """
    """

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    rfc_scores = {'Depth': [],
                  'Train Accuracy': [],
                  'Test Accuracy': [],
                  'Difference': []}

    for depth in depths:

        # Instantiate and fit an RFC model
        model = RandomForestClassifier(max_depth=depth, random_state=seed)
        model.fit(X_train, y_train)

        # Calculate accuracies
        train_accuracy = calc_accuracy(X_train, y_train, model)
        test_accuracy = calc_accuracy(X_test, y_test, model)

        rfc_scores['Depth'].append(depth)
        rfc_scores['Train Accuracy'].append(train_accuracy)
        rfc_scores['Test Accuracy'].append(test_accuracy)
        rfc_scores['Difference'].append(train_accuracy-test_accuracy)

    return(pd.DataFrame(rfc_scores).set_index('Depth'))

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")