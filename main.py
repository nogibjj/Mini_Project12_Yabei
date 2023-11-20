import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets
from mlflow.models import infer_signature


def main():
    """Runs a basic logistic regression model
    and logs it with mlflow using the wine dataset"""
    # Load the wine dataset
    X, y = datasets.load_wine(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 2000,
        "multi_class": "auto",
        "random_state": 1234,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy as a target loss metric
    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag for easy identification
        mlflow.set_tag("Training Info", "Basic LR model for wine data")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="wine_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="wine-classification-model",
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    return loaded_model


if __name__ == "__main__":
    main()
