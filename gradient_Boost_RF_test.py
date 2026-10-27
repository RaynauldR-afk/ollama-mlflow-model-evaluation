import mlflow
import mlflow.sklearn
from sklearn.datasets import  make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

X,y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Model Comparison:RF vs Gradient Boosting")

def train_and_log_model(model_name, model, params):
    with mlflow.start_run(run_name=model_name):

        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        precision = precision_score(y_test,predictions)

        #Log parameters, metrics and he model artificat
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.sklearn.load_model(model, artifact_path ="model")

        print(f"[{model_name}] Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

if __name__=="__main__":
    rf_params ={"n_estimators": 100, "max_depth":5, "random_state": 42}
    rf_model= RandomForestClassifier(**rf_params)
    train_and_log_model("Random Forest", rf_model, rf_params)

    #GB model
    gb_params={
        "n_estimators":100,
        "max_depth":3,
        "learning_rate":0.1,
        "random_state": 42,
    }

    gb_model=GradientBoostingClassifier(**gb_params)
    train_and_log_model("Gradient Boosting", gb_model, gb_params)