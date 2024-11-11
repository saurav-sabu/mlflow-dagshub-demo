import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/saurav-sabu/mlflow-dagshub-demo.mlflow")


dagshub.init(repo_owner='saurav-sabu', repo_name='mlflow-dagshub-demo', mlflow=True)

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

max_depth = 1
n_estimators = 100

mlflow.set_experiment("iris-dt")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_metric("accuracy",acc)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n estimators",n_estimators)
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf,"Random Forest")
    mlflow.set_tag("author","saurav")
    mlflow.set_tag("model_name","Random Forest")