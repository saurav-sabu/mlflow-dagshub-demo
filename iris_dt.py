import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

max_depth = 1

mlflow.set_experiment("iris-dt")

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)

    y_pred = dt.predict(X_test)

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
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt,"Decision Tree")
    mlflow.set_tag("author","saurav")
    mlflow.set_tag("model_name","Decision Tree")