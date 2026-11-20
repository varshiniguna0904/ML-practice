import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
print("ML environment ready")
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
iris_df = iris.frame
Xc = iris_df.drop(columns=["target"])
yc = iris.target
print(Xc.head())
print(iris.target_names)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xtr, Xte, ytr, yte = train_test_split(
Xc, yc, test_size=0.2,
random_state=42, stratify=yc
)
log_model = make_pipeline(
StandardScaler(),
LogisticRegression(max_iter=2000)
)
log_model.fit(Xtr, ytr)
pred_log = log_model.predict(Xte)
print(pred_log[:10])
from sklearn.neighbors import KNeighborsClassifier
knn = make_pipeline(
StandardScaler(),
KNeighborsClassifier(n_neighbors=5)
)
knn.fit(Xtr, ytr)
pred_knn = knn.predict(Xte)
print(pred_knn[:10])
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(Xtr, ytr)
pred_nb = nb.predict(Xte)
print(pred_nb[:10])
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(
max_depth=3, random_state=42
)
tree.fit(Xtr, ytr)
pred_tree = tree.predict(Xte)
plot_tree(tree, feature_names=Xc.columns,
class_names=iris.target_names,
filled=True)
plt.show()
from sklearn.svm import SVC
svm = make_pipeline(
StandardScaler(),
SVC(kernel="rbf")
)
svm.fit(Xtr, ytr)
pred_svm = svm.predict(Xte)
print(pred_svm[:10])
from sklearn.metrics import accuracy_score
print("Classes:", iris.target_names)
print("Logistic accuracy:",
accuracy_score(yte, pred_log))
print("Tree accuracy:",
accuracy_score(yte, pred_tree))
print("Balanced class weights enabled")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(
    Xtr, ytr
)
print("Before:", 
      (ytr).value_counts())
print("After:",
      pd.Series(y_res).value_counts())
from sklearn.inspection import DecisionBoundaryDisplay
X2 = Xc[["petal length (cm)",
"petal width (cm)"]]
boundary_model = make_pipeline(
StandardScaler(),
LogisticRegression(max_iter=2000)
)
boundary_model.fit(X2, yc)
DecisionBoundaryDisplay.from_estimator(
boundary_model, X2,
response_method="predict",
plot_method="contourf"
)
plt.scatter(X2.iloc[:,0], X2.iloc[:,1], c=yc)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()
from sklearn.metrics import (
accuracy_score, precision_score,
recall_score, f1_score
)
print("Accuracy:", accuracy_score(yte,pred_log))
print("Precision:", precision_score(
yte,pred_log,average="weighted"))
print("Recall:", recall_score(
yte,pred_log,average="weighted"))
print("F1:", f1_score(
yte,pred_log,average="weighted"))
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(
yte, pred_log,
display_labels=iris.target_names
)
plt.title("Iris Confusion Matrix")
plt.show()
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
models = {
"Logistic": log_model,
"KNN": knn,
"Naive Bayes": nb,
"Decision Tree": tree,
"SVM": svm
}
for name, model in models.items():
    pred = model.predict(Xte)
    print(name,
          "Accuracy:", round(accuracy_score(yte,pred),3),
          "F1:", round(f1_score(yte,pred,average="weighted"),3))
