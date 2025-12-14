from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data[:, :3]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

names = ["Decision Tree", "Random Forest", "Naive Bayes"]

classifiers =[
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    NB()]

for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print("Accuracy: %0.2f\n"
          "Confusion Matrix:\n%s"
          % (accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)))
