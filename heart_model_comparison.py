import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar datos
df = pd.read_csv("heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

# Árbol podado usando cost-complexity pruning
tree = DecisionTreeClassifier(random_state=1, ccp_alpha=0.01203502)
tree.fit(X_train, y_train)
y_tree_pred = tree.predict(X_test)

# MLP
mlp = MLPClassifier(random_state=1)
mlp.fit(X_train, y_train)
y_mlp_pred = mlp.predict(X_test)

# Accuracy
print(f"Árbol test accuracy: {accuracy_score(y_test, y_tree_pred):.4f}")
print(f"MLP test accuracy:   {accuracy_score(y_test, y_mlp_pred):.4f}")

# Matrices de confusión
print("\nConfusion Matrix - Árbol:")
print(confusion_matrix(y_test, y_tree_pred))

print("\nConfusion Matrix - MLP:")
print(confusion_matrix(y_test, y_mlp_pred))

# Classification Report
print("\nClassification Report - Árbol:")
print(classification_report(y_test, y_tree_pred))

print("\nClassification Report - MLP:")
print(classification_report(y_test, y_mlp_pred))
