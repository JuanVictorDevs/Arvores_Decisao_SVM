import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Mostrar os primeiros registros
print(pd.DataFrame(X, columns=iris.feature_names).head())
print(pd.Series(y).head())

# Dividir o conjunto de treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar árvore de decisão no conjunto de dados de treinamento
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Avaliar a precisão do modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualizar a árvore de decisão gerada
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

print(f'Accuracy: {accuracy}')