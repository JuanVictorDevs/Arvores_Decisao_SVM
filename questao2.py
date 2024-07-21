import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados
digits = datasets.load_digits()

# Mostrar exemplos de imagens e seus rótulos
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Training: {label}')

# Separando imagens para o treinamento
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Dividir o conjunto de dados em treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Treinando modelo
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# Avaliar a precisão
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

# Mostrando previsões
_, axes = plt.subplots(2, 4)
images_and_predictions = list(zip(digits.images[n_samples // 2:], y_pred))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.show()