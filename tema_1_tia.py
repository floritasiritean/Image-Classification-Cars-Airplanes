import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import random

# Funcția pentru extragerea momentelor Hu ca descriptori de formă
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Funcția pentru încărcarea și preprocesarea imaginilor
def load_and_preprocess_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (64, 64))
    return image

# Funcția pentru încărcarea datelor
def load_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            label = 1 if filename.startswith("m") else 0
            labels.append(label)
            image = load_and_preprocess_image(file_path)
            features = extract_hu_moments(image)
            data.append(features)
    return np.array(data), np.array(labels)

# Încărcarea datelor
data_car, labels_car = load_data('./Masini')
data_plane, labels_plane = load_data('./Avioane')

# Concatenarea datelor pentru mașini și avioane
data = np.concatenate((data_car, data_plane))
labels = np.concatenate((labels_car, labels_plane))

# Împărțirea datelor în seturi de antrenare (80%) și validare (20%)
X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2, random_state=42)

# Calea către folderul cu imagini pentru testare
test_folder_path = './Testare'

# Liste pentru stocarea datelor de testare, etichetelor și predicțiilor
test_data = []
test_labels = []
test_predictions = []

# Parcurgerea imaginilor pentru testare și încărcarea datelor
for filename in os.listdir(test_folder_path):
    if filename.endswith(".jpg"):
        file_path = os.path.join(test_folder_path, filename)
        label = 1 if filename.startswith("m") else 0
        test_labels.append(label)
        image = load_and_preprocess_image(file_path)
        features = extract_hu_moments(image)
        test_data.append(features)

# Convertirea listelor în array-uri NumPy
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Crearea unui clasificator k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=13)

# Antrenarea clasificatorului pe setul de antrenare
knn_classifier.fit(X_train, y_train)

# Realizarea predicțiilor pe setul de validare
y_valid_pred = knn_classifier.predict(X_valid)

# Evaluarea performanței pe setul de validare
accuracy_valid = metrics.accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {accuracy_valid:.2f}")

# Realizarea predicțiilor pe setul de testare
test_predictions = knn_classifier.predict(test_data)

# Evaluarea performanței pe setul de testare
accuracy_test = metrics.accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {accuracy_test:.2f}")

# Afișarea rezultatelor
print("\nResults on the Test Set:")
for i, filename in enumerate(os.listdir(test_folder_path)):
    if filename.endswith(".jpg"):
        prediction = test_predictions[i]
        true_label = test_labels[i]
        result = "Masina" if prediction == 1 else "Avion"
        true_class = "Masina" if true_label == 1 else "Avion"
        correct = "Corect" if prediction == true_label else "Incorect"
        print(f"Image: {filename}, Prediction: {result}, True Class: {true_class}, "f"True Label: {true_label}, Accuracy: {correct}")

# Calcularea matricei de confuzie
conf_matrix = confusion_matrix(test_labels, test_predictions)

# Afișarea matricei de confuzie sub formă de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Avion', 'Masina'], yticklabels=['Avion', 'Masina'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Subplot pentru ROC Curve
plt.subplot(1, 2, 2)
# Obțineți scorurile de probabilitate pentru clasa pozitivă (masini)
y_test_scores = knn_classifier.predict_proba(test_data)[:, 1]

# Calculați curba ROC și AUC
fpr, tpr, thresholds = roc_curve(test_labels, y_test_scores)
roc_auc = auc(fpr, tpr)

# Afișați ROC Curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Afișare generală
plt.tight_layout()
plt.show()

# Histograma pentru distribuția etichetelor în setul de date
plt.figure(figsize=(6, 4))
sns.countplot(x=labels, palette='viridis')
plt.title('Distribution of Labels in the Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# Realizarea predicțiilor pe setul de validare
y_train_pred = knn_classifier.predict(X_train)

# Evaluarea performanței pe setul de antrenare
accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy_train:.2f}")

# Calcularea matricei de confuzie pentru setul de antrenare
conf_matrix_train = confusion_matrix(y_train, y_train_pred)

# Afișarea matricei de confuzie pentru setul de antrenare sub formă de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Avion', 'Masina'], yticklabels=['Avion', 'Masina'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Set')
plt.show()
# Realizarea predicțiilor pe setul de validare
y_valid_pred = knn_classifier.predict(X_valid)

# Evaluarea performanței pe setul de validare
accuracy_valid = metrics.accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {accuracy_valid:.2f}")

# Calcularea matricei de confuzie pentru setul de validare
conf_matrix_valid = confusion_matrix(y_valid, y_valid_pred)

# Afișarea matricei de confuzie pentru setul de validare sub formă de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_valid, annot=True, fmt='d', cmap='Blues', xticklabels=['Avion', 'Masina'], yticklabels=['Avion', 'Masina'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# Afișarea rezultatelor pentru primele câteva imagini din setul de test
num_examples_to_display = 8
random_indices = random.sample(range(len(test_labels)), num_examples_to_display)

# Creare o figură pentru subplots
fig, axs = plt.subplots(1, num_examples_to_display, figsize=(15, 5))

for i, idx in enumerate(random_indices):
    filename = os.listdir(test_folder_path)[idx]
    file_path = os.path.join(test_folder_path, filename)
    label = "Avion" if test_labels[idx] == 0 else "Masina"
    prediction = "Avion" if test_predictions[idx] == 0 else "Masina"

    # Afișarea imaginii
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Afișarea imaginilor și adăugarea textului etichetelor
    axs[i].imshow(img)
    axs[i].set_title(f"True: {label}\nPredicted: {prediction}")
    axs[i].axis('off')

plt.show()


plt.show()



