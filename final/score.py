from rouge import Rouge
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the predicted labels data
predicted_labels_data = pd.read_csv('predicted_labels2.csv')

# Get the manually labeled and predicted labels
manually_labeled = predicted_labels_data['orj_labl'].tolist()
predicted_labels = predicted_labels_data['predicted_label'].tolist()

# Convert labels to lowercase for comparison
manually_labeled = [label.lower() for label in manually_labeled]
predicted_labels = [label.lower() for label in predicted_labels]

accuracy = accuracy_score(manually_labeled, predicted_labels)

precision = precision_score(
    manually_labeled, predicted_labels, average="weighted")
recall = recall_score(manually_labeled, predicted_labels, average="weighted")
f1 = f1_score(manually_labeled, predicted_labels, average="weighted")

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')

conf_matrix = confusion_matrix(manually_labeled, predicted_labels)
print(f'Confusion Matrix: {conf_matrix}')


print(manually_labeled[:5])
print(predicted_labels[:5])

# Calculate Rouge score
rouge = Rouge()
scores = rouge.get_scores(predicted_labels, manually_labeled, avg=True)

# Print the Rouge score
print("Rouge Score: ", scores)
