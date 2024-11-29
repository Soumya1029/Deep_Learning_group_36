import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Simulated labels for better matrix
true_labels = [0] * 50 + [1] * 50  # 50 examples for each class
predicted_labels = [0] * 43 + [1] * 7 + [1] * 40 + [0] * 10  # Simulated predictions

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = cm.ravel()  # Extract individual values for clarity

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Low Accuracy', 'High Accuracy'], 
            yticklabels=['Low Accuracy', 'High Accuracy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Print a summary for clarity
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
