import matplotlib.pyplot as plt
import os

plt.figure(figsize=(8,5))
plt.plot(cnn_train_losses, label="Training Loss")
plt.plot(cnn_val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
os.makedirs("results", exist_ok=True)
plt.savefig("results/cnn_training_and_validation_loss.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(cnn_train_accuracies, label="Training Accuracy")
plt.plot(cnn_val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracies")
plt.legend()
plt.savefig("results/cnn_training_and_validation_accuracies.png", dpi=150, bbox_inches="tight")
plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np

cnn_labels, cnn_preds = get_test_predictions(net_cnn)

# 1. compute confusion matrix and find top 5 confused pairs automatically
cnn_cm = confusion_matrix(cnn_labels, cnn_preds)
cm_off_diag = cnn_cm.copy()
np.fill_diagonal(cm_off_diag, 0)  # ignore correct predictions

top_n = 5
flat_indices = np.argsort(cm_off_diag, axis=None)[::-1][:top_n]
top_pairs = [np.unravel_index(idx, cm_off_diag.shape) for idx in flat_indices]

print("Top 5 misclassified category pairs:")
for true_idx, pred_idx in top_pairs:
    count = cm_off_diag[true_idx, pred_idx]
    print(f"  {classes[true_idx]} misclassified as {classes[pred_idx]}: {count} times")

# display
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cnn_cm, display_labels=classes)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("results/cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. collect misclassified images once (with which true/pred pair they belong to)
cnn_misclassified = []
net_cnn.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net_cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        wrong = predicted != labels
        for img, pred, truth in zip(inputs[wrong], predicted[wrong], labels[wrong]):
            cnn_misclassified.append((img.cpu(), pred.cpu().item(), truth.cpu().item()))

# 3. for each top confused pair, grab a few example images
def show_examples_for_pair(misclassified_list, true_idx, pred_idx, n=4):
    examples = [(img, pred, truth) for img, pred, truth in misclassified_list
                if truth == true_idx and pred == pred_idx][:n]
    return examples

fig, axes = plt.subplots(top_n, 4, figsize=(10, 12))
for row, (true_idx, pred_idx) in enumerate(top_pairs):
    examples = show_examples_for_pair(cnn_misclassified, true_idx, pred_idx, n=4)
    for col in range(4):
        ax = axes[row, col]
        if col < len(examples):
            img, pred, truth = examples[col]
            img = img / 2 + 0.5
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"P: {classes[pred]}\nT: {classes[truth]}", fontsize=8)
        ax.axis("off")

plt.tight_layout()
plt.savefig("results/cnn_misclassified.png", dpi=150, bbox_inches="tight")
plt.show()

# precision: how many images predicted as a class were actually the class?
# recall: how many images belonging to a class did the model correctly identify?
# F1 score: the harmonic mean of precision and recall, providing a balanced measure of performance

from sklearn.metrics import classification_report

print(
    classification_report(
        cnn_labels, cnn_preds,
        target_names=classes
    )
)