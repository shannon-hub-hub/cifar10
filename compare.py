print(type(net_cnn))
print(type(net_res))

cnn_labels, cnn_preds = get_test_predictions(net_cnn)
test_accuracy_cnn = 100 * accuracy_score(cnn_labels, cnn_preds)
print(f"CNN test accuracy: {test_accuracy_cnn:.2f}%")

resnet_labels, resnet_preds = get_test_predictions(net_res)
test_accuracy_resnet = 100 * accuracy_score(resnet_labels, resnet_preds)
print(f"ResNet test accuracy: {test_accuracy_resnet:.2f}%")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cnn_params = count_params(net_cnn)      # your plain CNN instance
resnet_params = count_params(net_res)  # your ResNet18_CIFAR instance

print(f"CNN parameters: {cnn_params:,}")
print(f"ResNet parameters: {resnet_params:,}")

import pandas as pd

results_table = pd.DataFrame({
    "Model": ["Plain CNN", "ResNet-18"],
    "Test Accuracy (%)": [test_accuracy_cnn, test_accuracy_resnet],
    "Parameters": [cnn_params, resnet_params],
    "Epochs Trained": [20, 20],
})
print(results_table.to_markdown(index=False))
