import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

net_res = ResNet18_CIFAR().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_res.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

num_epochs = 20
res_train_losses = []
res_val_losses = []
res_val_accuracies = []
res_train_accuracies = []
epoch_times = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    net_res.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net_res(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    res_train_loss = running_loss / len(trainloader)
    res_train_accuracy = 100 * correct / total
    res_train_losses.append(res_train_loss)
    res_train_accuracies.append(res_train_accuracy)

    ##### validation
    net_res.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net_res(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    res_val_loss = running_loss / len(valloader)
    res_val_accuracy = 100 * correct / total
    res_val_losses.append(res_val_loss)
    res_val_accuracies.append(res_val_accuracy)


    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train loss: {res_train_loss:.4f} | Train acc: {res_train_accuracy:.2f}% | "
        f"Val loss: {res_val_loss:.4f} | Val acc: {res_val_accuracy:.2f}% | "
        f"Time: {epoch_time:.1f}s"
    )

total_time = sum(epoch_times)
print(f"finished training in {total_time/60:.1f} min | avg epoch time: {total_time/len(epoch_times):.1f}s")


PATH = './cifar_resnet18.pt'
torch.save(net_res.state_dict(), PATH)
