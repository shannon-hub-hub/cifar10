import torch.optim as optim
import time

print("CELL STARTED")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_cnn.parameters(), lr=0.01)
print(next(net_cnn.parameters()) is optimizer.param_groups[0]["params"][0])

num_epochs = 20

cnn_train_losses = []
cnn_val_losses = []
cnn_train_accuracies = []
cnn_val_accuracies = []
epoch_times = []

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)



for epoch in range(num_epochs):
    print("epoch loop entered", epoch)   

    epoch_start = time.time()
    net_cnn.train()
    running_loss = 0.0
    correct = 0
    total = 0

    ##### training
    print("about to start batch loop")
    for inputs, labels in trainloader:

        # move the data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients (clear old)
        optimizer.zero_grad()
        
        #forward + backward + optimize
        outputs = net_cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    #compute avg losses over all batches
    cnn_train_loss = running_loss / len(trainloader)
    cnn_train_accuracy = 100 * correct / total
    
    cnn_train_losses.append(cnn_train_loss)
    cnn_train_accuracies.append(cnn_train_accuracy)

    
    ##### validation
    net_cnn.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #forward + backward + optimize
            outputs = net_cnn(inputs)
            loss = criterion(outputs, labels)
            
            #print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cnn_val_loss = running_loss / len(valloader)
    cnn_val_accuracy = 100 * correct/ total

    cnn_val_losses.append(cnn_val_loss)
    cnn_val_accuracies.append(cnn_val_accuracy)
    
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(
        f"Epoch {epoch + 1} / {num_epochs} | "
        f"Train loss: {cnn_train_loss:.4f} | "
        f"Train accuracy: {cnn_train_accuracy:.2f}% | "
        f"Val loss: {cnn_val_loss:.4f} | "
        f"Val accuracy: {cnn_val_accuracy:.2f}% | "
        f"Time: {epoch_time:.1f}s"
    )
total_time = sum(epoch_times)
print(f"finished training in {total_time/60:.1f} min | avg epoch time: {total_time/len(epoch_times):.1f}s")

