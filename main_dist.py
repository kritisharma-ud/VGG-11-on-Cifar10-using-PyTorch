import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel
import model as mdl
import torch.distributed as dist
from torch.multiprocessing import Process

def train_model(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #if batch_idx % 20 == 0:
        if batch_idx % 40 == 39:    
            print(f'Epoch [{epoch + 1}, Batch {batch_idx}] Loss: {running_loss / 20:.3f}')
            running_loss = 0.0

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader, 1):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

    return accuracy

def main(rank, world_size):
    learning_rate = 0.01
    weight_decay = 1e-4
    batch_size = 256

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the device
    model = mdl.VGG11()
    model = DistributedDataParallel(model)  # Wrap the model with DistributedDataParallel
    model.to(device)

    # Define the optimizer, criterion, and data loaders
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    training_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # Use DistributedSampler to distribute the dataset among the workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(training_set, num_workers=2, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    
    # Test set does not need to be distributed
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=batch_size, shuffle=False, pin_memory=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(1):
        train_model(model, train_loader, optimizer, criterion, epoch, device)
        test_acc = test_model(model, test_loader, criterion, device)
        scheduler.step(test_acc)

    if test_acc is not None:
        print(f"Accuracy achieved: {test_acc:.2f}%")

if __name__ == "__main__":
    # Initialing environment
    dist.init_process_group(backend='gloo')
    
    # getting current process rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # calling with rank and world sizr
    main(rank, world_size)
