import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import Protonet, load_protonet_conv
from data import load_data

def main():
    parser = argparse.ArgumentParser(description='Train a prototypical network')
    parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot'], help='which dataset to train on')
    parser.add_argument('--way', type=int, default=60, help='number of classes per episode')
    parser.add_argument('--shot', type=int, default=5, help='number of support examples per class')
    parser.add_argument('--query', type=int, default=5, help='number of query examples per class')
    parser.add_argument('--train_episodes', type=int, default=100, help='number of train episodes')
    parser.add_argument('--val_episodes', type=int, default=100, help='number of validation episodes')
    parser.add_argument('--test_episodes', type=int, default=1000, help='number of test episodes')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('--save_path', type=str, default='./experiments/best_model.pth', help='path to save the model')
    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(args, ['train', 'val'])

    # Load model
    model = load_protonet_conv(args.dataset, args.way).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # Training loop
    for epoch in range(args.train_episodes):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for i, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            # Extract data and target from batch
            data, target = batch['xs'], batch['xq']
            data, target = data.to(device), target.to(device)

            # Compute model output
            output = model(data)

            # Compute loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            acc = (predicted == target).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}/{args.train_episodes}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        total_val_acc = 0.0

        for batch in val_loader:
            data, target = batch['xs'], batch['xq']
            data, target = data.to(device), target.to(device)

            # Compute model output
            output = model(data)

            # Compute loss
            val_loss = criterion(output, target)

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            val_acc = (predicted == target).float().mean()

            total_val_loss += val_loss.item()
            total_val_acc += val_acc.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        print(f"Validation, Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
