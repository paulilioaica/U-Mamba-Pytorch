from dataset import ShakespeareDataset, get_data
from torch.utils.data import DataLoader
from u_mamba import U_Mamba
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import argparse
from trainer import train_on_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--height", type=int, default=32)
parser.add_argument("--target_classes", type=int, default=2)

parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--rank", type=int, default=3)
parser.add_argument("--state_size", type=int, default=10)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()


#TBD
#from test.ipynb
def run(batch_size, width, height, target_classes, learning_rate, num_epochs, num_layers, hidden_size, rank, state_size, kernel_size, device):
    train_dataset, test_dataset = get_data()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = U_Mamba(channels=3, width=width, height=height, hidden_size=hidden_size, rank=rank, state_size=state_size, kernel_size=kernel_size, num_layers=num_layers, target_classes=target_classes)
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = train_on_dataset(model, train_dataloader, test_dataloader, optimizer, criterion, device, num_epochs)
    return model