from dataset import ShakespeareDataset, get_data
from torch.utils.data import DataLoader
from model import Mamba
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import argparse
from trainer import train, generate_text


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sequence_len", type=int, default=30)
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