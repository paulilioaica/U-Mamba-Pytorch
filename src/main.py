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



def run(batch_size, sequence_len, learning_rate, 
        num_epochs, num_layers, hidden_size, rank,
        state_size, kernel_size, device):
        

        text, chars, char_to_index, index_to_char = get_data()
        
        dataset = ShakespeareDataset(text, char_to_index=char_to_index, sequence_len=sequence_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = Mamba(hidden_size, len(chars), rank, state_size, kernel_size, num_layers).cuda()
        model = model.to(device)

        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        model = train(model, num_epochs, dataloader, optimizer, criterion, len(chars), device)

        generate_text(model, "ROMEO:", char_to_index, index_to_char, gen_length=200, temperature=0.1)
        

if __name__ == "__main__":
    run(args.batch_size, 
        args.sequence_len, 
        args.learning_rate, 
        args.num_epochs, 
        args.num_layers, 
        args.hidden_size, 
        args.rank, 
        args.state_size, 
        args.kernel_size,
        args.device)