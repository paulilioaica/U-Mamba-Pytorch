import torch 
from torch.utils import clip_grad_norm_
from tqdm import tqdm 

def train(model, num_epochs, dataloader, optimizer, criterion, vocab_size, device):
    for epoch in range(num_epochs):  
        running_loss = 0.0  
        for i, (x, y) in tqdm(enumerate(dataloader)):  
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)  
            output = model(x)  
            loss = criterion(output.view(-1, vocab_size), y.view(-1).cuda())  

            loss.backward()  
            clip_grad_norm_(model.parameters(),max_norm=1)

            optimizer.step()  
            running_loss += loss.item()  
        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader)}")  



def generate_text(model, start_string, char_to_index, index_to_char, gen_length=100, temperature=0.1, ):
    model.eval()
    with torch.no_grad():
        input_text = start_string
        for _ in range(gen_length):
            input_tensor = torch.tensor([char_to_index[c] for c in input_text], dtype=torch.long).unsqueeze(0).cuda()
            output = model(input_tensor)
            output = output / temperature
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_char_index = torch.multinomial(probs, num_samples=1).squeeze().item()
            input_text += index_to_char[next_char_index]
    return input_text
