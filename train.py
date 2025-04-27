import torch 
import torch.nn as nn
import torch.optim as optim
from model import TinyLLM

from tqdm import trange

# import os allows for python to interact with the operating system 
# as in file manipulation, process management, and environment interactions
    # https://docs.python.org/3/library/os.html
    # https://www.geeksforgeeks.org/os-module-python-examples/
import os

# Pickle allows for serialization and deserialization of Python objects (turns things into a byte stream)
# https://docs.python.org/3/library/pickle.html
    # - Kaitlynn 
import pickle

with open('data.txt', 'r') as f:
    text = f.read().lower()

    text = text.replace('\n', ' ').replace('\r', '').strip()

words = text.split()
stoi = {word: i for i, word in enumerate(set(words))}
itos = {i: word for word, i in stoi.items()}

def encode(s):
    return [stoi[word] for word in s.split()]

def decode(l):
    return ' '.join([itos[i] for i in l])

vocab_size = len(stoi)
data = torch.tensor(encode(text), dtype=torch.long)



#Training the model
embedding_dim = 2048
hidden_dim = 4096
seq_len = 50
batch_size = 32
epochs = 2000
lr = 0.00005

model = TinyLLM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def get_batch():
    idx = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in idx])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in idx])
    return x, y

print(" Starting training...\n")
print(" Training data preview:", text[:100])

for epoch in trange(epochs, desc="Training"):
    x, y = get_batch()
    model.train()
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print loss for the current epoch
    print(f" Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

# Save only the final model
torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos}, 'tinyllm.pth')
print("\n Training complete! Model saved to tinyllm.pth")


# Got the model generateing text, and then had to find out how to get it to generate non-gibberish
    # https://chatgpt.com/c/68092126-4b64-8012-b1e5-23dfc756c5a3

# Kinda was just using random text to test the model
# https://chatgpt.com/c/680933ce-50dc-8012-b586-e73395cd01d1