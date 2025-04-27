import torch
import torch.nn as nn

# Pickle allows for serialization and deserialization of Python objects (turns things into a byte stream)
# https://docs.python.org/3/library/pickle.html
    # - Kaitlynn 
import pickle



# Defines the TinyLLM model
    # Initizlizes the model with vocab_size, embedding_dim, and hidden_dim
class TinyLLM(nn.Module):
    def __init__ (self, vocab_size, embedding_dim= 64, hidden_dim=128):
        super(TinyLLM, self).__init__()

        # Maps each character to a unique index
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # The RNN layer will process sequences of embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Maps the hidden state of the RNN to the vocabulary size (one per character)
        self.fc = nn.Linear(hidden_dim, vocab_size)


# Similiar to Kaitlynn's Lab10 forward method 
    # In which a Stack Overflow was used to help with that
    # https://stackoverflow.com/questions/60713781/pytorch-device-and-todevice-method
    def forward(self, x, hidden=None):
        # Convert to embeddings
        x = self.embed(x)

        # Run through RNN
        output, hidden = self.rnn(x, hidden)

        # Map to vocab predictions
        logits = self.fc(output)
        return logits, hidden
    
