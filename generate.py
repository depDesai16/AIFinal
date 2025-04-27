import torch
from model import TinyLLM
import random


#Pytorch and loading models:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
checkpoint = torch.load('tinyllm.pth')
stoi, itos = checkpoint['stoi'], checkpoint['itos']
vocab_size = len(stoi)

# Uses the same link as above to load and define the model class
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
model = TinyLLM(vocab_size)
model.load_state_dict(checkpoint['model'])
model.eval()



# https://chatgpt.com/c/680912d6-6f00-8012-aa28-e02acae3954e
    # and pulled from Lab 10 (Kaitlynn)
    # https://chatgpt.com/c/68063b56-3ce8-8000-9343-d16d959bbbed

def generate_text(prompt, length=100, k=10, temperature=1.0):
    # Tokenize the prompt into words
    input_ids = torch.tensor([stoi[word] for word in prompt.split() if word in stoi], dtype=torch.long).unsqueeze(0)
    hidden = None
    output_text = prompt

    for _ in range(length):
        logits, hidden = model(input_ids, hidden)
        logits = logits[0, -1] / temperature  # Apply temperature
        probs = torch.softmax(logits, dim=0)

        # Sample from the top-k tokens
        top_k_probs, top_k_indices = torch.topk(probs, k)
        next_id = top_k_indices[torch.multinomial(top_k_probs, num_samples=1).item()]
        next_word = itos[next_id.item()]

        output_text += ' ' + next_word

        # Update input for the next iteration
        input_ids = torch.tensor([[next_id]], dtype=torch.long)

    return output_text


        
if __name__ == "__main__":
    prompt = input("Prompt: ")
    print (generate_text(prompt))