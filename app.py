import os
import torch
import numpy as np
import torch.nn as nn
import streamlit as st
from PIL import Image

# Define the CharRNN model class
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout_rate=0.3):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout_rate = dropout_rate

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = output.contiguous().view(-1, output.shape[2])
        logits = self.fc(output)
        return logits, hidden

# Loading the model
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = CharRNN(checkpoint['vocab_size'],
                            checkpoint['embed_size'],
                            checkpoint['hidden_size'],
                            checkpoint['num_layers'],
                            dropout_rate=checkpoint['dropout_rate']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    return loaded_model, char2idx, idx2char

# Define NEWLINE_TOKEN
NEWLINE_TOKEN = "<NEWLINE>"

# Text Generation
def generate_text(model, start_text, char2idx, idx2char, generation_length=200, temperature=0.8):
    model.eval()
    input_indices = [char2idx.get(ch, 0) for ch in start_text]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    hidden = None
    generated_text = start_text

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_tensor, hidden)
            logits = logits[-1] / temperature
            probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)

            next_char = idx2char[next_char_idx]
            generated_text += next_char

            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    # Replace <NEWLINE> token with actual newline
    generated_text = generated_text.replace(NEWLINE_TOKEN, '\n')

    return generated_text


# Load the *best* model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'my_char_rnn_model.pth'  # Ensure this is the correct path to your model file
loaded_model, loaded_char2idx, loaded_idx2char = load_model(model_path, device)

# Streamlit UI
st.title("✨ Poetica: The AI Poetry Generator ")


# Take user input for the prompt text
start_text = st.text_input("Enter the starting word or line for the poem:", "pyar")

# Slider for controlling the length of the generated text
generation_length = st.slider("Length of the generated text:", min_value=50, max_value=500, value=200)

# Slider for controlling the temperature (controls creativity)
temperature = st.slider("Temperature (controls randomness):", min_value=0.1, max_value=1.5, value=0.7)

# Button to generate poetry
if st.button("Generate Poetry"):
    with st.spinner("Generating..."):
        generated_poetry = generate_text(loaded_model, start_text, loaded_char2idx, loaded_idx2char, generation_length, temperature)
    
    st.text(generated_poetry)

    # Button to download the generated poetry
    st.download_button(
        label="Download Poetry",
        data=generated_poetry,
        file_name="generated_poetry.txt",
        mime="text/plain"
    )

# Reset button to clear inputs
if st.button("Reset"):
    st.experimental_rerun()
