from transformers import BertTokenizer, BertModel
import torch

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Hello, my name is BERT."
inputs = tokenizer(text, return_tensors='pt')

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    
# Extract the embeddings from the last hidden state
last_hidden_state = outputs.last_hidden_state

print("Shape of last hidden state:", last_hidden_state.shape)
print("Embedding for [CLS] token:", last_hidden_state[0, 0, :])
