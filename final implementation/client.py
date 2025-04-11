import socket
import torch
import pickle
import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn as nn

# Load TinyBERT model and tokenizer
tinybert = BertForSequenceClassification.from_pretrained("./pruned_model/")
tokenizer = BertTokenizer.from_pretrained("./pruned_model/")

# Define modelpart1 (Embeddings and first encoder layer)
modelpart1 = nn.ModuleList([
    tinybert.bert.embeddings,
    tinybert.bert.encoder.layer[0]  # First encoder layer (Layer 0)
])

# Function to process and send input to server
def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True)

    # Pass through modelpart1
    with torch.no_grad():
        # Get embeddings
        hidden_states = modelpart1[0](inputs['input_ids'])
        # Pass through the first encoder layer (Layer 0)
        hidden_states = modelpart1[1](hidden_states)[0]  # Only Layer 0

    # Serialize the output and send it to the server
    data = pickle.dumps(hidden_states)

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('192.168.76.80', 10300))  # Use the server's IP and port
        client_socket.sendall(data)
        client_socket.shutdown(socket.SHUT_WR)  # Indicate that sending is complete

        print("Intermediate output sent to server.")

        # Receive the predictions from the server
        received_data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            received_data += packet

        # Deserialize the received predictions
        predictions = pickle.loads(received_data)
        if predictions.item() == 0:
            return "Negative"
        else:
            return "Positive"

    except Exception as e:
        print(f"Error: {e}")
        return "Error connecting to the server."

    finally:
        client_socket.close()


# Set up Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="text",
    title="TinyBERT Text Classification with Server Communication",
    description="Enter a sentence to classify its sentiment using TinyBERT."
)

iface.launch()