# Server Side Code (modelpart2)
import socket
import torch
import pickle
from transformers import BertForSequenceClassification
import torch.nn as nn

# Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'
port = 5000
server_socket.bind((host, port))
server_socket.listen(1)
print(f"Server listening on {host}:{port}...")

# Load TinyBERT model
tinybert = BertForSequenceClassification.from_pretrained("./pruned_model/")


# Define modelpart2 to include remaining encoder layers, pooler, and classifier
modelpart2_layers = nn.ModuleList([
    *tinybert.bert.encoder.layer[1:],  # Remaining encoder layers after the first
    tinybert.bert.pooler                # Pooling layer
])

# Combine modelpart2_layers and classifier into a single module
class ModelPart2(nn.Module):
    def __init__(self, layers, classifier):
        super(ModelPart2, self).__init__()
        self.layers = layers
        self.classifier = classifier

    def forward(self, hidden_states):
        # Pass through each encoder layer in modelpart2
        for layer in self.layers[:-1]:  # Exclude the pooling layer for now
            hidden_states = layer(hidden_states)[0]
       
        # Now, use the last layer's output for pooling
        pooled_output = self.layers[-1](hidden_states)  # Pooling layer output
       
        # Classification
        output = self.classifier(pooled_output)  # Classification layer output
        return output

# Instantiate modelpart2
modelpart2 = ModelPart2(modelpart2_layers, tinybert.classifier)

while True:
    # Accept connection from client
    client_socket, client_address = server_socket.accept()
    print(f"Connected to client at {client_address}")

    try:
        # Receive the intermediate tensor from client
        data = b''
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        if data:
            print("Intermediate output received from client.")
            # Deserialize the received data
            intermediate_output = pickle.loads(data)
            print(f"Intermediate output shape: {intermediate_output.shape}")

            # Forward pass through modelpart2
            with torch.no_grad():
                hidden_states = intermediate_output  # Start from the intermediate output received
                output = modelpart2(hidden_states)  # Get the final output

            # Get predictions
            predictions = torch.argmax(output, dim=1)  # Predicted class indices
            print(predictions)
            # Serialize and send predictions back to client
            serialized_predictions = pickle.dumps(predictions)
            client_socket.sendall(serialized_predictions)
            print("Predictions sent back to client.")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Close the socket connection for the client
        client_socket.close()
        #print("Connection closed.")
