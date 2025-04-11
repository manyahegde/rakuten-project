How to Run the System:

1. **Set up the environment**:
    - Install the necessary dependencies by running the following command:
        
        ```
        pip install -r requirements.txt
        ```
        
2. **Train the model (optional)**:
    - If you want to optimize the model and apply pruning, run the following command:
        
        ```
        python pso.py
        ```
        
    - This will use Particle Swarm Optimization (PSO) to find the best hyperparameters and apply L1 pruning to the TinyBERT model, saving the optimized model in the `pruned_model/` folder.
3. **Run the server**:
    - Start the server by running the following command:
        
        ```
        python server.py
        ```
        
    - This will initialize the server, which listens for incoming client connections on port `10300`.
4. **Run the client**:
    - Start the client by running the following command:
        
        ```
        python client.py
        ```
        
    - This will launch a Gradio interface where you can input text for sentiment classification.
    - The client will send the tokenized text to the server, which will process it and send back the sentiment classification ("Positive" or "Negative").

Example Workflow:

1. **Client** sends a sentence like "I love this movie!" to the server.
2. **Server** processes the intermediate output, classifies the sentiment, and sends back "Positive".
3. **Client** displays the result: "Positive".
