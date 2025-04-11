import numpy as np
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pyswarm import pso

# Load IMDB dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

def tokenize_function(examples):
    """
    Tokenizes the input text data by padding and truncating to a max length of 128 tokens.
    
    Args:
    - examples: Dictionary containing the input text data from the IMDB dataset.
    
    Returns:
    - A dictionary containing tokenized data.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"]).rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Split into train and test datasets
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

def get_model():
    """
    Loads the TinyBERT model for sequence classification.
    
    Returns:
    - A model instance of type AutoModelForSequenceClassification.
    """
    return AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)

def compute_metrics(pred):
    """
    Computes evaluation metrics including accuracy, precision, recall, and F1 score.
    
    Args:
    - pred: The predictions object from the Trainer class, containing true labels and predicted labels.
    
    Returns:
    - A dictionary with evaluation metrics (accuracy, precision, recall, F1).
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def evaluate_model(hyperparameters):
    """
    Evaluates the model performance for a given set of hyperparameters.
    
    Args:
    - hyperparameters: A tuple containing learning rate and number of epochs.
    
    Returns:
    - A value representing the loss (1 - accuracy) to minimize in PSO.
    """
    learning_rate, num_epochs = hyperparameters
    model = get_model()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=int(num_epochs),
        weight_decay=0.01,
        logging_steps=10,
        disable_tqdm=True,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return 1 - eval_result["eval_accuracy"]  # We minimize (1 - accuracy)

# PSO bounds and run configuration
lb = [1e-5, 1]   # Lower bounds for learning rate and num_epochs
ub = [5e-4, 5]   # Upper bounds for learning rate and num_epochs

# Apply Particle Swarm Optimization to find the best hyperparameters
best_hyperparameters, _ = pso(evaluate_model, lb, ub, swarmsize=10, maxiter=5)
best_learning_rate, best_num_epochs = best_hyperparameters

# Train final model with best hyperparameters
model = get_model()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=best_learning_rate,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=int(best_num_epochs),
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

def apply_l1_pruning(model, amount=0.4):
    """
    Applies L1 pruning to all Linear layers of the model to reduce the number of parameters.
    
    Args:
    - model: The model to prune.
    - amount: Fraction of the parameters to prune (between 0 and 1).
    
    Returns:
    - The pruned model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

# Apply L1 pruning
pruned_model = apply_l1_pruning(model, amount=0.4)

# Re-evaluate the pruned model
trainer = Trainer(
    model=pruned_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

final_metrics = trainer.evaluate()

# Print performance metrics
print("\nPerformance with optimized hyperparameters and L1 pruning:")
print(f"Learning Rate: {best_learning_rate}")
print(f"Number of Epochs: {int(best_num_epochs)}")
print(f"Accuracy: {final_metrics['eval_accuracy']:.4f}")
print(f"Loss: {final_metrics['eval_loss']:.4f}")
print(f"Precision: {final_metrics['eval_precision']:.4f}")
print(f"Recall: {final_metrics['eval_recall']:.4f}")
print(f"F1 Score: {final_metrics['eval_f1']:.4f}")

# Save the pruned model
pruned_model.save_pretrained("./pruned_model")
