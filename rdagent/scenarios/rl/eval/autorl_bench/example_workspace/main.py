import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset

def main():
    # Configuration
    model_path = "/models"
    save_path = "./trained_model"
    dataset_name = "imdb"  # Example dataset for training
    split = "train[:1%]"  # Use a small subset of data for simplicity
    max_input_length = 512
    max_train_steps = 1000

    # Load the tokenizer and the base model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Prepare the dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split=split)
    def preprocess_function(examples):
        inputs = examples["text"]
        inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")
        return inputs
    dataset = dataset.map(preprocess_function, batched=True)

    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1.41e-5,
        batch_size=16,
        log_with="tensorboard",  # Log training metrics
    )

    # Initialize PPO Trainer
    print("Initializing PPO Trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Training loop
    print("Starting training...")
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= max_train_steps:
            break

        # Forward pass through the model
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors)
        
        # Compute rewards (dummy reward for example purposes)
        rewards = torch.ones(response_tensors.shape[0])  # Replace with actual reward computation

        # Run a PPO training step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Print training stats
        if step % 10 == 0:
            print(f"Step {step}: {stats}")

    # Save the trained model
    print("Saving the model...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()