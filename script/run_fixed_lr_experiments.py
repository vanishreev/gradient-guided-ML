
# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# TASK: AUTOMATED FIXED LEARNING-RATE EXPERIMENTS
# ============================================================

# PURPOSE:
# Investigate how different fixed learning rates affect
# neural network training with configurable hidden neurons.
#
# RESEARCH QUESTIONS:
# 1. Which fixed learning rate produces the best accuracy?
# 2. Does increasing hidden neurons change the optimal LR?
# 3. Which configuration reaches target accuracy fastest?
#
# EXPERIMENTAL CONTROLS:
# - Same MNIST train/validation split.
# - Same initial weights within each architecture and seed.
# - Same training mini-batch order for all learning rates.
# - Same batch size and epoch budget.
#
# EXPECTED RESULTS:
# - Epoch-level training loss and validation accuracy.
# - Best and final validation accuracy.
# - Training steps and elapsed time to target accuracy.
# - Model weights and CSV results saved separately.
#
# IMPORTANT:
# The MNIST test dataset remains untouched.


# ============================================================
# STEP 01: IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path
import csv
import json
import time

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# STEP 02: CONFIGURE EXPERIMENT SETTINGS
# ============================================================

# PURPOSE:
# Keep experiment settings together so the training
# implementation does not need to change.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# RANDOM_SEED = 42
# RANDOM_SEED = 123
#RANDOM_SEED = 456
#RANDOM_SEED = 789
RANDOM_SEED = 2026

BATCH_SIZE = 64
EPOCHS = 10
TARGET_ACCURACY = 0.97

# CHANGE THIS VALUE TO TEST DIFFERENT HIDDEN NEURON COUNTS.
HIDDEN_NEURONS = 256

# CHANGE ONLY THIS LIST TO TEST DIFFERENT LEARNING RATES.
LEARNING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00
]

# Save the results separately for each architecture.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "fixed_lr_experiments"
    / f"hidden_{HIDDEN_NEURONS}"
    / f"seed_{RANDOM_SEED}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 03: LOAD AND PREPARE MNIST
# ============================================================

# PURPOSE:
# Load normalized MNIST images and reproduce the same
# training/validation split from previous experiments.
#
# DATASET SPLIT:
# Training: 48,000 images.
# Validation: 12,000 images.
# Test: Reserved and not used here.

image_transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=image_transform
)

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_subset, validation_subset = random_split(
    full_train_dataset,
    [48000, 12000],
    generator=split_generator
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

assert len(train_subset) == 48000
assert len(validation_subset) == 12000

print("\nMNIST DATASET VERIFICATION")
print("Training images:", len(train_subset))
print("Validation images:", len(validation_subset))


# ============================================================
# STEP 04: DEFINE THE NEURAL NETWORK ARCHITECTURE
# ============================================================

# PURPOSE:
# Build the same type of neural network as before,
# but allow the hidden layer size to be configured.
#
# ARCHITECTURE:
# Input layer: 784 normalized image pixels.
# Hidden layer: HIDDEN_NEURONS with ReLU activation.
# Output layer: 10 digit classification scores.
#
# WHY:
# The number of hidden neurons changes model capacity.
# We want to investigate how model size affects training.

class DigitClassifier(nn.Module):

    def __init__(self, hidden_neurons):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# ============================================================
# STEP 05: CREATE REPRODUCIBLE INITIAL WEIGHTS
# ============================================================

# PURPOSE:
# Generate one set of initial weights for the selected
# architecture and reuse it for every learning rate.
#
# WHY:
# Previously saved 128-neuron weights cannot be loaded
# into a 256-neuron model.
#
# FAIR COMPARISON:
# Every learning-rate experiment with the same hidden
# neuron count must begin with identical weights.

torch.manual_seed(RANDOM_SEED)

initial_model = DigitClassifier(HIDDEN_NEURONS)

# Save independent copies of the initial parameter tensors.
initial_state = {
    name: value.detach().clone()
    for name, value in initial_model.state_dict().items()
}

INITIALIZATION_DIR = (
    PROJECT_ROOT
    / "output"
    / "initialization"
    / f"hidden_{HIDDEN_NEURONS}"
    / f"seed_{RANDOM_SEED}"
)

INITIALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

initial_weights_path = (
    INITIALIZATION_DIR / "mnist_initial_weights.pt"
)

torch.save(
    initial_state,
    initial_weights_path
)

total_parameters = sum(
    parameter.numel()
    for parameter in initial_model.parameters()
)

print("\nMODEL INITIALIZATION")
print("Hidden neurons:", HIDDEN_NEURONS)
print("Trainable parameters:", total_parameters)
print("Initial weights saved:", initial_weights_path)


# ============================================================
# STEP 06: CALCULATE VALIDATION ACCURACY
# ============================================================

# PURPOSE:
# Evaluate the model on validation data without modifying
# its parameters.
#
# Validation accuracy tells us how well the model predicts
# handwritten digits it did not train on.

def calculate_validation_accuracy(model, loader):

    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            predicted_labels = outputs.argmax(dim=1)

            correct_predictions += (
                predicted_labels == labels
            ).sum().item()

            total_predictions += labels.size(0)

    return correct_predictions / total_predictions


# ============================================================
# STEP 07: DEFINE REUSABLE FIXED-SGD TRAINING
# ============================================================

# PURPOSE:
# Train one neural network with a chosen fixed learning rate.
#
# Every experiment:
# - Uses the same initial weights.
# - Uses the same training data and batch order.
# - Uses the same loss function and architecture.
# - Trains for the same number of epochs.
#
# Only the learning rate changes.

def run_experiment(learning_rate):

    print("\n" + "=" * 65)
    print(f"FIXED SGD | LR = {learning_rate}")
    print(f"Hidden neurons = {HIDDEN_NEURONS}")
    print("=" * 65)

    # STEP 07A:
    # Create a fresh model for this learning rate.

    model = DigitClassifier(HIDDEN_NEURONS)

    # STEP 07B:
    # Load the shared initial parameters.
    # This guarantees the same starting point.

    model.load_state_dict(initial_state)

    # STEP 07C:
    # Reset the batch-shuffling generator so that
    # every LR receives the same mini-batch order.

    train_generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0
    )

    # STEP 07D:
    # Configure cross-entropy loss and plain SGD.

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # STEP 07E:
    # Prepare metrics and start the timer.

    history = []

    start_time = time.perf_counter()

    # ========================================================
    # STEP 08: TRAIN FOR THE SPECIFIED NUMBER OF EPOCHS
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        total_training_loss = 0.0
        total_training_samples = 0

        for images, labels in train_loader:

            # Forward pass: predict digit classes.
            predictions = model(images)

            # Calculate the prediction error.
            loss = loss_function(predictions, labels)

            # Remove gradients from the previous batch.
            optimizer.zero_grad()

            # Backpropagation: calculate current gradients.
            loss.backward()

            # Update weights using the fixed learning rate.
            optimizer.step()

            # Accumulate loss weighted by batch size.
            current_batch_size = labels.size(0)

            total_training_loss += (
                loss.item() * current_batch_size
            )

            total_training_samples += current_batch_size

        # STEP 08A:
        # Calculate the average training loss.

        average_loss = (
            total_training_loss / total_training_samples
        )

        # STEP 08B:
        # Calculate validation accuracy.

        validation_accuracy = calculate_validation_accuracy(
            model,
            validation_loader
        )

        # STEP 08C:
        # Record elapsed time after validation.

        elapsed_time = time.perf_counter() - start_time

        completed_steps = (
            (epoch + 1) * len(train_loader)
        )

        history.append({
            "epoch": epoch + 1,
            "completed_steps": completed_steps,
            "learning_rate": learning_rate,
            "training_loss": average_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": elapsed_time
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {learning_rate:.3f} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {validation_accuracy:.2%} | "
            f"Steps: {completed_steps} | "
            f"Time: {elapsed_time:.2f}s"
        )

    # ========================================================
    # STEP 09: SAVE EXPERIMENT RESULTS
    # ========================================================

    # PURPOSE:
    # Preserve the experiment's training metrics and
    # final model weights for future analysis.

    lr_label = (
        f"{learning_rate:.5f}".rstrip("0").rstrip(".")
    )

    metrics_path = (
        OUTPUT_DIR / f"LR_{lr_label}_metrics.csv"
    )

    with open(metrics_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

    model_path = (
        OUTPUT_DIR / f"LR_{lr_label}_model.pt"
    )

    torch.save(model.state_dict(), model_path)

    # ========================================================
    # STEP 10: CALCULATE EXPERIMENT SUMMARY
    # ========================================================

    # PURPOSE:
    # Identify the best validation accuracy and measure
    # efficiency using a common target accuracy.
    #
    # LIMITATION:
    # Validation is measured only once per epoch, so
    # target steps represent the first recorded qualifying
    # epoch rather than the exact first qualifying batch.

    best_result = max(
        history,
        key=lambda row: row["validation_accuracy"]
    )

    target_result = next(
        (
            row for row in history
            if row["validation_accuracy"] >= TARGET_ACCURACY
        ),
        None
    )

    return {
        "hidden_neurons": HIDDEN_NEURONS,
        "random_seed": RANDOM_SEED,
        "learning_rate": learning_rate,
        "final_accuracy": history[-1]["validation_accuracy"],
        "best_accuracy": best_result["validation_accuracy"],
        "best_epoch": best_result["epoch"],
        "final_loss": history[-1]["training_loss"],
        "total_seconds": history[-1]["elapsed_seconds"],
        "target_accuracy": TARGET_ACCURACY,
        "target_epoch": (
            target_result["epoch"]
            if target_result is not None else None
        ),
        "target_steps": (
            target_result["completed_steps"]
            if target_result is not None else None
        ),
        "target_seconds": (
            target_result["elapsed_seconds"]
            if target_result is not None else None
        )
    }


# ============================================================
# STEP 11: RUN ALL FIXED LEARNING-RATE EXPERIMENTS
# ============================================================

# PURPOSE:
# Automatically train a fresh model for every learning
# rate, then rank the completed experiments.

def main():

    experiment_results = []

    print("\nSTARTING FIXED LEARNING-RATE EXPERIMENTS")

    for learning_rate in LEARNING_RATES:

        result = run_experiment(learning_rate)

        experiment_results.append(result)

    # ========================================================
    # STEP 12: RANK RESULTS
    # ========================================================

    ranked_results = sorted(
        experiment_results,
        key=lambda row: row["final_accuracy"],
        reverse=True
    )

    print("\n" + "=" * 80)
    print("FIXED LEARNING-RATE EXPERIMENT COMPARISON")
    print("=" * 80)

    for result in ranked_results:

        target_steps = (
            str(result["target_steps"])
            if result["target_steps"] is not None
            else "Not reached"
        )

        target_time = (
            f"{result['target_seconds']:.2f}s"
            if result["target_seconds"] is not None
            else "Not reached"
        )

        print(
            f"LR: {result['learning_rate']:.3f} | "
            f"Final: {result['final_accuracy']:.2%} | "
            f"Best: {result['best_accuracy']:.2%} | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Steps to 97%: {target_steps} | "
            f"Time to 97%: {target_time} | "
            f"Total: {result['total_seconds']:.2f}s"
        )

    # ========================================================
    # STEP 13: SAVE CONSOLIDATED RESULTS
    # ========================================================

    summary_path = OUTPUT_DIR / "fixed_lr_summary.csv"

    with open(summary_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(ranked_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(ranked_results)

    # Save the experiment settings for reproducibility.
    config = {
        "hidden_neurons": HIDDEN_NEURONS,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "target_accuracy": TARGET_ACCURACY,
        "learning_rates": LEARNING_RATES,
        "architecture": (
            f"784-{HIDDEN_NEURONS}-ReLU-10"
        ),
        "optimizer": "SGD",
        "dataset": "MNIST",
        "validation_frequency": "once per epoch",
        "pytorch_version": torch.__version__
    }

    config_path = OUTPUT_DIR / "experiment_config.json"

    with open(config_path, "w") as file:

        json.dump(config, file, indent=4)

    print("\nResults saved to:", OUTPUT_DIR)
    print("TASK: FIXED SGD EXPERIMENTS COMPLETED")


# ============================================================
# STEP 14: EXECUTE EXPERIMENTS
# ============================================================

if __name__ == "__main__":
    main()