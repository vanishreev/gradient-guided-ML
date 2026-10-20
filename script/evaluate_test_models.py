
# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# PHASE 2: HELD-OUT TEST DATA EVALUATION
# ============================================================

# PURPOSE:
# Evaluate our three selected optimization methods on the
# original 10,000 MNIST test images.
#
# RESEARCH QUESTION:
# Do models selected using validation accuracy generalize
# to previously unseen test data?
#
# IMPORTANT:
# - Do not retrain the models.
# - Do not update their weights.
# - Do not change the learning rates based on test results.
# - Use exactly the same test images for all three models.


# ============================================================
# STEP 01: IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import csv

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# STEP 02: DEFINE EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

HIDDEN_NEURONS = 256
RANDOM_SEED = 42
BATCH_SIZE = 64

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "test_evaluation"
    / f"hidden_{HIDDEN_NEURONS}"
    / f"seed_{RANDOM_SEED}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 03: IDENTIFY THE THREE SAVED MODELS
# ============================================================

# PURPOSE:
# Locate model checkpoints produced by our completed
# Fixed, Scheduled, and Gradient-Guided SGD experiments.
#
# If your saved folder structure differs, update only
# the corresponding model path below.

MODEL_PATHS = {
    "Fixed SGD (LR 0.60)": (
        PROJECT_ROOT
        / "output"
        / "fixed_lr_experiments"
        / "hidden_256"
        / "seed_42"
        / "LR_0.6_model.pt"
    ),

    "Scheduled SGD (S13)": (
        PROJECT_ROOT
        / "output"
        / "scheduled_lr_experiments"
        / "hidden_256"
        / "seed_42"
        / "S13"
        / "model.pt"
    ),

    "Gradient-Guided SGD": (
        PROJECT_ROOT
        / "output"
        / "gradient_guided"
        / "hidden_256"
        / "seed_42"
        / "GG_STR0.500_WEAK0.375_NEG0.500_THR0.600"
        / "guided_model.pt"
    )
}


# ============================================================
# STEP 04: DEFINE THE SAME NEURAL NETWORK ARCHITECTURE
# ============================================================

# PURPOSE:
# Recreate the exact architecture used during training.
#
# The architecture must match the checkpoint dimensions:
# 784 input features -> 256 hidden neurons -> 10 outputs.

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
        return self.network(self.flatten(x))


# ============================================================
# STEP 05: LOAD THE MNIST TEST DATASET
# ============================================================

# PURPOSE:
# Load the original MNIST test split.
#
# train=False is essential:
# It loads the 10,000-image test dataset, not the data
# previously used for training and validation.
#
# Use ToTensor() to match training preprocessing.

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

assert len(test_dataset) == 10000

print("\nMNIST TEST DATASET")
print("Test images:", len(test_dataset))


# ============================================================
# STEP 06: DEFINE TEST EVALUATION FUNCTION
# ============================================================

# PURPOSE:
# Measure classification accuracy and cross-entropy loss.
#
# model.eval():
# Switches the neural network into evaluation mode.
#
# torch.no_grad():
# Prevents gradient calculation and weight updates.
#
# We do not use optimizer.step() anywhere in this script.

def evaluate_model(model):

    model.eval()

    loss_function = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            batch_loss = loss_function(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += batch_loss.item()

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_predictions += labels.size(0)

    test_accuracy = (
        correct_predictions / total_predictions
    )

    average_test_loss = (
        total_loss / total_predictions
    )

    return {
        "test_accuracy": test_accuracy,
        "test_loss": average_test_loss,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }


# ============================================================
# STEP 07: EVALUATE ALL THREE SAVED MODELS
# ============================================================

# PURPOSE:
# Load each trained model and evaluate it on exactly
# the same 10,000 test images.

results = []

for method_name, model_path in MODEL_PATHS.items():

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    model = DigitClassifier(HIDDEN_NEURONS)

    saved_weights = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(saved_weights)

    metrics = evaluate_model(model)

    results.append({
        "method": method_name,
        "hidden_neurons": HIDDEN_NEURONS,
        "random_seed": RANDOM_SEED,
        **metrics
    })

    print("\n" + "=" * 60)
    print(method_name)
    print("=" * 60)

    print(
        f"Test accuracy: "
        f"{metrics['test_accuracy']:.2%}"
    )

    print(
        f"Test loss: "
        f"{metrics['test_loss']:.4f}"
    )

    print(
        f"Correct predictions: "
        f"{metrics['correct_predictions']}/"
        f"{metrics['total_predictions']}"
    )


# ============================================================
# STEP 08: SAVE THE TEST RESULTS
# ============================================================

# PURPOSE:
# Preserve the final test results independently from
# training and validation experiment files.

results_path = OUTPUT_DIR / "test_results.csv"

with open(results_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# STEP 09: PRINT FINAL COMPARISON
# ============================================================

# PURPOSE:
# Display all test results together.
#
# IMPORTANT:
# This is a final held-out evaluation.
# Do not tune models using these test results.

print("\n" + "=" * 75)
print("FINAL MNIST TEST DATA COMPARISON")
print("=" * 75)

for result in results:

    print(
        f"{result['method']} | "
        f"Accuracy: {result['test_accuracy']:.2%} | "
        f"Loss: {result['test_loss']:.4f} | "
        f"Correct: {result['correct_predictions']}/10000"
    )

print("\nSaved results:", results_path)

print("\nPHASE 2: HELD-OUT TEST EVALUATION COMPLETED")