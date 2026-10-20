
# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# TASK: AUTOMATED SCHEDULED SGD EXPERIMENTS
# ============================================================

# PURPOSE:
# Investigate how different learning-rate reduction schedules
# affect neural network training performance.
#
# RESEARCH QUESTIONS:
# 1. Can Scheduled SGD outperform a well-tuned Fixed SGD?
# 2. When should the learning rate be reduced?
# 3. How much should the learning rate be reduced?
# 4. Does the number of hidden neurons affect the results?
#
# EXPERIMENTAL CONTROLS:
# - Same MNIST training/validation split.
# - Same architecture for experiments with equal neuron count.
# - Same initial weights as the updated Fixed SGD script.
# - Same mini-batch order and random seed.
# - Same batch size and epoch budget.
#
# EXPECTED RESULTS:
# - Training loss and validation accuracy for every epoch.
# - Best and final validation accuracy.
# - Steps and time to reach a target validation accuracy.
# - Saved model checkpoints and experiment summaries.
#
# IMPORTANT:
# We will not use the untouched MNIST test set.


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
# Control the experiment without editing the training loop.
#
# IMPORTANT:
# Match HIDDEN_NEURONS and RANDOM_SEED with the Fixed SGD
# experiment when comparing their results.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

#RANDOM_SEED = 42
#RANDOM_SEED = 123
#RANDOM_SEED = 456
#RANDOM_SEED = 789
RANDOM_SEED = 2026
BATCH_SIZE = 64
EPOCHS = 10
TARGET_ACCURACY = 0.97

# CHANGE THIS TO TEST DIFFERENT HIDDEN LAYER SIZES.
HIDDEN_NEURONS = 256

# SCHEDULE CONFIGURATION:
#
# initial_lr:
# Learning rate used at the beginning of training.
#
# reduce_after:
# Number of completed epochs before changing the rate.
# Example: 7 means Epochs 1-7 use initial_lr and
# Epochs 8-10 use new_lr.
#
# new_lr:
# Learning rate for the remaining epochs.
#
# MODIFY ONLY THIS LIST TO TEST OTHER SCHEDULES.

SCHEDULES = [
    {
        "id": "S11",
        "initial_lr": 0.50,
        "reduce_after": 3,
        "new_lr": 0.25
    },
    {
        "id": "S12",
        "initial_lr": 0.50,
        "reduce_after": 5,
        "new_lr": 0.25
    },
    {
        "id": "S13",
        "initial_lr": 0.50,
        "reduce_after": 7,
        "new_lr": 0.25
    },
    {
        "id": "S14",
        "initial_lr": 0.50,
        "reduce_after": 5,
        "new_lr": 0.40
    },
    {
        "id": "S15",
        "initial_lr": 0.50,
        "reduce_after": 7,
        "new_lr": 0.40
    },
    {
        "id": "S16",
        "initial_lr": 0.50,
        "reduce_after": 5,
        "new_lr": 0.10
    }
]


# STEP 02A:
# Save results separately for each model size and seed.

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "scheduled_lr_experiments"
    / f"hidden_{HIDDEN_NEURONS}"
    / f"seed_{RANDOM_SEED}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 03: LOAD AND PREPARE MNIST
# ============================================================

# PURPOSE:
# Use the same image normalization and reproducible
# training/validation split as our Fixed SGD experiment.

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
# Use the same neural network as the updated Fixed SGD code.
#
# ARCHITECTURE:
# 784 input features -> configurable hidden neurons -> 10 outputs
#
# ReLU introduces nonlinearity between the linear layers.

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
# STEP 05: PREPARE IDENTICAL INITIAL MODEL WEIGHTS
# ============================================================

# PURPOSE:
# Reuse the initial weights generated by Fixed SGD.
#
# WHY:
# Fixed and Scheduled SGD must begin at the same point
# when we compare them.
#
# If the initialization file does not exist, generate it
# using the same seed and model architecture.
#
# Never load the old 128-neuron checkpoint into a
# 256-neuron model.

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

INITIAL_WEIGHTS_PATH = (
    INITIALIZATION_DIR / "mnist_initial_weights.pt"
)

if INITIAL_WEIGHTS_PATH.exists():

    initial_state = torch.load(
        INITIAL_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )

else:

    # Reproduce the same initialization as the Fixed SGD script.
    torch.manual_seed(RANDOM_SEED)

    initial_model = DigitClassifier(HIDDEN_NEURONS)

    initial_state = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }

    torch.save(
        initial_state,
        INITIAL_WEIGHTS_PATH
    )

# Verify compatibility before starting long experiments.
verification_model = DigitClassifier(HIDDEN_NEURONS)
verification_model.load_state_dict(initial_state)

print("\nMODEL INITIALIZATION")
print("Hidden neurons:", HIDDEN_NEURONS)
print("Initial weights:", INITIAL_WEIGHTS_PATH)
print("Initialization verification: PASSED")


# ============================================================
# STEP 06: CALCULATE VALIDATION ACCURACY
# ============================================================

# PURPOSE:
# Evaluate model predictions without modifying weights.

def calculate_validation_accuracy(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            predicted_labels = outputs.argmax(dim=1)

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# ============================================================
# STEP 07: TRAIN ONE SCHEDULED SGD CONFIGURATION
# ============================================================

# PURPOSE:
# Train one model using a specified learning-rate schedule.
#
# FAIR COMPARISON:
# Every experiment uses the same initial weights,
# dataset split, mini-batch ordering, and epoch budget.

def run_experiment(config):

    experiment_id = config["id"]
    initial_lr = config["initial_lr"]
    reduce_after = config["reduce_after"]
    new_lr = config["new_lr"]

    print("\n" + "=" * 65)
    print(f"SCHEDULED SGD EXPERIMENT: {experiment_id}")
    print(f"Hidden neurons: {HIDDEN_NEURONS}")
    print(f"Initial LR: {initial_lr}")
    print(f"Reduce after epoch: {reduce_after}")
    print(f"New LR: {new_lr}")
    print("=" * 65)

    # STEP 07A:
    # Create a fresh model and load shared initial weights.

    model = DigitClassifier(HIDDEN_NEURONS)
    model.load_state_dict(initial_state)

    # STEP 07B:
    # Reset batch order for every experiment.

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

    # STEP 07C:
    # Configure SGD with the initial learning rate.

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr
    )

    loss_function = nn.CrossEntropyLoss()

    history = []
    start_time = time.perf_counter()

    # ========================================================
    # STEP 08: TRAIN FOR THE SPECIFIED EPOCHS
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        total_training_loss = 0.0
        total_training_samples = 0

        # STEP 08A:
        # Choose the learning rate for the current epoch.
        #
        # Example reduce_after = 7:
        # Epochs 1-7: initial_lr
        # Epochs 8-10: new_lr

        if epoch >= reduce_after:
            current_lr = new_lr
        else:
            current_lr = initial_lr

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # STEP 08B:
        # Train on every mini-batch.

        for images, labels in train_loader:

            predictions = model(images)

            loss = loss_function(predictions, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            current_batch_size = labels.size(0)

            total_training_loss += (
                loss.item() * current_batch_size
            )

            total_training_samples += current_batch_size

        # STEP 08C:
        # Calculate average training loss.

        average_loss = (
            total_training_loss / total_training_samples
        )

        # STEP 08D:
        # Measure validation accuracy.

        validation_accuracy = calculate_validation_accuracy(
            model,
            validation_loader
        )

        # STEP 08E:
        # Record cumulative elapsed time and completed steps.

        elapsed_time = time.perf_counter() - start_time

        completed_steps = (
            (epoch + 1) * len(train_loader)
        )

        history.append({
            "experiment": experiment_id,
            "epoch": epoch + 1,
            "completed_steps": completed_steps,
            "learning_rate": current_lr,
            "training_loss": average_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": elapsed_time
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {current_lr:.3f} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {validation_accuracy:.2%} | "
            f"Steps: {completed_steps} | "
            f"Time: {elapsed_time:.2f}s"
        )

    # ========================================================
    # STEP 09: SAVE INDIVIDUAL EXPERIMENT RESULTS
    # ========================================================

    # PURPOSE:
    # Preserve the model, training history, and configuration.
    #
    # Use the experiment ID to keep runs separate.

    experiment_dir = OUTPUT_DIR / experiment_id

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = experiment_dir / "metrics.csv"

    with open(metrics_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

    model_path = experiment_dir / "model.pt"

    torch.save(
        model.state_dict(),
        model_path
    )

    with open(experiment_dir / "config.json", "w") as file:

        json.dump(
            {
                "experiment_id": experiment_id,
                "hidden_neurons": HIDDEN_NEURONS,
                "random_seed": RANDOM_SEED,
                "initial_learning_rate": initial_lr,
                "reduce_after": reduce_after,
                "new_learning_rate": new_lr,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "target_accuracy": TARGET_ACCURACY,
                "validation_frequency": "once per epoch",
                "pytorch_version": torch.__version__
            },
            file,
            indent=4
        )

    # ========================================================
    # STEP 10: SUMMARIZE THE EXPERIMENT
    # ========================================================

    # PURPOSE:
    # Measure best accuracy, final accuracy, and the earliest
    # recorded checkpoint reaching our target accuracy.
    #
    # NOTE:
    # Validation happens once per epoch, so the exact first
    # batch reaching 97% cannot be determined.

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
        "experiment": experiment_id,
        "hidden_neurons": HIDDEN_NEURONS,
        "random_seed": RANDOM_SEED,
        "initial_lr": initial_lr,
        "reduce_after": reduce_after,
        "new_lr": new_lr,
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
# STEP 11: RUN ALL SCHEDULED SGD EXPERIMENTS
# ============================================================

# PURPOSE:
# Automatically run the configured schedules, rank the
# results, and save a consolidated summary.

def main():

    # Validate experiment configurations before training.
    experiment_ids = [
        config["id"] for config in SCHEDULES
    ]

    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Experiment IDs must be unique.")

    for config in SCHEDULES:

        if not 0 <= config["reduce_after"] <= EPOCHS:
            raise ValueError(
                f"Invalid reduce_after for {config['id']}"
            )

        if (
            config["initial_lr"] <= 0
            or config["new_lr"] <= 0
        ):
            raise ValueError(
                f"Learning rates must be positive: {config['id']}"
            )

    results = []

    print("\nSTARTING SCHEDULED SGD EXPERIMENTS")

    for config in SCHEDULES:

        result = run_experiment(config)
        results.append(result)

    # ========================================================
    # STEP 12: RANK RESULTS
    # ========================================================

    ranked_results = sorted(
        results,
        key=lambda row: row["final_accuracy"],
        reverse=True
    )

    print("\n" + "=" * 85)
    print("SCHEDULED SGD EXPERIMENT COMPARISON")
    print("=" * 85)

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
            f"{result['experiment']} | "
            f"Start LR: {result['initial_lr']:.3f} | "
            f"After: {result['reduce_after']} epochs | "
            f"New LR: {result['new_lr']:.3f} | "
            f"Final: {result['final_accuracy']:.2%} | "
            f"Best: {result['best_accuracy']:.2%} | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Steps to 97%: {target_steps} | "
            f"Time to 97%: {target_time} | "
            f"Total: {result['total_seconds']:.2f}s"
        )

    # ========================================================
    # STEP 13: SAVE CONSOLIDATED SUMMARY
    # ========================================================

    summary_path = OUTPUT_DIR / "scheduled_lr_summary.csv"

    with open(summary_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(ranked_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(ranked_results)

    print("\nResults saved to:", OUTPUT_DIR)
    print("TASK: SCHEDULED SGD EXPERIMENTS COMPLETED")


# ============================================================
# STEP 14: EXECUTE EXPERIMENTS
# ============================================================

if __name__ == "__main__":
    main()