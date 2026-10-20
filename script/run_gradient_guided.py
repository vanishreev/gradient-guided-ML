
# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# TASK: CONFIGURABLE GRADIENT-GUIDED SGD EXPERIMENT
# ============================================================

# PURPOSE:
# Investigate whether cosine similarity between consecutive
# gradients can guide learning-rate decisionss and improve
# neural network training efficiency.
#
# RESEARCH QUESTIONS:
# 1. Does gradient-guided SGD improve validation accuracy?
# 2. Can it reach 97% accuracy in fewer training steps?
# 3. Does gradient alignment provide useful information
#    for selecting the learning rate?
# 4. Does the method behave differently when the number
#    of hidden neurons changes?
#
# EXPERIMENTAL CONTROLS:
# - Same MNIST training/validation split.
# - Same architecture for a given hidden-neuron count.
# - Same saved initialization as Fixed and Scheduled SGD.
# - Same random seed and mini-batch order.
# - Same batch size and number of epochs.
# - Original MNIST test data remains untouched.
#
# EXPECTED RESULTS:
# - Gradient similarity and selected LR at every step.
# - Training loss and validation accuracy per epoch.
# - Best and final validation accuracy.
# - Steps and time to the first recorded 97% checkpoint.
# - Saved experiment configuration and model parameters.


# ============================================================
# STEP 01: IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path
import csv
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# STEP 02: CONFIGURE EXPERIMENT SETTINGS
# ============================================================

# PURPOSE:
# Keep architecture and learning-rate settings here so
# experiments can be changed without changing training logic.

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

# Change this to test a different neural network size.
HIDDEN_NEURONS = 256

# ============================================================
# GRADIENT-GUIDED LEARNING-RATE SETTINGS
# ============================================================

# PURPOSE:
# Use cosine similarity to select a learning rate.
#
# Strong agreement: use 0.50.
# Weak agreement: use 0.375.
# Negative agreement: use 0.50.
#
# These are experimental choices, not proven optimal rules.

BASE_LEARNING_RATE = 0.50

STRONG_THRESHOLD = 0.50

STRONG_LEARNING_RATE = 0.50
WEAK_LEARNING_RATE = 0.375
NEGATIVE_LEARNING_RATE = 0.50


# ============================================================
# STEP 03: DEFINE EXPERIMENT ID AND OUTPUT PATHS
# ============================================================

# PURPOSE:
# Keep experiments separate by architecture, seed, and
# learning-rate configuration.
#
# Different settings should not overwrite earlier results.

EXPERIMENT_ID = (
    f"GG_STR{STRONG_LEARNING_RATE:.3f}"
    f"_WEAK{WEAK_LEARNING_RATE:.3f}"
    f"_NEG{NEGATIVE_LEARNING_RATE:.3f}"
    f"_THR{STRONG_THRESHOLD:.3f}"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "gradient_guided"
    / f"hidden_{HIDDEN_NEURONS}"
    / f"seed_{RANDOM_SEED}"
    / EXPERIMENT_ID
)

# Stop before overwriting results of an existing experiment.
if OUTPUT_DIR.exists() and (
    (OUTPUT_DIR / "experiment_summary.json").exists()
    or (OUTPUT_DIR / "guided_epoch_summary.csv").exists()
):
    raise FileExistsError(
        f"Results already exist: {OUTPUT_DIR}\n"
        "Choose a different experiment configuration or "
        "archive the existing results before rerunning."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 04: LOAD AND PREPARE MNIST
# ============================================================

# PURPOSE:
# Use the exact same normalization and data split as
# the Fixed and Scheduled SGD experiments.

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
# STEP 05: DEFINE THE NEURAL NETWORK ARCHITECTURE
# ============================================================

# PURPOSE:
# Match the neural network architecture in our updated
# Fixed and Scheduled SGD scripts.
#
# ARCHITECTURE:
# 784 input features
# -> HIDDEN_NEURONS with ReLU
# -> 10 output scores

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
# STEP 06: PREPARE IDENTICAL INITIAL MODEL WEIGHTS
# ============================================================

# PURPOSE:
# Load the same initialization file used by the updated
# Fixed and Scheduled SGD scripts.
#
# If the file is absent, create it deterministically
# using the same initialization procedure.
#
# IMPORTANT:
# Do not use an old 128-neuron checkpoint when testing
# a model with 256 hidden neurons.

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

# Verify checkpoint compatibility before training.

model = DigitClassifier(HIDDEN_NEURONS)

model.load_state_dict(initial_state)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nMODEL INITIALIZATION")
print("Hidden neurons:", HIDDEN_NEURONS)
print("Trainable parameters:", total_parameters)
print("Initial weights:", INITIAL_WEIGHTS_PATH)
print("Initialization verification: PASSED")


# ============================================================
# STEP 07: DEFINE VALIDATION ACCURACY
# ============================================================

# PURPOSE:
# Calculate validation accuracy without modifying weights.

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
# STEP 08: EXTRACT GRADIENT VECTOR
# ============================================================

# PURPOSE:
# Combine gradients for all weights and biases into
# one vector for cosine similarity calculations.
#
# MATHEMATICAL CONCEPT:
# Gradient = (dL/dw1, dL/dw2, ..., dL/dwn)
#
# detach():
# Stops tracking derivatives of the extracted values.
#
# clone():
# Creates an independent copy.

def get_gradient_vector(model):

    gradients = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients.append(
                parameter.grad.detach().flatten()
            )

    return torch.cat(gradients).clone()


# ============================================================
# STEP 09: CALCULATE GRADIENT COSINE SIMILARITY
# ============================================================

# PURPOSE:
# Measure the agreement between consecutive gradient
# directions before updating the model weights.
#
# FORMULA:
# Cosine similarity =
#
#    dot(previous_gradient, current_gradient)
#    ----------------------------------------
#    norm(previous_gradient) * norm(current_gradient)
#
# +1: Exactly the same direction.
#  0: Perpendicular directions.
# -1: Exactly opposite directions.
#
# Undefined similarity:
# First step or zero-length gradient.

def calculate_cosine_similarity(previous, current):

    if previous is None:
        return None

    previous_norm = torch.linalg.vector_norm(previous)

    current_norm = torch.linalg.vector_norm(current)

    if (
        previous_norm.item() == 0
        or current_norm.item() == 0
    ):
        return None

    similarity = F.cosine_similarity(
        previous.unsqueeze(0),
        current.unsqueeze(0),
        dim=1,
        eps=1e-12
    )

    result = similarity.item()

    if not math.isfinite(result):
        return None

    return result


# ============================================================
# STEP 10: CHOOSE THE GRADIENT-GUIDED LEARNING RATE
# ============================================================

# PURPOSE:
# Convert cosine similarity into a learning-rate decision.
#
# RULE:
# s > 0.60       -> LR = 0.50
# 0 <= s <= 0.60 -> LR = 0.375
# s < 0          -> LR = 0.50
#
# When similarity is undefined, use the base rate.
#
# All learning rates are direct values, not multipliers.

def choose_learning_rate(similarity):

    if similarity is None:
        return BASE_LEARNING_RATE, "undefined"

    if similarity > STRONG_THRESHOLD:
        return STRONG_LEARNING_RATE, "strong"

    if similarity >= 0:
        return WEAK_LEARNING_RATE, "weak"

    return NEGATIVE_LEARNING_RATE, "negative"


# ============================================================
# STEP 11: CONFIGURE MODEL, LOSS, AND OPTIMIZER
# ============================================================

# PURPOSE:
# Train using the same cross-entropy loss and plain SGD
# as the Fixed and Scheduled SGD experiments.

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=BASE_LEARNING_RATE
)


# ============================================================
# STEP 12: SAVE THE EXPERIMENT CONFIGURATION
# ============================================================

# PURPOSE:
# Record the architecture, seed, and gradient-guided
# learning-rate rule for reproducibility.

experiment_config = {
    "experiment_id": EXPERIMENT_ID,
    "hidden_neurons": HIDDEN_NEURONS,
    "random_seed": RANDOM_SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "target_accuracy": TARGET_ACCURACY,
    "base_learning_rate": BASE_LEARNING_RATE,
    "strong_threshold": STRONG_THRESHOLD,
    "strong_learning_rate": STRONG_LEARNING_RATE,
    "weak_learning_rate": WEAK_LEARNING_RATE,
    "negative_learning_rate": NEGATIVE_LEARNING_RATE,
    "architecture": f"784-{HIDDEN_NEURONS}-ReLU-10",
    "optimizer": "SGD",
    "dataset": "MNIST",
    "validation_frequency": "once per epoch",
    "pytorch_version": torch.__version__
}

config_path = OUTPUT_DIR / "experiment_config.json"

with open(config_path, "w") as file:

    json.dump(experiment_config, file, indent=4)


# ============================================================
# STEP 13: PREPARE EXPERIMENT TRACKING
# ============================================================

# PURPOSE:
# Record learning-rate decisions, gradient statistics,
# model accuracy, training steps, and elapsed time.

previous_gradient = None

step_history = []
epoch_history = []

global_step = 0

start_time = time.perf_counter()

print("\n" + "=" * 65)
print("GRADIENT-GUIDED SGD EXPERIMENT")
print("=" * 65)

print("Experiment ID:", EXPERIMENT_ID)
print("Hidden neurons:", HIDDEN_NEURONS)
print("Base learning rate:", BASE_LEARNING_RATE)
print("Strong threshold:", STRONG_THRESHOLD)
print("Strong LR:", STRONG_LEARNING_RATE)
print("Weak LR:", WEAK_LEARNING_RATE)
print("Negative LR:", NEGATIVE_LEARNING_RATE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# STEP 14: TRAIN USING GRADIENT-GUIDED SGD
# ============================================================

# PURPOSE:
# Calculate current gradients, compare directions,
# choose a learning rate, and update model parameters.

for epoch in range(EPOCHS):

    model.train()

    total_training_loss = 0.0
    total_training_samples = 0

    rate_counts = {
        "strong": 0,
        "weak": 0,
        "negative": 0,
        "undefined": 0
    }

    similarities = []

    for images, labels in train_loader:

        global_step += 1

        # STEP 14A:
        # Calculate predictions and classification loss.

        predictions = model(images)

        loss = loss_function(predictions, labels)

        # STEP 14B:
        # Clear previous gradients and calculate current ones.

        optimizer.zero_grad()

        loss.backward()

        # STEP 14C:
        # Extract the current gradient vector.

        current_gradient = get_gradient_vector(model)

        # STEP 14D:
        # Compare current and previous gradients.

        similarity = calculate_cosine_similarity(
            previous_gradient,
            current_gradient
        )

        # STEP 14E:
        # Select the learning rate from cosine similarity.

        current_lr, category = choose_learning_rate(
            similarity
        )

        # STEP 14F:
        # Apply the learning rate to SGD.

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # STEP 14G:
        # Record the gradient and learning-rate decision.

        rate_counts[category] += 1

        if similarity is not None:
            similarities.append(similarity)

        step_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "cosine_similarity": similarity,
            "learning_rate": current_lr,
            "category": category,
            "gradient_norm": (
                torch.linalg.vector_norm(
                    current_gradient
                ).item()
            )
        })

        # STEP 14H:
        # Save the current gradient for comparison with
        # the next batch's gradient.

        previous_gradient = current_gradient

        # STEP 14I:
        # Update all trainable parameters using SGD.

        optimizer.step()

        # STEP 14J:
        # Accumulate sample-weighted loss.

        current_batch_size = labels.size(0)

        total_training_loss += (
            loss.item() * current_batch_size
        )

        total_training_samples += current_batch_size

    # ========================================================
    # STEP 15: EVALUATE AFTER EACH EPOCH
    # ========================================================

    average_loss = (
        total_training_loss / total_training_samples
    )

    accuracy = calculate_validation_accuracy(
        model,
        validation_loader
    )

    elapsed_time = time.perf_counter() - start_time

    mean_similarity = (
        sum(similarities) / len(similarities)
        if similarities else None
    )

    epoch_history.append({
        "epoch": epoch + 1,
        "completed_steps": global_step,
        "training_loss": average_loss,
        "validation_accuracy": accuracy,
        "mean_similarity": mean_similarity,
        "strong_count": rate_counts["strong"],
        "weak_count": rate_counts["weak"],
        "negative_count": rate_counts["negative"],
        "undefined_count": rate_counts["undefined"],
        "elapsed_seconds": elapsed_time
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.2%} | "
        f"Steps: {global_step} | "
        f"Strong ({STRONG_LEARNING_RATE:.3f}): "
        f"{rate_counts['strong']} | "
        f"Weak ({WEAK_LEARNING_RATE:.3f}): "
        f"{rate_counts['weak']} | "
        f"Negative ({NEGATIVE_LEARNING_RATE:.3f}): "
        f"{rate_counts['negative']} | "
        f"Undefined: {rate_counts['undefined']} | "
        f"Time: {elapsed_time:.2f}s"
    )


# ============================================================
# STEP 16: SAVE EXPERIMENT RESULTS
# ============================================================

# PURPOSE:
# Preserve detailed step metrics, epoch metrics,
# and trained model parameters.

def save_csv(filename, rows):

    path = OUTPUT_DIR / filename

    with open(path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


save_csv("guided_step_metrics.csv", step_history)

save_csv("guided_epoch_summary.csv", epoch_history)

trained_model_path = OUTPUT_DIR / "guided_model.pt"

torch.save(
    model.state_dict(),
    trained_model_path
)


# ============================================================
# STEP 17: CALCULATE BEST ACCURACY AND TARGET METRICS
# ============================================================

# PURPOSE:
# Compare convergence efficiency and final performance.
#
# TARGET:
# Find the first recorded epoch at which validation
# accuracy reaches or exceeds 97%.
#
# LIMITATION:
# Validation is only measured at epoch boundaries.
# Therefore target steps are checkpoint-based, not
# the exact first batch reaching the target.

best_result = max(
    epoch_history,
    key=lambda row: row["validation_accuracy"]
)

target_result = next(
    (
        row
        for row in epoch_history
        if row["validation_accuracy"] >= TARGET_ACCURACY
    ),
    None
)

total_counts = {
    category: sum(
        row[f"{category}_count"]
        for row in epoch_history
    )
    for category in (
        "strong",
        "weak",
        "negative",
        "undefined"
    )
}

total_steps = len(step_history)

average_learning_rate = sum(
    row["learning_rate"]
    for row in step_history
) / total_steps


# ============================================================
# STEP 18: DISPLAY THE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT-GUIDED SGD EXPERIMENT SUMMARY")
print("=" * 70)

print(
    f"Final accuracy: "
    f"{epoch_history[-1]['validation_accuracy']:.2%}"
)

print(
    f"Best accuracy: "
    f"{best_result['validation_accuracy']:.2%}"
)

print("Best epoch:", best_result["epoch"])

print(
    f"Final training loss: "
    f"{epoch_history[-1]['training_loss']:.4f}"
)

print(
    f"Total time: "
    f"{epoch_history[-1]['elapsed_seconds']:.2f}s"
)

print("Total training steps:", total_steps)

if target_result is not None:

    print(f"\nTarget accuracy: {TARGET_ACCURACY:.0%}")

    print("First target epoch:", target_result["epoch"])

    print(
        "Completed steps at target:",
        target_result["completed_steps"]
    )

    print(
        f"Time at target: "
        f"{target_result['elapsed_seconds']:.2f}s"
    )

else:

    print(
        f"\nTarget of {TARGET_ACCURACY:.0%} "
        "not reached during training."
    )

print(
    f"\nAverage applied learning rate: "
    f"{average_learning_rate:.4f}"
)

print("\nGradient category distribution:")

for category, count in total_counts.items():

    percentage = 100 * count / total_steps

    print(
        f"{category}: {count} updates "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STEP 19: SAVE FINAL EXPERIMENT SUMMARY
# ============================================================

# PURPOSE:
# Store a concise machine-readable record for comparison
# against Fixed and Scheduled SGD experiments.

summary = {
    "experiment_id": EXPERIMENT_ID,
    "hidden_neurons": HIDDEN_NEURONS,
    "random_seed": RANDOM_SEED,
    "final_accuracy": epoch_history[-1]["validation_accuracy"],
    "best_accuracy": best_result["validation_accuracy"],
    "best_epoch": best_result["epoch"],
    "final_loss": epoch_history[-1]["training_loss"],
    "total_seconds": epoch_history[-1]["elapsed_seconds"],
    "total_steps": total_steps,
    "average_learning_rate": average_learning_rate,
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
    ),
    "gradient_category_counts": total_counts
}

summary_path = OUTPUT_DIR / "experiment_summary.json"

with open(summary_path, "w") as file:

    json.dump(summary, file, indent=4)

print("\nExperiment summary saved:", summary_path)
print("Model saved:", trained_model_path)
print("GRADIENT-GUIDED EXPERIMENT COMPLETED")