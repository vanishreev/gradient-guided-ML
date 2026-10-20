
# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# TASK 08: GRADIENT DIRECTION ANALYSIS
# ============================================================

# PURPOSE:
# Measure how gradient directions change between consecutive
# training batches using cosine similarity.
#
# RESEARCH QUESTIONS:
# 1. How often do consecutive gradients agree?
# 2. How often do they point in opposite directions?
# 3. Does gradient alignment change across training epochs?
#
# IMPORTANT:
# This experiment measures gradients but does NOT use
# similarity to change the learning rate.
#
# EXPECTED OUTPUT:
# - Gradient similarity for each training step.
# - Epoch-level training loss and validation accuracy.
# - Summary of gradient agreement and disagreement.
# - CSV files for further research analysis.


# ============================================================
# STEP 01: IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# STEP 02: CONFIGURE EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "output" / "gradient_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "output"
    / "initialization"
    / "mnist_initial_weights.pt"
)

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.50


# ============================================================
# STEP 03: LOAD AND PREPARE MNIST
# ============================================================

# PURPOSE:
# Reproduce the exact dataset split used previously.

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_subset, validation_subset = random_split(
    dataset,
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


# ============================================================
# STEP 04: DEFINE THE ORIGINAL NEURAL NETWORK
# ============================================================

# PURPOSE:
# Use the same architecture as our earlier experiments.

class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(self.flatten(x))


# ============================================================
# STEP 05: CALCULATE VALIDATION ACCURACY
# ============================================================

def validation_accuracy(model):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_loader:

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ============================================================
# STEP 06: EXTRACT THE GRADIENT VECTOR
# ============================================================

# PURPOSE:
# Collect gradients for every trainable weight and bias
# into one vector so consecutive gradients can be compared.
#
# Example:
# Weight gradients: (4, 2)
# Bias gradients:   (1)
# Combined vector:  (4, 2, 1)

def get_gradient_vector(model):

    gradients = []

    for parameter in model.parameters():

        if parameter.grad is not None:
            gradients.append(
                parameter.grad.detach().flatten()
            )

    return torch.cat(gradients)


# ============================================================
# STEP 07: DEFINE COSINE SIMILARITY
# ============================================================

# MATHEMATICAL CONCEPT:
# Similarity = dot(g_previous, g_current)
#              --------------------------
#              |g_previous| * |g_current|
#
# INTERPRETATION:
# +1 = Same direction
#  0 = Perpendicular directions
# -1 = Opposite directions

def gradient_similarity(previous, current):

    # A zero gradient has no defined direction.
    if previous is None:
        return None

    if (
        torch.linalg.vector_norm(previous).item() == 0
        or torch.linalg.vector_norm(current).item() == 0
    ):
        return None

    similarity = F.cosine_similarity(
        previous.unsqueeze(0),
        current.unsqueeze(0),
        dim=1,
        eps=1e-12
    )

    return similarity.item()


# ============================================================
# STEP 08: INITIALIZE MODEL AND OPTIMIZER
# ============================================================

if not INITIAL_WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Initial weights not found: {INITIAL_WEIGHTS_PATH}"
    )

model = DigitClassifier()

model.load_state_dict(
    torch.load(
        INITIAL_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()


# ============================================================
# STEP 09: PREPARE TRACKING
# ============================================================

previous_gradient = None

step_history = []
epoch_history = []

global_step = 0
start_time = time.perf_counter()

print("\nTASK 08 - GRADIENT DIRECTION ANALYSIS")


# ============================================================
# STEP 10: TRAIN AND ANALYZE GRADIENTS
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_samples = 0

    epoch_similarities = []

    for images, labels in train_loader:

        global_step += 1

        # STEP 10A:
        # Calculate predictions and loss.

        predictions = model(images)

        loss = loss_function(predictions, labels)

        # STEP 10B:
        # Clear previously calculated gradients.

        optimizer.zero_grad()

        # STEP 10C:
        # Calculate current gradients.

        loss.backward()

        # STEP 10D:
        # Combine all parameter gradients into one vector.

        current_gradient = get_gradient_vector(model)

        # STEP 10E:
        # Compare current and previous gradients.

        similarity = gradient_similarity(
            previous_gradient,
            current_gradient
        )

        # STEP 10F:
        # Record similarity before updating the weights.

        if similarity is not None:

            epoch_similarities.append(similarity)

        step_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "cosine_similarity": similarity,
            "gradient_norm": (
                torch.linalg.vector_norm(
                    current_gradient
                ).item()
            )
        })

        # STEP 10G:
        # Save the current gradient for the next batch.
        # clone() preserves values even after optimizer.step().

        previous_gradient = current_gradient.clone()

        # STEP 10H:
        # Update weights using fixed SGD at 0.50.

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # ========================================================
    # STEP 11: SUMMARIZE THIS EPOCH
    # ========================================================

    average_loss = total_loss / total_samples

    accuracy = validation_accuracy(model)

    mean_similarity = (
        sum(epoch_similarities) / len(epoch_similarities)
        if epoch_similarities else None
    )

    negative_count = sum(
        value < 0 for value in epoch_similarities
    )

    negative_percent = (
        100 * negative_count / len(epoch_similarities)
        if epoch_similarities else 0
    )

    elapsed = time.perf_counter() - start_time

    epoch_history.append({
        "epoch": epoch + 1,
        "training_loss": average_loss,
        "validation_accuracy": accuracy,
        "mean_cosine_similarity": mean_similarity,
        "negative_similarity_percent": negative_percent,
        "elapsed_seconds": elapsed
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.2%} | "
        f"Mean Similarity: "
        f"{mean_similarity if mean_similarity is not None else float('nan'):.4f} | "
        f"Negative Gradients: {negative_percent:.2f}%"
    )


# ============================================================
# STEP 12: SAVE GRADIENT ANALYSIS RESULTS
# ============================================================

def save_csv(filename, records):

    file_path = OUTPUT_DIR / filename

    with open(file_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print("Saved:", file_path)


save_csv("gradient_step_metrics.csv", step_history)

save_csv("gradient_epoch_summary.csv", epoch_history)


# ============================================================
# STEP 13: PRINT OVERALL GRADIENT STATISTICS
# ============================================================

valid_similarities = [
    row["cosine_similarity"]
    for row in step_history
    if row["cosine_similarity"] is not None
]

strong_agreement = sum(
    value > 0.5 for value in valid_similarities
)

weak_agreement = sum(
    0 <= value <= 0.5 for value in valid_similarities
)

disagreement = sum(
    value < 0 for value in valid_similarities
)

total = len(valid_similarities)

print("\n" + "=" * 55)
print("GRADIENT ALIGNMENT SUMMARY")
print("=" * 55)

print("Total training steps:", global_step)
print("Valid gradient comparisons:", total)

if total:
    print(f"Strong agreement (> 0.5): {100*strong_agreement/total:.2f}%")
    print(f"Weak agreement (0 to 0.5): {100*weak_agreement/total:.2f}%")
    print(f"Disagreement (< 0): {100*disagreement/total:.2f}%")

print("\nTASK 08: COMPLETED")