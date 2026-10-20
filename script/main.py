# ============================================================
# PROJECT: GRADIENT-GUIDED ML
# ============================================================

# PROJECT PURPOSE:
# Investigate whether gradient alignment can guide learning
# rate adjustments to improve machine learning training
# efficiency and stability.

# RESEARCH QUESTION:
# Can comparing consecutive gradient directions help a
# machine learning model reach a target accuracy faster
# than conventional learning rate strategies?

# ============================================================
# PHASE 1: DATA PREPARATION
# ============================================================

# TASK 01: Load the MNIST handwritten digit dataset.

# PURPOSE:
# Load handwritten digit images that will be used to train
# and evaluate our Gradient-Guided ML models.
# #WHAT WE WANT TO UNDERSTAND:
# How many images are available for training and testing?
# What is the size of each image?
# What does each label represent?
# What does an individual image look like?
# EXPECTED RESULT:
# Confirm that MNIST contains 60,000 training images and
# 10,000 test images, each with dimensions of 28 x 28 pixels.

# STEP 01: Import the required libraries.
from numpy import floating
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# STEP 02: Download and load the original MNsIST datasets.
# The test dataset must remain untouched during model development.
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True
)


# STEP 03: Inspect the dataset dimensions and labels.
print("Number of training images:", len(train_dataset))
print("Number of test images:", len(test_dataset))

image, label = train_dataset[0]

print("First image size:", image.size)
print("First image label:", label)


# STEP 04: Visualize the first handwritten digit.
plt.imshow(image, cmap="gray")
plt.title(f"MNIST Sample - Digit {label}")
plt.axis("off")
plt.show()

# TASK 02: Normalize the input images.

#PURPOSE:
# Convert grayscale image pixel values from 0-255 into
# floating-point values between 0 and 1.
# WHY:
#Consistent input scaling helps neural networks train
#efficiently and allows fair comparison of our
#learning-rate optimization methods.
#STEP 01:
#Define the transformation used to normalize images.
image_transform = transforms.ToTensor()


#STEP 02:
#Apply the same transformation to training and test data.
#The test set remains reserved for final evaluation.
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=image_transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=image_transform
)


# STEP 03:
# Retrieve one normalized image and its label.
image, label = train_dataset[0]


# STEP 04:
#Verify the image's shape and pixel value range.
print("\nTASK 02 - NORMALIZATION CHECK")
print("Image shape:", image.shape)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())
print("Data type:", image.dtype)
print("Digit label:", label)


# STEP 05:
# Display the normalized image.
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Normalized MNIST Image - Digit {label}")
plt.axis("off")
plt.show()

# TASK 03: Create training, validation, and test datasets.
# PURPOSE:
# Split MNIST into three independent datasets to ensure
# reliable and fair evaluation of our ML experiments.
#
# WHY:
# - Training data is used to calculate gradients and
#   update model weights.
# - Validation data is used to compare learning-rate
#   strategies and select hyperparameters.
# - Test data is reserved for final evaluation only.
#
# EXPECTED RESULT:
# Training:   48,000 images
# Validation: 12,000 images
# Test:       10,000 images


# STEP 01: Import the tools needed for splitting the dataset.

import torch
from torch.utils.data import random_split


# STEP 02: Define a random seed for reproducibility.
# Using the same seed produces the same split each time.

RANDOM_SEED = 42

split_generator = torch.Generator().manual_seed(RANDOM_SEED)


# STEP 03: Define the training and validation sizes.
# MNIST provides 60,000 original training images.

TRAIN_SIZE = 48000
VALIDATION_SIZE = 12000


# STEP 04: Split the original training dataset.
# The fixed generator makes the split reproducible.

train_subset, validation_subset = random_split(
    train_dataset,
    [TRAIN_SIZE, VALIDATION_SIZE],
    generator=split_generator
)


# STEP 05: Preserve the original test dataset.
# The test images must not be used to select our
# learning-rate strategy or tune the model.

final_test_dataset = test_dataset


# STEP 06: Verify the dataset sizes.

print("\nTASK 03 - DATASET SPLIT VERIFICATION")

print("Training images:", len(train_subset))
print("Validation images:", len(validation_subset))
print("Test images:", len(final_test_dataset))


# STEP 07: Check that training and validation indices
# do not overlap.

training_indices = set(train_subset.indices)
validation_indices = set(validation_subset.indices)

overlap = training_indices.intersection(validation_indices)

print("Overlapping training/validation images:", len(overlap))


# STEP 08: Verify the total number of images.

total_images = (
    len(train_subset)
    + len(validation_subset)
    + len(final_test_dataset)
)

print("Total images:", total_images)


# STEP 09: Verify that all split conditions are satisfied.

assert len(train_subset) == 48000
assert len(validation_subset) == 12000
assert len(final_test_dataset) == 10000
assert len(overlap) == 0

print("Dataset split verification: PASSED")

# ============================================================
# PHASE 2: MODEL DEVELOPMENT
# ============================================================

# TASK 04: Define a simple neural network.

# PURPOSE:
# Build a simple neural network capable of classifying
# handwritten digits from the MNIST dataset.
#
# WHY:
# Our research compares learning-rate optimization methods.
# All methods must use the exact same model architecture
# to ensure a fair comparison.
#
# MODEL ARCHITECTURE:
# Input layer:  784 features (28 x 28 image pixels)
# Hidden layer: 128 neurons with ReLU activation
# Output layer: 10 scores (digits 0 through 9)
#
# EXPECTED RESULT:
# Verify that the neural network accepts MNIST images
# and produces 10 output scores per image.


# STEP 01: Import PyTorch neural network components.

import torch.nn as nn

# STEP 02: Define the neural network class.

class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Convert each 28 x 28 image into 784 input values.
        self.flatten = nn.Flatten()

        # Define the layers used to process image features.
        self.network = nn.Sequential(

            # Connect 784 input values to 128 hidden neurons.
            nn.Linear(784, 128),

            # Apply a nonlinear activation function.
            nn.ReLU(),

            # Produce 10 scores, one for each digit.
            nn.Linear(128, 10)
        )

    # STEP 03: Define how input images move through the model.
    def forward(self, x):

        # Convert image dimensions into a flat vector.
        x = self.flatten(x)

        # Calculate the 10 output scores.
        x = self.network(x)

        return x

# STEP 04: Create an instance of the neural network.

model = DigitClassifier()

# STEP 05: Display the model architecture.

print("\nTASK 04 - NEURAL NETWORK ARCHITECTURE")
print(model)

# STEP 06: Retrieve one normalized image from MNIST.

sample_image, sample_label = train_subset[0]

# STEP 07: Add a batch dimension to the image.
# Neural networks expect inputs with a batch dimension.
# Shape changes from [1, 28, 28] to [1, 1, 28, 28].

sample_batch = sample_image.unsqueeze(0)

# STEP 08: Perform a forward pass without training.
# This only checks that the model processes the input.

with torch.no_grad():
    output = model(sample_batch)

# STEP 09: Verify the input and output dimensions.

print("Input shape:", sample_batch.shape)
print("Output shape:", output.shape)
print("Actual digit label:", sample_label)

# STEP 10: Confirm that the model produces 10 scores.

assert output.shape == (1, 10)

print("Neural network verification: PASSED")

# TASK 05: Initialize and save the starting model weights.

# PURPOSE:
# Examine the neural network's initial weights and biases,
# then save them for reproducible training experiments.
# WHY:
# Our research compares three learning-rate strategies.
# Each strategy must begin with identical model parameters.
#
# EXPECTED RESULT:
# 1. Display the number of trainable parameters.
# 2. Inspect sample weight and bias values.
# 3. Save the initial model state.
# 4. Verify that the saved state can be loaded successfully.


# STEP 01: Import utilities for handling file paths.

from pathlib import Path


# STEP 02: Set the random seed for reproducible initialization.
# Create a new model after setting the seed so that its
# starting weights are determined by RANDOM_SEED.

torch.manual_seed(RANDOM_SEED)

model = DigitClassifier()


# STEP 03: Count all trainable weights and biases.

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTASK 05 - MODEL PARAMETER INITIALIZATION")
print("Total trainable parameters:", total_parameters)


# STEP 04: Inspect each parameter tensor.
# The first linear layer has 128 x 784 weights and 128 biases.
# The output layer has 10 x 128 weights and 10 biases.

for name, parameter in model.named_parameters():

    print("\nParameter name:", name)
    print("Shape:", tuple(parameter.shape))
    print("Number of values:", parameter.numel())

    # Show only the first five values for readability.
    sample_values = parameter.detach().flatten()[:5]

    print("First five values:", sample_values.tolist())


# STEP 05: Define an output directory.
# Resolve the path relative to this script's parent project
# folder so it works on both Windows and macOS.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "initialization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# STEP 06: Save the starting weights and biases.
# The state_dict includes all model parameters.

initial_weights_path = OUTPUT_DIR / "mnist_initial_weights.pt"

torch.save(model.state_dict(), initial_weights_path)

print("\nSaved initial weights to:", initial_weights_path)


# STEP 07: Create another identical model.

verification_model = DigitClassifier()


# STEP 08: Load the saved initial weights.

saved_state = torch.load(
    initial_weights_path,
    map_location="cpu",
    weights_only=True
)

verification_model.load_state_dict(saved_state)


# STEP 09: Verify that both models have identical parameters.

for original, loaded in zip(
    model.parameters(),
    verification_model.parameters()
):
    assert torch.equal(original, loaded)

print("Initial weights verification: PASSED")

# ============================================================
# PHASE 3: BASELINE TRAINING
# ============================================================
# TASK 06: Train using SGD with a fixed learning rate.
# PURPOSE:
# Train our MNIST neural network using standard SGD with
# a fixed learning rate to establish a baseline.
#
# RESEARCH OBJECTIVE:
# Measure how efficiently the model learns without using
# gradient alignment to control its learning rate.
#
# WHY:
# We need a reproducible baseline to compare against
# scheduled SGD and our gradient-guided SGD method.
#
# EXPECTED RESULT:
# 1. Train for 10 epochs using fixed-rate SGD.
# 2. Record training loss and validation accuracy.
# 3. Measure total training time.
# 4. Save results and trained model weights.
#
# IMPORTANT:
# We use training data to update model weights.
# Validation data is only used to measure performance.
# The test dataset remains untouched.


# STEP 01: Import training utilities.

import time
import csv
from torch.utils.data import DataLoader


# STEP 02: Define baseline experiment settings.

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10


# STEP 03: Create reproducible training and validation loaders.
# The training loader shuffles images during training.
# The generator ensures the shuffle order is reproducible.

train_generator = torch.Generator().manual_seed(RANDOM_SEED)

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


# STEP 04: Create the baseline model and restore the
# saved initial weights from Task 05.

baseline_model = DigitClassifier()

baseline_model.load_state_dict(
    torch.load(
        initial_weights_path,
        map_location="cpu",
        weights_only=True
    )
)


# STEP 05: Configure the loss function.
# CrossEntropyLoss compares 10 digit-class scores
# against the correct digit label.

loss_function = nn.CrossEntropyLoss()


# STEP 06: Configure standard SGD with fixed learning rate.
# Momentum is zero by default, so updates follow the
# basic gradient descent equation.

optimizer = torch.optim.SGD(
    baseline_model.parameters(),
    lr=LEARNING_RATE
)


# STEP 07: Define the validation function.
# Validation measures accuracy without changing weights.

def calculate_validation_accuracy(model, data_loader):

    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in data_loader:

            outputs = model(images)

            predicted_labels = outputs.argmax(dim=1)

            correct_predictions += (
                predicted_labels == labels
            ).sum().item()

            total_predictions += labels.size(0)

    return correct_predictions / total_predictions


# STEP 08: Prepare a list to record experiment metrics.

training_history = []

print("\nTASK 06 - FIXED-RATE SGD BASELINE")
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# STEP 09: Start measuring training time.

training_start_time = time.perf_counter()


# STEP 10: Train the model for the configured epochs.

for epoch in range(EPOCHS):

    baseline_model.train()

    total_training_loss = 0.0
    total_training_samples = 0

    for images, labels in train_loader:

        # Forward pass: calculate digit predictions.
        predictions = baseline_model(images)

        # Calculate prediction error.
        loss = loss_function(predictions, labels)

        # Clear gradients left from the previous batch.
        optimizer.zero_grad()

        # Backpropagation: calculate current gradients.
        loss.backward()

        # Apply Gradient Descent to update the weights.
        optimizer.step()

        # Accumulate the sample-weighted training loss.
        batch_size = labels.size(0)

        total_training_loss += loss.item() * batch_size
        total_training_samples += batch_size

    # Calculate average loss across training images.
    average_training_loss = (
        total_training_loss / total_training_samples
    )

    # Measure accuracy using validation data only.
    validation_accuracy = calculate_validation_accuracy(
        baseline_model,
        validation_loader
    )

    # Record elapsed time since training began.
    elapsed_time = (
        time.perf_counter() - training_start_time
    )

    # Save the metrics for this epoch.
    training_history.append({
        "epoch": epoch + 1,
        "training_loss": average_training_loss,
        "validation_accuracy": validation_accuracy,
        "elapsed_seconds": elapsed_time
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_training_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.2%} | "
        f"Time: {elapsed_time:.2f}s"
    )


# STEP 11: Save experiment metrics to a CSV file.

BASELINE_OUTPUT_DIR = OUTPUT_DIR.parent / "baseline"

BASELINE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

metrics_path = BASELINE_OUTPUT_DIR / "fixed_sgd_metrics.csv"

with open(metrics_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "training_loss",
            "validation_accuracy",
            "elapsed_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(training_history)


# STEP 12: Save the trained model parameters.

trained_model_path = (
    BASELINE_OUTPUT_DIR / "fixed_sgd_model.pt"
)

torch.save(
    baseline_model.state_dict(),
    trained_model_path
)


# STEP 13: Confirm experiment completion.

print("\nFixed-rate SGD baseline: COMPLETED")
print("Metrics saved to:", metrics_path)
print("Model saved to:", trained_model_path)

# TASK 07: Train using SGD with a predefined learning rate
# schedule.
# PURPOSE:
# Train the same MNIST model using SGD with a learning rate
# that decreases according to a predefined schedule.
#
# RESEARCH OBJECTIVE:
# Determine whether scheduled learning-rate reductions
# improve training compared with fixed-rate SGD.
#
# WHY:
# We need an established learning-rate strategy to compare
# against our future gradient-guided learning-rate method.
#
# EXPERIMENT SETTINGS:
# Initial learning rate: 0.01
# Batch size: 64
# Epochs: 10
# Scheduler: StepLR
# Step interval: Every 3 epochs
# Reduction factor: 0.5
#
# EXPECTED RESULT:
# Record training loss, validation accuracy, elapsed time,
# and learning-rate history for scheduled SGD.


# STEP 01: Restore the exact same initial model weights
# used in Task 06.

scheduled_model = DigitClassifier()

scheduled_model.load_state_dict(
    torch.load(
        initial_weights_path,
        map_location="cpu",
        weights_only=True
    )
)


# STEP 02: Create a new SGD optimizer.

scheduled_optimizer = torch.optim.SGD(
    scheduled_model.parameters(),
    lr=LEARNING_RATE
)


# STEP 03: Create the learning-rate scheduler.
# StepLR multiplies the learning rate by 0.5 after
# every three completed epochs.

scheduler = torch.optim.lr_scheduler.StepLR(
    scheduled_optimizer,
    step_size=3,
    gamma=0.5
)


# STEP 04: Reset the batch-shuffling generator.
# This ensures Task 07 receives the same training
# batch order as Task 06.

scheduled_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

scheduled_train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=scheduled_generator,
    num_workers=0
)


# STEP 05: Prepare experiment tracking.

scheduled_history = []

scheduled_start_time = time.perf_counter()

print("\nTASK 07 - SCHEDULED SGD BASELINE")


# STEP 06: Train for 10 epochs.

for epoch in range(EPOCHS):

    scheduled_model.train()

    total_training_loss = 0.0
    total_training_samples = 0

    # Record the learning rate used during this epoch.
    current_lr = scheduled_optimizer.param_groups[0]["lr"]

    for images, labels in scheduled_train_loader:

        # Calculate predictions.
        predictions = scheduled_model(images)

        # Calculate prediction error.
        loss = loss_function(predictions, labels)

        # Clear previously calculated gradients.
        scheduled_optimizer.zero_grad()

        # Calculate new gradients through backpropagation.
        loss.backward()

        # Update weights using the current learning rate.
        scheduled_optimizer.step()

        # Accumulate sample-weighted loss.
        batch_size = labels.size(0)

        total_training_loss += loss.item() * batch_size
        total_training_samples += batch_size

    # Calculate average training loss.
    average_training_loss = (
        total_training_loss / total_training_samples
    )

    # Measure validation accuracy.
    validation_accuracy = calculate_validation_accuracy(
        scheduled_model,
        validation_loader
    )

    # Measure elapsed time.
    elapsed_time = time.perf_counter() - scheduled_start_time

    # Record experiment metrics.
    scheduled_history.append({
        "epoch": epoch + 1,
        "learning_rate": current_lr,
        "training_loss": average_training_loss,
        "validation_accuracy": validation_accuracy,
        "elapsed_seconds": elapsed_time
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"LR: {current_lr:.5f} | "
        f"Loss: {average_training_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.2%} | "
        f"Time: {elapsed_time:.2f}s"
    )

    # Update the learning rate for the next epoch.
    scheduler.step()


# STEP 07: Create an output directory.

SCHEDULED_OUTPUT_DIR = OUTPUT_DIR.parent / "scheduled"

SCHEDULED_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# ============================================================
# TASK 07A: SCHEDULED SGD - MULTIPLE EXPERIMENTS
# ============================================================

# PURPOSE:
# Compare multiple learning-rate schedules using the
# same MNIST neural network and experimental conditions.
#
# RESEARCH QUESTION:
# Which combination of step_size and gamma gives the
# best validation performance during 10 training epochs?
#
# EXPERIMENTAL CONTROLS:
# - Same neural network architecture.
# - Same saved initial weights.
# - Same training and validation datasets.
# - Same initial learning rate (0.01).
# - Same batch size (64).
# - Same number of epochs (10).
# - Same random seed and training batch order.
#
# EXPECTED RESULT:
# 1. Run five scheduled SGD configurations.
# 2. Record loss, validation accuracy and elapsed time.
# 3. Save results separately for every experiment.
# 4. Create a summary comparing all five configurations.


# STEP 01: Define our experimental configurations.

schedule_experiments = [
    {"id": "S02", "step_size": 5, "gamma": 0.5},
    {"id": "S03", "step_size": 7, "gamma": 0.5},
    {"id": "S04", "step_size": 3, "gamma": 0.8},
    {"id": "S05", "step_size": 5, "gamma": 0.8},
    {"id": "S06", "step_size": 7, "gamma": 0.8},
]


# STEP 02: Create the directory for experiment outputs.

EXPERIMENTS_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "scheduled_experiments"
)

EXPERIMENTS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# STEP 03: Prepare a collection for experiment summaries.

experiment_summaries = []


# STEP 04: Run every scheduled SGD configuration.

for experiment in schedule_experiments:

    experiment_id = experiment["id"]
    step_size = experiment["step_size"]
    gamma = experiment["gamma"]

    print("\n" + "=" * 55)
    print(f"EXPERIMENT: {experiment_id}")
    print(f"Step size: {step_size} epochs")
    print(f"Gamma: {gamma}")
    print("=" * 55)

    # STEP 04A:
    # Create a fresh model and load identical starting
    # weights for this experiment.

    experiment_model = DigitClassifier()

    experiment_model.load_state_dict(
        torch.load(
            initial_weights_path,
            map_location="cpu",
            weights_only=True
        )
    )

    # STEP 04B:
    # Create an independent SGD optimizer.

    experiment_optimizer = torch.optim.SGD(
        experiment_model.parameters(),
        lr=LEARNING_RATE
    )

    # STEP 04C:
    # Configure the learning-rate schedule.

    experiment_scheduler = torch.optim.lr_scheduler.StepLR(
        experiment_optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # STEP 04D:
    # Reset the batch-order generator for every experiment.
    # This ensures all experiments receive the same
    # image batches in the same sequence.

    experiment_generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    experiment_train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=experiment_generator,
        num_workers=0
    )

    # STEP 04E:
    # Prepare tracking for this experiment.

    experiment_history = []

    experiment_start_time = time.perf_counter()

    # STEP 04F:
    # Run the complete training process.

    for epoch in range(EPOCHS):

        experiment_model.train()

        total_loss = 0.0
        total_samples = 0

        current_lr = experiment_optimizer.param_groups[0]["lr"]

        for images, labels in experiment_train_loader:

            # Predict digits.
            predictions = experiment_model(images)

            # Calculate prediction error.
            loss = loss_function(predictions, labels)

            # Clear previous gradients.
            experiment_optimizer.zero_grad()

            # Calculate gradients using backpropagation.
            loss.backward()

            # Update model weights using SGD.
            experiment_optimizer.step()

            # Accumulate sample-weighted training loss.
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Calculate average training loss.
        average_loss = total_loss / total_samples

        # Evaluate using validation images only.
        validation_accuracy = calculate_validation_accuracy(
            experiment_model,
            validation_loader
        )

        # Calculate elapsed training time.
        elapsed_time = (
            time.perf_counter() - experiment_start_time
        )

        # Save this epoch's results.
        experiment_history.append({
            "experiment": experiment_id,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "training_loss": average_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": elapsed_time
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {current_lr:.5f} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"Time: {elapsed_time:.2f}s"
        )

        # Update the learning rate for the next epoch.
        experiment_scheduler.step()

    # STEP 04G:
    # Save metrics for the completed experiment.

    metrics_file = (
        EXPERIMENTS_OUTPUT_DIR
        / f"{experiment_id}_metrics.csv"
    )

    with open(metrics_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "epoch",
                "learning_rate",
                "training_loss",
                "validation_accuracy",
                "elapsed_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(experiment_history)

    # STEP 04H:
    # Save the trained model's weights.

    model_file = (
        EXPERIMENTS_OUTPUT_DIR
        / f"{experiment_id}_model.pt"
    )

    torch.save(
        experiment_model.state_dict(),
        model_file
    )

    # STEP 04I:
    # Store the final metrics in a summary collection.

    final_result = experiment_history[-1]

    experiment_summaries.append({
        "experiment": experiment_id,
        "step_size": step_size,
        "gamma": gamma,
        "final_loss": final_result["training_loss"],
        "final_validation_accuracy": final_result[
            "validation_accuracy"
        ],
        "total_seconds": final_result["elapsed_seconds"]
    })

    print(f"\n{experiment_id}: COMPLETED")


# STEP 05: Save a consolidated comparison table.

summary_path = (
    EXPERIMENTS_OUTPUT_DIR / "experiment_summary.csv"
)

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "experiment",
            "step_size",
            "gamma",
            "final_loss",
            "final_validation_accuracy",
            "total_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(experiment_summaries)


# STEP 06: Display experiments ranked by validation accuracy.

ranked_experiments = sorted(
    experiment_summaries,
    key=lambda result: result["final_validation_accuracy"],
    reverse=True
)

print("\n" + "=" * 65)
print("SCHEDULED SGD EXPERIMENT COMPARISON")
print("=" * 65)

for result in ranked_experiments:

    print(
        f"{result['experiment']} | "
        f"Step: {result['step_size']} | "
        f"Gamma: {result['gamma']} | "
        f"Accuracy: "
        f"{result['final_validation_accuracy']:.2%} | "
        f"Loss: {result['final_loss']:.4f} | "
        f"Time: {result['total_seconds']:.2f}s"
    )

print("\nExperiment summary saved to:", summary_path)
print("Task 07A: COMPLETED")

# STEP 08: Save experiment metrics.

scheduled_metrics_path = (
    SCHEDULED_OUTPUT_DIR / "scheduled_sgd_metrics.csv"
)

with open(scheduled_metrics_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "learning_rate",
            "training_loss",
            "validation_accuracy",
            "elapsed_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(scheduled_history)


# STEP 09: Save trained model weights.

scheduled_model_path = (
    SCHEDULED_OUTPUT_DIR / "scheduled_sgd_model.pt"
)

torch.save(
    scheduled_model.state_dict(),
    scheduled_model_path
)


# STEP 10: Confirm completion.

print("\nScheduled SGD baseline: COMPLETED")
print("Metrics saved to:", scheduled_metrics_path)
print("Model saved to:", scheduled_model_path)

# ============================================================
# TASK 07B: EXPERIMENT WITH HIGHER LEARNING RATES
# ============================================================

# PURPOSE:
# Investigate whether increasing the initial learning rate
# from 0.01 to 0.02 improves MNIST training performance.
#
# RESEARCH QUESTIONS:
# 1. Does fixed SGD at 0.02 outperform fixed SGD at 0.01?
# 2. Does reducing the learning rate from 0.02 to 0.01
#    improve performance compared with fixed SGD at 0.02?
# 3. Does the timing of the reduction matter?
#
# EXPERIMENTAL CONTROLS:
# - Same MNIST training and validation datasets.
# - Same neural network architecture.
# - Same saved initial model weights.
# - Same batch size and training budget.
# - Same random seed and batch order.
#
# EXPECTED RESULT:
# Run F02, S09, and S10, save each experiment's results,
# and generate a summary comparing validation accuracy.


# STEP 01: Define the new experimental configurations.

learning_rate_experiments = [
    {
        "id": "F02",
        "initial_lr": 0.02,
        "reduce_after": None,
        "final_lr": 0.02
    },
    {
        "id": "S09",
        "initial_lr": 0.02,
        "reduce_after": 5,
        "final_lr": 0.01
    },
    {
        "id": "S10",
        "initial_lr": 0.02,
        "reduce_after": 7,
        "final_lr": 0.01
    }
]


# STEP 02: Create a directory for experiment outputs.

LR_EXPERIMENT_DIR = (
    PROJECT_ROOT / "output" / "learning_rate_experiments"
)

LR_EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# STEP 03: Prepare a list to collect experiment summaries.

lr_experiment_summaries = []


# STEP 04: Run the three configurations.

for experiment in learning_rate_experiments:

    experiment_id = experiment["id"]

    initial_lr = experiment["initial_lr"]

    reduce_after = experiment["reduce_after"]

    final_lr = experiment["final_lr"]

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {experiment_id}")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Reduce after epoch: {reduce_after}")
    print(f"Final learning rate: {final_lr}")
    print("=" * 60)

    # STEP 04A:
    # Create a fresh model and load the same initial weights.

    lr_model = DigitClassifier()

    lr_model.load_state_dict(
        torch.load(
            initial_weights_path,
            map_location="cpu",
            weights_only=True
        )
    )

    # STEP 04B:
    # Create a new SGD optimizer using the experiment's
    # initial learning rate.

    lr_optimizer = torch.optim.SGD(
        lr_model.parameters(),
        lr=initial_lr
    )

    # STEP 04C:
    # Reset the training batch order for fair comparison.

    lr_generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    lr_train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=lr_generator,
        num_workers=0
    )

    # STEP 04D:
    # Prepare metrics and start measuring training time.

    lr_history = []

    lr_start_time = time.perf_counter()

    # STEP 04E:
    # Train the model for 10 epochs.

    for epoch in range(EPOCHS):

        lr_model.train()

        total_loss = 0.0
        total_samples = 0

        # Select the learning rate for the current epoch.
        # F02 always uses 0.02.
        # S09 changes to 0.01 starting at Epoch 6.
        # S10 changes to 0.01 starting at Epoch 8.

        if (
            reduce_after is not None
            and epoch >= reduce_after
        ):
            current_lr = final_lr
        else:
            current_lr = initial_lr

        # Apply the learning rate to the optimizer.
        for parameter_group in lr_optimizer.param_groups:
            parameter_group["lr"] = current_lr

        # Process every training batch.

        for images, labels in lr_train_loader:

            # Make predictions.
            predictions = lr_model(images)

            # Calculate classification loss.
            loss = loss_function(predictions, labels)

            # Clear gradients from the previous batch.
            lr_optimizer.zero_grad()

            # Calculate current gradients.
            loss.backward()

            # Update weights using the selected learning rate.
            lr_optimizer.step()

            # Accumulate sample-weighted training loss.
            current_batch_size = labels.size(0)

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        # Calculate average training loss.
        average_loss = total_loss / total_samples

        # Calculate validation accuracy.
        validation_accuracy = calculate_validation_accuracy(
            lr_model,
            validation_loader
        )

        # Record elapsed time.
        elapsed_time = time.perf_counter() - lr_start_time

        # Save epoch-level metrics.
        lr_history.append({
            "experiment": experiment_id,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "training_loss": average_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": elapsed_time
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {current_lr:.5f} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"Time: {elapsed_time:.2f}s"
        )

    # STEP 04F:
    # Save the metrics for the completed experiment.

    metrics_path = (
        LR_EXPERIMENT_DIR / f"{experiment_id}_metrics.csv"
    )

    with open(metrics_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "epoch",
                "learning_rate",
                "training_loss",
                "validation_accuracy",
                "elapsed_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(lr_history)

    # STEP 04G:
    # Save the trained model parameters.

    model_path = (
        LR_EXPERIMENT_DIR / f"{experiment_id}_model.pt"
    )

    torch.save(lr_model.state_dict(), model_path)

    # STEP 04H:
    # Collect final and best validation accuracy.
    # Best accuracy is descriptive for now; later research
    # comparisons will also use predefined target metrics.

    final_result = lr_history[-1]

    best_result = max(
        lr_history,
        key=lambda result: result["validation_accuracy"]
    )

    lr_experiment_summaries.append({
        "experiment": experiment_id,
        "initial_lr": initial_lr,
        "reduce_after": (
            "Never" if reduce_after is None else reduce_after
        ),
        "final_accuracy": final_result["validation_accuracy"],
        "best_accuracy": best_result["validation_accuracy"],
        "best_epoch": best_result["epoch"],
        "final_loss": final_result["training_loss"],
        "total_seconds": final_result["elapsed_seconds"]
    })

    print(f"\nExperiment {experiment_id}: COMPLETED")


# STEP 05: Save a consolidated experiment summary.

summary_path = (
    LR_EXPERIMENT_DIR / "learning_rate_summary.csv"
)

summary_fields = [
    "experiment",
    "initial_lr",
    "reduce_after",
    "final_accuracy",
    "best_accuracy",
    "best_epoch",
    "final_loss",
    "total_seconds"
]

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_fields
    )

    writer.writeheader()
    writer.writerows(lr_experiment_summaries)


# STEP 06: Display the experiments ranked by final accuracy.

ranked_results = sorted(
    lr_experiment_summaries,
    key=lambda result: result["final_accuracy"],
    reverse=True
)

print("\n" + "=" * 70)
print("TASK 07B - LEARNING RATE EXPERIMENT COMPARISON")
print("=" * 70)

print("Original Fixed SGD (0.01): 91.71%")
print("-" * 70)

for result in ranked_results:

    print(
        f"{result['experiment']} | "
        f"Final Accuracy: {result['final_accuracy']:.2%} | "
        f"Best Accuracy: {result['best_accuracy']:.2%} | "
        f"Best Epoch: {result['best_epoch']} | "
        f"Loss: {result['final_loss']:.4f} | "
        f"Time: {result['total_seconds']:.2f}s"
    )

print("\nSummary saved to:", summary_path)
print("TASK 07B: COMPLETED")

# ============================================================
# PHASE 4: GRADIENT-GUIDED TRAINING
# ============================================================

# TASK 08: Calculate and store consecutive gradient vectors.

# TASK 09: Calculate cosine similarity between gradients.

# TASK 10: Adjust the learning rate based on alignment.

# TASK 11: Train using gradient-guided SGD.


# ============================================================
# PHASE 5: EVALUATION
# ============================================================

# TASK 12: Record loss, accuracy, training steps, and time.

# TASK 13: Repeat experiments with controlled random seeds.

# TASK 14: Compare training efficiency and stability.

# TASK 15: Run final evaluation on untouched test data.


# ============================================================
# PHASE 6: RESEARCH ANALYSIS
# ============================================================

# TASK 16: Visualize training results.

# TASK 17: Investigate when gradient alignment helps.

# TASK 18: Document findings and research limitations.