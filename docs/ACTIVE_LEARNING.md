# Active Learning Documentation

This document describes the Active Learning functionality implemented in the `trustai` tool. Active Learning helps in selecting the most informative samples for annotation, thereby reducing the labeling effort required to achieve high model performance.

## Overview

The Active Learning loop typically consists of the following steps:
1.  **Train**: The model trains on the currently labeled dataset.
2.  **Predict**: The model predicts on the unlabeled pool.
3.  **Select**: An uncertainty sampling strategy selects the "most uncertain" samples.
4.  **Annotate**: These samples are presented to the user for annotation.
5.  **Repeat**: The loop continues with the expanded labeled dataset.

## Components

The system is built around these core components:

*   **`ActiveLearningManager`** (`active_learning.py`): The core logic handler. It manages the sampling strategies, clustering (for diversity), and interaction with the API client.
*   **`ActiveLearningRunner`** (`active_learning_runner.py`): The interactive CLI runner that orchestrates the workflow, prompting the user for actions and displaying progress.
*   **`UncertaintySampling`** (`active_learning.py`): A utility class implementing various mathematical strategies to calculate uncertainty.

## Sampling Strategies

The system supports several uncertainty sampling strategies, defined in `active_learning.py`:

| Strategy | Description |
|----------|-------------|
| **Entropy** | Measures the average information content. Higher entropy implies higher uncertainty. Formula: $-\sum p_i \log(p_i)$ |
| **Least Confidence** | Focuses on the probability of the most likely class. Lower max probability implies higher uncertainty. |
| **Margin** | The difference between the top two class probabilities. Smaller margin implies higher uncertainty. |
| **Ratio** | Ratio between the top two class probabilities. (Note: effectively similar to margin in ranking). |

## Usage

### Running the Loop

You can start the active learning loop using the runner script:

```bash
python -m trustai.active_learning_runner
```

Or via the example script which helps with setup:

```bash
python -m trustai.run_active_learning
```

### The Interactive Workflow

When running `active_learning_runner.py`, you will be guided through:

1.  **Configuration**: Connecting to your Label Studio instance.
2.  **Project Selection**: Choosing the project to run active learning on.
3.  **Model Connection**: Ensuring an ML backend is connected.
4.  **The Loop**:
    *   The system checks for training status.
    *   It retrieves predictions for unlabeled tasks.
    *   It calculates uncertainty scores.
    *   It recommends a batch of tasks to label.
    *   You (the annotator) label these tasks in the Label Studio interface.
    *   You prompt the system to re-train the model.

## Diversity Sampling

To avoid selecting redundant examples (e.g., adjacent video frames or very similar images), the system can also employ **clustering-based diversity sampling**. 

If enabled, the system:
1.  Extracts embeddings (or uses raw features) from the data.
2.  Clusters the data using K-Means.
3.  Selects uncertain samples that are also diverse (i.e., from different clusters).

## Configuration

Standard project configuration (as detailed in `PROJECT_CONFIG.md`) is used to set up the environment. The active learning specific parameters are typically handled interactively or can be passed during the initialization of the runner.

## ML Backend Requirements

For the Active Learning loop to function correctly, your Machine Learning backend must implement certain endpoints.

### The `/train` Endpoint

Crucially, your ML backend **must** implement a `/train` endpoint if you wish to use the re-training capabilities of the active learning loop.

*   **Method**: `POST`
*   **Path**: `/train`
*   **Payload**: The payload typically contains information about the project and the annotations (or a reference to them).
*   **Behavior**: When this endpoint is called, your backend should trigger a training process using the latest labeled data. This process is often asynchronous.
*   **Output**: The active learning runner expects a success status to confirm training has started.

**Example Python (Flask) Implementation snippet:**

```python
@app.route('/train', methods=['POST'])
def train():
    data = request.json
    # ... logic to start model training ...
    return jsonify({"status": "training_started"}), 200
```

Without this endpoint, the "Train" step in the Active Learning loop will fail or do nothing, and the model will not improve based on new feedback.
