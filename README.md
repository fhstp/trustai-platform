# TrustAI: Active Learning with Label Studio

TrustAI is a comprehensive toolkit designed to streamline the integration of Active Learning workflows with Label Studio. It provides a set of Python utilities and scripts to manage projects, connect machine learning backends, and run interactive active learning loops to efficiently annotate data.

## Features

*   **Project Management**: Programmatically create and configure Label Studio projects via JSON configuration files.
*   **Active Learning**: specialized runner for interactive active learning loops with various uncertainty sampling strategies (Entropy, Least Confidence, Margin).
*   **ML Backend Integration**: Tools to easily connect and manage ML backends.
*   **Data Import/Export**: Utilities for importing tasks from files or URLs and exporting annotations.
*   **Interactive CLI**: Rich terminal interface for managing projects and running active learning sessions.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Setup Configuration

First, you need to configure the connection to your Label Studio instance. Create or edit `config.json` in the root directory:

```json
{
  "api_base_url": "http://localhost:8080",
  "api_key": "your-api-key-here"
}
```

*   `api_base_url`: The URL where your Label Studio instance is running.
*   `api_key`: Your personal API key, which can be found in Label Studio under Account & Settings > Account.

### 2. Setup a Project

You can define your project structure in a JSON file for reproducible setups. See [Project Configuration](docs/PROJECT_CONFIG.md) for details.

### 3. Run Active Learning

Start the interactive active learning runner to select the most informative samples for annotation.

```bash
python -m trustai.active_learning_runner
```

For a detailed guide on the active learning process and strategies, refer to the [Active Learning Documentation](docs/ACTIVE_LEARNING.md).

## Documentation

*   [Active Learning Guide](docs/ACTIVE_LEARNING.md): rigorous details on sampling strategies, the active learning loop, and diversity sampling.
*   [Project Configuration](docs/PROJECT_CONFIG.md): Schema and examples for configuring projects via JSON.

## Requirements

*   Python 3.8+
*   Label Studio instance (running locally or remotely)
*   `requests`, `rich`, `numpy`, `torch`, `scikit-learn` (see `requirements.txt`)
