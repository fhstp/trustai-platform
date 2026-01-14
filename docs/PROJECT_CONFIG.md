# Project Configuration Documentation

This document describes the configuration file format used to create Label Studio projects programmatically using the `trustai` tool.

## Overview

You can define a project's structure, including its Label Studio configuration, Machine Learning backend, and initial dataset, in a single JSON file. This allows for reproducible project setups and quick initialization.

## Configuration File Structure

The configuration file is a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | **Yes** | The title of the project. |
| `description` | string | No | A description for the project. |
| `label_config` | string | No | The Label Studio XML configuration. Can be the XML string itself or a path to an `.xml` file. |
| `sampling` | string | No | Sampling method (e.g., "Sequential sampling"). Default is "Sequential sampling". |
| `ml_backend` | object | No | Configuration for connecting a Machine Learning backend. |
| `initial_data` | object | No | specificiation for importing initial tasks/data. |

### ML Backend Configuration (`ml_backend`)

If you want to connect a machine learning model to your project immediately upon creation, use the `ml_backend` object.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | **Yes** | The URL of the ML backend (e.g., `http://localhost:9090`). |
| `title` | string | No | Display name for the ML backend. |
| `description` | string | No | Description of the ML backend. |
| `is_interactive`| boolean| No | Whether the backend supports interactive pre-annotations. Default `false`. |
| `auth_method` | string | No | Authentication method. Options: `NONE`, `BASIC_AUTH`, `API_KEY`. Default `NONE`. |
| `basic_auth_user`| string | No | Username for Basic Auth. |
| `basic_auth_pass`| string | No | Password for Basic Auth. |

> **Note on `/train` Requirement**: To support the `train` action in the active learning loop, the ML backend at the specified `url` must implement a `POST /train` endpoint. See `ACTIVE_LEARNING.md` for implementation details.

### Initial Data Configuration (`initial_data`)

To import data automatically, use the `initial_data` object.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **Yes** | The source type. Supported values: `url`, `file`, `list`. |
| `source` | string/list | **Yes** | The location of the data. URL for `url` type, file path for `file` type, or a list of task objects for `list` type. |

## Examples

### 1. Basic Project with URL Data Import

```json
{
  "title": "Sentiment Analysis Project",
  "description": "Classifying tweets as positive or negative.",
  "label_config": "<View><Text name=\"text\" value=\"$text\"/><Choices name=\"sentiment\" toName=\"text\"><Choice value=\"Positive\"/><Choice value=\"Negative\"/></Choices></View>",
  "initial_data": {
    "type": "url",
    "source": "https://raw.githubusercontent.com/heartexlabs/label-studio/master/examples/sentiment_analysis/tasks.json"
  }
}
```

### 2. Project with ML Backend and Local File Data

```json
{
  "title": "Toxicity Detection",
  "description": "Detecting toxic comments with active learning.",
  "label_config": "/path/to/server/label_config.xml",
  "sampling": "Uncertainty sampling",
  "ml_backend": {
    "url": "http://localhost:9090",
    "title": "BERT Classifier",
    "description": "Fine-tuned BERT model for toxicity",
    "is_interactive": true
  },
  "initial_data": {
    "type": "file",
    "source": "./data/initial_batch.json"
  }
}
```

## Usage

To use this configuration file, ensure your client is initialized and call the creation method (assuming you are using the `trustai` CLI or script wrapper):

```python
helper.create_project_from_config("my_project_config.json")
```

## Note

When configuring the ML backend, you can also use the `/train` endpoint to trigger training jobs.
