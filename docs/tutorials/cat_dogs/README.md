# Cat & Dog Active Learning Tutorial

This tutorial guides you through setting up a complete Active Learning environment using Label Studio, a custom ML backend, and the TrustAI Active Learning Loop.

The goal is to classify images of Cats and Dogs using an interactive learning process where the model queries the user for the most uncertain samples.

## Prerequisites

- **OS**: Linux (Ubuntu recommended) or macOS
- **Software**:
  - Python 3.8+
  - Docker & Docker Compose
  - `curl`, `unzip`
- **Ports**: Ensure ports `8080` (Label Studio), `9090` (ML Backend), and `8000` (Image Server) are free.

---

## 1. Environment Setup

### 1.1 Clone the Repository
Clone the repository and navigate to the tutorial directory:

```bash
git clone https://github.com/fhstp/trustai-platform
cd trustai
```

### 1.2 Install Python Dependencies
It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r docs/tutorials/cat_dogs/dog_cat_backend/requirements-base.txt
```

### 1.3 Install Docker (If not installed)
If you don't have Docker installed, you can use the provided script (Ubuntu only):

```bash
cd docs/tutorials/cat_dogs
./install_docker.sh
# You may need to log out and back in for group changes to take effect
```

---

## 2. Dataset Setup

Download and prepare the Kaggle Cats and Dogs dataset. This script downloads the dataset, cleans it, and sets up the directory structure.

```bash
cd docs/tutorials/cat_dogs
./setup_dataset.sh
```

---

## 3. Starting the Services

We need to start three components:
1. **File Server**: Serves images to Label Studio (Port 8000).
2. **ML Backend**: The machine learning model server (Port 9090).
3. **Label Studio**: The annotation interface (Port 8080).

You can start all of them using the helper script:

```bash
cd docs/tutorials/cat_dogs
./start_all.sh
```

> **Note**: This script runs processes in the background and starts a Docker container. Keep this terminal open or monitor the processes.

---

## 4. Label Studio Configuration

### 4.1 Account Setup
1. Open your browser and go to `http://localhost:8080`.
2. Register a new account.
3. Retrieve your API Token:
   - Got to **Organization** -> **API Token Settings** (top right button) and enable **Legacy Tokens**
   - Go to **Account & Settings** (top right icon) -> **Legacy Token**.
   - Copy the **Access Token**.

### 4.2 Application Configuration
We need to configure the tutorial scripts to talk to your Label Studio instance.

1. Create a `.env` file in `docs/tutorials/cat_dogs/`:

```bash
cd docs/tutorials/cat_dogs
touch .env
```

2. Add the following content to `.env`:

```env
API_BASE_URL=http://localhost:8080
API_TOKEN=your_copied_token_here
PROJECT_ID=1
ML_BACKEND_URL=http://localhost:9090
IMAGE_BASE_URL=http://localhost:8000
```

### 4.3 Create the Project
1. In Label Studio, click **Create Project**.
2. **Project Name**: `Cat vs Dog`
3. **Labeling Setup**:
   - Go to **Labeling Interface** -> **Code**.
   - Replace the configuration with the following XML:

```xml
<View>
  <Choices name="choice" toName="image">
    <Choice value="Dog"/>
    <Choice value="Cat" />
  </Choices>
  <Image name="image" value="$image"/>
  <Image name="explanation" value="$explanation"/>
  <BrushLabels name="tag" toName="explanation">
	  <Label value="missing foreground" background="#33d17a"/>
	  <Label value="wrong background" background="#D4380D"/>
  </BrushLabels>
</View>
```

4. Click **Save**.
5. Note the Project ID from the URL (e.g., `/projects/1`) and obtain it. Update `PROJECT_ID` in your `.env` file if it's not `1`.

### 4.4 Import Data
Run the import script to add image tasks to the project. This script serves images from your local server.

```bash
cd docs/tutorials/cat_dogs
./import_data.sh
```

### 4.5 Connect ML Backend
1. In your project, go to **Settings** -> **Machine Learning**.
2. Click **Add Model**.
3. **Title**: `CatDog Backend`
4. **URL**: `http://localhost:9090`
5. Toggle **Interactive preannotations** to ON.
6. Click **Validate and Save**.

---

## 5. Running Active Learning

Now that the infrastructure is ready, we can use the `trustai` Active Learning Runner.

### 5.1 Manual Predictions 

In order to use active learning, we first have to calculate the uncertainties associated with each data point. You can do this using the `run_predictions.py` script. This is also useful for pre-populating the interface with model suggestions or testing the model's performance on the entire dataset.

**Prerequisites**:
- Ensure all services are running (see Section 3).
- Ensure your Python environment is activated.

```bash
cd docs/tutorials/cat_dogs
python run_predictions.py
```

This script will:
1. Fetch all tasks from Label Studio.
2. Send them to the ML Backend for inference.
3. Upload the predictions back to Label Studio, where they will appear as pre-annotations.


### 5.2 Configure the Runner
The main runner in the root directory needs its own `config.json`.

1. Go to the root of the repository:
   ```bash
   cd ../..
   ```
2. Create or edit `config.json`:

```json
{
  "api_base_url": "http://localhost:8080",
  "api_key": "your_copied_token_here"
}
```

### 5.3 Start the Loop
Run the active learning script. This script acts as the "Manager", identifying uncertain samples and orchestrating the workflow.

```bash
python -m trustai.run_active_learning
```

Follow the interactive prompts:
1. Select your **Cat vs Dog** project.
2. The system will retrieve uncertainty scores from the model.
3. It will present the most uncertain tasks for you to annotate in the browser.
4. Annotate the images in Label Studio.
5. Repeat!

