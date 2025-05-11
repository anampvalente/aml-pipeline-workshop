---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: This sample shows how to use Azure ML pipelines with the Python SDK to train a model using the NYC taxi dataset.
---

# NYC Taxi Data Regression with Azure ML SDK

This is an end-to-end machine learning pipeline which runs a linear regression to predict taxi fares in NYC. The pipeline is made up of components, each serving different functions, which can be registered with the workspace, versioned, and reused with various inputs and outputs. 

## Pipeline Components

The pipeline consists of five key components:

* **Prepare Data**
  * This component takes the raw taxi dataset and cleans/preprocesses it.
  * Input: Raw data under `data/` folder (CSV files)
  * Output: Single cleaned dataset (.csv)

* **Transform Data**
  * This component creates features out of the taxi data to be used in training. 
  * Input: Filtered dataset from previous step (.csv)
  * Output: Dataset with engineered features (.csv)

* **Train Linear Regression Model**
  * This component splits the dataset into train/test sets and trains an sklearn Linear Regressor. 
  * Input: Data with feature set
  * Output: Trained model (mlflow_model) and data subset for test

* **Predict Taxi Fares**
  * This component uses the trained model to predict taxi fares on the test set.
  * Input: Linear regression model and test data from previous step
  * Output: Test data with predictions added as a column

* **Score Model**
  * This component scores the model based on how accurate the predictions are in the test set. 
  * Input: Test data with predictions and model
  * Output: Report with model coefficients and evaluation scores (.txt) 

## Running the Pipeline

### Option 1: Run the Jupyter Notebook

The notebook provides an interactive way to execute and visualize the pipeline:

1. Open the [nyc_taxi_data_regression.ipynb](nyc_taxi_data_regression.ipynb) notebook in Jupyter or Azure ML Studio.
2. Make sure you have a compute instance running.
3. Run each cell sequentially to:
   - Connect to your Azure ML workspace
   - Load the pipeline components
   - Build the pipeline
   - Submit and monitor the pipeline job

### Option 2: Run the Python Script

For automation or running outside a notebook environment:

1. Ensure you have the Azure ML SDK v2 installed:
   ```bash
   pip install azure-ai-ml azure-identity
   ```

2. Run the Python script directly:
   ```bash
   python nyc_taxi_data_regression.py
   ```

The script will:
- Authenticate with Azure
- Load all pipeline components from YAML files
- Configure the pipeline job
- Submit the pipeline to run on your Azure ML compute cluster
- Print a link to monitor the job in Azure ML Studio

## Environment Setup

If you need to create a Python environment for running the pipeline, use the provided `create_env.sh` script in the parent directory:

```bash
../create_env.sh
```

## Learn More

You can learn more about creating reusable components for your pipeline in the [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines?view=azureml-api-2).