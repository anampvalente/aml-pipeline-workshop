# Azure ML Pipeline Components with CLI v2

This folder demonstrates how to create and run Azure Machine Learning pipelines with reusable components using the Azure CLI v2. The example implements a NYC Taxi fare prediction pipeline using a modular component-based approach.

## Table of Contents

- [Component Overview](#component-overview)
- [Component Structure](#component-structure)
- [Pipeline Structure](#pipeline-structure)
- [Registering Components](#registering-components)
- [Running Pipelines](#running-pipelines)
- [Registered vs. Local Components](#registered-vs-local-components)
- [Best Practices](#best-practices)
- [Registered vs. Local Components](#registered-vs-local-components)
- [Best Practices](#best-practices)

## Component Overview

In Azure ML, components are self-contained pieces of code that perform a specific step in your machine learning workflow. This project includes five key components:

1. **Prep Component (`prep.yml`)**: Cleans and preprocesses raw taxi data
2. **Transform Component (`transform.yml`)**: Engineers features for model training
3. **Train Component (`train.yml`)**: Trains a regression model on the transformed data
4. **Predict Component (`predict.yml`)**: Uses the trained model to generate predictions
5. **Score Component (`score.yml`)**: Evaluates model performance using test data


## Component Structure

Each component follows the Azure ML component schema and consists of:

- YAML definition file (e.g., `prep.yml`)
- Source code folder (e.g., `prep_src/`)

### Component YAML Structure

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_taxi_data
display_name: Prepare Taxi Data
version: 1
type: command
inputs:
  raw_data:
    type: uri_folder
outputs:
  prep_data:
    type: uri_folder
code: ./prep_src
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
command: >-
  python prep.py 
  --raw_data ${{inputs.raw_data}} 
  --prep_data ${{outputs.prep_data}}
```

Key sections of a component definition:
- **name**: Unique identifier
- **inputs/outputs**: Data that flows in and out of the component
- **code**: Path to the source code folder
- **environment**: Computing environment specification
- **command**: Command to execute the component code

## Pipeline Structure

Pipelines connect multiple components together, defining the workflow of your machine learning process. This repository provides two pipeline examples:

### 1. Local Component Reference Pipeline (`pipeline.yml`)

This pipeline references component definitions stored locally in YAML files.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
jobs:
  prep_job:
    type: command
    component: ./prep.yml  # Local reference
    inputs:
      raw_data:
        type: uri_folder
        path: ./data
```

### 2. Registered Component Reference Pipeline (`pipeline_registered_components.yml`)

This pipeline references components that have been registered in your Azure ML workspace.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
jobs:
  prep_job:
    type: command
    component: azureml:prep_taxi_data@latest  # Registered reference
    inputs:
      raw_data:
        type: uri_folder
        path: ./data
```

## Registering Components

Components can be registered in your Azure ML workspace to be reused across different pipelines and projects. Registration creates a versioned asset that can be referenced by name.

### Steps to Register Components

1. Use the `az ml component create` command:

```bash
az ml component create --file ./prep.yml
az ml component create --file ./transform.yml
az ml component create --file ./train.yml
az ml component create --file ./predict.yml
az ml component create --file ./score.yml
```

2. Once registered, a component can be referenced in a pipeline using the format:
   `azureml:<component_name>@<version>`

The script `run-with-registered-components.sh` demonstrates how to register all components and then run a pipeline using these registered components.

## Running Pipelines

### Running Pipeline with Local Components

```bash
az ml job create --file pipeline.yml
```

### Running Pipeline with Registered Components

```bash
az ml job create --file ./pipeline_registered_components.yml
```

### Monitoring Job Progress

```bash
# Stream logs for a specific job
az ml job stream -n <job-name>

# Get job details
az ml job show -n <job-name>
```

## Registered vs. Local Components

### Local Components
- **Advantages**:
  - Easier for development and testing
  - Self-contained in your project
  - Changes take effect immediately
- **Disadvantages**: 
  - Not versioned or shareable across projects
  - Less governance and control

### Registered Components
- **Advantages**:
  - Versioned and trackable
  - Reusable across different pipelines and projects
  - Better governance and access control
  - Supports CI/CD workflows
- **Disadvantages**:
  - Requires additional steps to register and update

## Best Practices

1. **Version Your Components**: Always version your components for reproducibility
2. **Use Descriptive Names**: Choose clear, descriptive names for components and inputs/outputs
3. **Component Granularity**: Design components that perform a single logical task
4. **Define Dependencies Clearly**: Explicitly list all dependencies in your environment configurations
5. **Input Validation**: Add validation code for component inputs
6. **Error Handling**: Implement proper error handling in component code
7. **Documentation**: Document the purpose and usage of each component
8. **Testing**: Test components individually before combining them into pipelines

## Useful Commands

```bash
# List registered components
az ml component list

# Show details of a specific component
az ml component show --name <component_name> --version <version>

# List jobs
az ml job list

# Set default workspace for commands
az configure --defaults workspace=<workspace_name> group=<resource_group_name>
```

For more information, refer to the [official Azure ML component pipelines documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli?view=azureml-api-2).

## How to Run `run-with-registered-components.sh`

To run the script that registers components and executes the pipeline:

1. Make sure you have the Azure CLI and ML extension installed:
```bash
az extension add -n ml
```

2. Log in to Azure and set your default subscription:
```bash
az login
az account set -s <your-subscription-id>
```

3. Navigate to the CLI directory:
```bash
cd /home/azureuser/cloudfiles/code/Users/avalente/aml-pipeline-workshop/cli
```

4. Make the script executable:
```bash
chmod +x run-with-registered-components.sh
```

5. Execute the script:
```bash
./run-with-registered-components.sh
```

This script will first register all the components defined in the YAML files, and then create and run a pipeline job that uses the registered components.

## Sample Output

When you run a pipeline job, you'll see output similar to this:

```json
{
  "name": "nyc_taxi_data_regression_job",
      "command": "",
      "component": "azureml:df45efbf-8373-82fd-7d5e-56fa3cd31c05:1",
      "environment_variables": {},
      "inputs": {
        "training_data": "${{parent.jobs.transform-job.outputs.transformed_data}}"
      },
      "outputs": {
        "model_output": "${{parent.outputs.pipeline_job_trained_model}}",
        "test_data": "${{parent.outputs.pipeline_job_test_data}}"
      },
      "type": "command"
    },
    "transform-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:107ae7d3-7813-1399-34b1-17335735496c:1",
      "environment_variables": {},
      "inputs": {
        "clean_data": "${{parent.jobs.prep-job.outputs.prep_data}}"
      },
      "outputs": {
        "transformed_data": "${{parent.outputs.pipeline_job_transformed_data}}"
      },
      "type": "command"
    }
  },
  "name": "6cef8ff4-2bd3-4101-adf2-11e0b62e6f6d",
  "outputs": {
    "pipeline_job_predictions": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_prepped_data": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_score_report": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_test_data": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_trained_model": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_transformed_data": {
      "mode": "upload",
      "type": "uri_folder"
    }
  },
  "properties": {
    "azureml.continue_on_step_failure": "False",
    "azureml.git.dirty": "True",
    "azureml.parameters": "{}",
    "azureml.pipelineComponent": "pipelinerun",
    "azureml.runsource": "azureml.PipelineRun",
    "mlflow.source.git.branch": "march-cli-preview",
    "mlflow.source.git.commit": "8e28ab743fd680a95d71a50e456c68757669ccc7",
    "mlflow.source.git.repoURL": "https://github.com/Azure/azureml-examples.git",
    "runSource": "MFE",
    "runType": "HTTP"
  },
  "resourceGroup": "pipeline-pm",
  "services": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/6cef8ff4-2bd3-4101-adf2-11e0b62e6f6d?wsid=/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourcegroups/pipeline-pm/workspaces/pm-dev&tid=72f988bf-86f1-41af-91ab-2d7cd011db47",
      "type": "Studio"
    },
    "Tracking": {
      "endpoint": "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourceGroups/pipeline-pm/providers/Microsoft.MachineLearningServices/workspaces/pm-dev?",
      "type": "Tracking"
    }
  },
  "settings": {
    "continue_on_step_failure": false,
    "default_compute": "cpu-cluster",
    "default_datastore": "workspaceblobstore"
  },
  "status": "Preparing",
  "tags": {
    "azureml.Designer": "true"
  },
  "type": "pipeline"
}
```


