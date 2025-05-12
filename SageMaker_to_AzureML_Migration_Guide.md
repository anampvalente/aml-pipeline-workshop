# SageMaker to Azure ML Migration Guide

This guide provides detailed technical steps to help you migrate machine learning pipelines from Amazon SageMaker to Azure Machine Learning.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Comparison](#architecture-comparison)
3. [Authentication & Access Control](#authentication--access-control)
4. [Data Storage & Access](#data-storage--access)
5. [Pipeline Component Conversion](#pipeline-component-conversion)
6. [Environment Setup](#environment-setup)
7. [Step-by-Step Migration Process](#step-by-step-migration-process)
8. [Model Deployment Comparison](#model-deployment-comparison)
9. [Monitoring & Logging](#monitoring--logging)
10. [Cost Management](#cost-management)

## Prerequisites

Before starting your migration:

- **Azure Subscription**: Create an Azure account and subscription
- **Azure Machine Learning Workspace**: Set up an AML workspace with storage, key vault, and application insights
- **Azure CLI & ML Extension**: Install Azure CLI and ML extension for command-line operations
- **Python Environment**: Set up a Python environment with Azure ML SDK v2
- **Development Tools**: Configure VS Code with Azure ML extension or use Azure ML Studio

## Architecture Comparison

### SageMaker Architecture

```
AWS Account
  └── SageMaker
       ├── Processing Jobs (data processing)
       ├── Training Jobs (model training)
       ├── Hyperparameter Tuning Jobs (tuning)
       ├── Model Registry (model management)
       ├── Pipelines (workflow orchestration)
       └── Endpoints (model serving)
```

### Azure ML Architecture

```
Azure Subscription
  └── Resource Group
       └── Azure ML Workspace
            ├── Components (reusable steps)
            ├── Data (datasets and datastores)
            ├── Environments (Docker containers)
            ├── Jobs (pipeline & component execution)
            ├── Models (model registry)
            └── Endpoints (model serving)
```

## Authentication & Access Control

### SageMaker to Azure ML Mapping

| SageMaker | Azure ML |
|-----------|----------|
| AWS IAM Roles | Microsoft Entra ID (Azure AD) |
| IAM Policies | Azure RBAC (Role-Based Access Control) |
| SageMaker Execution Role | Managed Identity |
| AWS Secrets Manager | Azure Key Vault |

### Authentication Code Migration

SageMaker:
```python
import boto3
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
sagemaker_session = sagemaker.Session()
```

Azure ML:
```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
```

## Data Storage & Access

### Storage Mapping

| SageMaker | Azure ML |
|-----------|----------|
| Amazon S3 | Azure Blob Storage / Azure Data Lake Storage |
| S3 URIs (s3://bucket/path) | Azure URIs (azureml://datastores/name/paths/path) |
| SageMaker Data Wrangler | Azure ML Data Wrangler |
| SageMaker Feature Store | Azure Feature Store |

### Data Access Code Migration

SageMaker:
```python
# Access data from S3
s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data="s3://bucket/training-data",
    content_type="csv"
)
```

Azure ML:
```python
# Access data from Azure Storage
from azure.ai.ml import Input

input_data = Input(
    type="uri_folder",
    path="azureml://datastores/workspaceblobstore/paths/training-data"
)
```

## Pipeline Component Conversion

### SageMaker Processing to Azure ML Component

SageMaker:
```python
from sagemaker.sklearn.processing import SKLearnProcessor

sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1
)

processing_step = ProcessingStep(
    name="preprocess-data",
    processor=sklearn_processor,
    inputs=[ProcessingInput(
        source="s3://bucket/raw-data",
        destination="/opt/ml/processing/input"
    )],
    outputs=[ProcessingOutput(
        output_name="processed_data",
        source="/opt/ml/processing/output",
        destination="s3://bucket/processed-data"
    )],
    code="preprocessing.py"
)
```

Azure ML:
```yaml
# prep.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: preprocess_data
version: 1
type: command
inputs:
  raw_data:
    type: uri_folder
outputs:
  processed_data:
    type: uri_folder
environment: azureml://registries/azureml/environments/sklearn-1.0/labels/latest
code: ./prep_src
command: >-
  python prep.py 
  --raw_data ${{inputs.raw_data}} 
  --processed_data ${{outputs.processed_data}}
```

### SageMaker Training to Azure ML Component

SageMaker:
```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="train.py",
    framework_version="0.23-1",
    instance_type="ml.m5.large",
    role=role,
    hyperparameters={'test-split-ratio': 0.2}
)

training_step = TrainingStep(
    name="train-model",
    estimator=sklearn_estimator,
    inputs={
        "training": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)
```

Azure ML:
```yaml
# train.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: train_linear_regression_model
version: 1
type: command
inputs:
  training_data: 
    type: uri_folder
  test_split_ratio:
    type: number
    default: 0.2
outputs:
  model_output:
    type: mlflow_model
  test_data:
    type: uri_folder
code: ./train_src
environment: azureml://registries/azureml/environments/sklearn-1.5/labels/latest
command: >-
  python train.py 
  --training_data ${{inputs.training_data}} 
  --test_data ${{outputs.test_data}} 
  --model_output ${{outputs.model_output}}
  --test_split_ratio ${{inputs.test_split_ratio}}
```

## Environment Setup

### SageMaker Environment to Azure ML Environment

SageMaker:
```python
# Use pre-built container
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="script.py",
    framework_version="0.23-1",
    py_version="py3",
    instance_type="ml.m5.large"
)

# Custom container
from sagemaker.estimator import Estimator

custom_estimator = Estimator(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)
```

Azure ML:
```python
# Use pre-built environment
from azure.ai.ml.entities import Environment

curated_env = Environment(
    image="mcr.microsoft.com/azureml/sklearn-1.0:latest"
)

# Or reference a registered environment
registered_env = "azureml://registries/azureml/environments/sklearn-1.0/labels/latest"

# Custom Docker environment
custom_env = Environment(
    image="myregistry.azurecr.io/my-custom-image:latest",
    conda_file="./conda.yml"
)
```

## Step-by-Step Migration Process

### 1. Set up Azure ML Workspace

```python
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# Connect to subscription
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group>",
    workspace_name="<workspace_name>"
)
```

### 2. Create Data Assets

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Create data asset
my_data = Data(
    path="https://azuremlexamples.blob.core.windows.net/datasets/taxi-data.csv",
    type=AssetTypes.URI_FILE,
    description="NYC Taxi dataset",
    name="nyc-taxi-data",
    version="1"
)

ml_client.data.create_or_update(my_data)
```

### 3. Define Components (from SageMaker steps)

For each SageMaker step, create a corresponding Azure ML component:

1. Create a YAML file for component definition
2. Implement the component code in Python
3. Register the component with Azure ML

Example:
```bash
az ml component create --file ./train.yml
```

### 4. Define Pipeline

Using the SDK:
```python
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml import load_component

# Load components
prepare_data_component = load_component(source="./prep.yml")
train_model_component = load_component(source="./train.yml")

# Define pipeline
@dsl.pipeline(
    name="nyc-taxi-pipeline",
    description="NYC taxi data pipeline"
)
def nyc_taxi_pipeline(pipeline_input_data):
    prepare_data_step = prepare_data_component(
        raw_data=pipeline_input_data
    )
    
    train_model_step = train_model_component(
        training_data=prepare_data_step.outputs.processed_data
    )
    
    return {
        "pipeline_trained_model": train_model_step.outputs.model_output
    }

# Create pipeline job
pipeline_job = nyc_taxi_pipeline(
    pipeline_input_data=Input(type="uri_folder", path="path/to/data")
)

# Submit pipeline job
ml_client.jobs.create_or_update(pipeline_job)
```

Using CLI (YAML):
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: nyc_taxi_data_regression
jobs:
  prep_job:
    type: command
    component: ./prep.yml
    inputs:
      raw_data:
        type: uri_folder
        path: ./data
    outputs:
      processed_data: 
  train_job:
    type: command
    component: ./train.yml
    inputs:
      training_data: ${{parent.jobs.prep_job.outputs.processed_data}}
```

### 5. Execute Pipeline

```bash
az ml job create --file ./pipeline.yml
```

## Model Deployment Comparison

### Model Registration

SageMaker:
```python
# Register model in SageMaker Model Registry
model_package_group_name = "nyc-taxi-model-group"
model_package = sagemaker.ModelPackage(
    model_package_group_name=model_package_group_name,
    model_metrics=model_metrics,
    approval_status="Approved",
    model_package_description="NYC Taxi Fare Prediction"
)
model_package.create()
```

Azure ML:
```python
# Register model in Azure ML Model Registry
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path="azureml://jobs/<job-id>/outputs/model",
    name="nyc-taxi-model",
    description="NYC Taxi Fare Prediction Model",
    type=AssetTypes.MLFLOW_MODEL
)

ml_client.models.create_or_update(model)
```

### Model Deployment

SageMaker:
```python
# Deploy model to SageMaker endpoint
predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name="nyc-taxi-endpoint"
)
```

Azure ML:
```python
# Deploy model to Azure ML online endpoint
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="nyc-taxi-endpoint",
    description="NYC Taxi Prediction Endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="nyc-taxi-deployment",
    endpoint_name=endpoint.name,
    model="azureml:nyc-taxi-model:1",
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```

## Monitoring & Logging

### Logging & Metrics

SageMaker:
```python
# Log metrics to CloudWatch
import boto3
import time

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    MetricData=[
        {
            'MetricName': 'ModelAccuracy',
            'Dimensions': [{'Name': 'ModelVersion', 'Value': '1.0'}],
            'Value': 0.92,
            'Unit': 'None',
            'Timestamp': time.time()
        }
    ],
    Namespace='SageMakerCustomMetrics'
)
```

Azure ML:
```python
# Log metrics with MLflow
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.92)
mlflow.end_run()
```

### Data Drift Monitoring

SageMaker:
```python
# Set up Model Monitor
from sagemaker.model_monitor import DataCaptureConfig, ModelMonitor

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri="s3://bucket/monitor-data"
)

model_monitor = ModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)
```

Azure ML:
```python
# Set up Data Drift Monitor
from azure.ai.ml.entities import DataDriftMonitor

data_drift_monitor = DataDriftMonitor(
    name="nyc-taxi-drift-monitor",
    target_data="azureml:nyc-taxi-train-data:1",
    reference_data="azureml:nyc-taxi-test-data:1",
    compute="cpu-cluster",
    frequency="Day",
    latency=2,
    drift_threshold=0.3,
    features_to_monitor=["distance", "passengers"]
)

ml_client.data_drift_monitors.begin_create_or_update(data_drift_monitor).result()
```

## Cost Management

### Compute Cost Optimization

#### SageMaker to Azure ML Compute Mapping

| SageMaker Instance | Azure ML VM Series |
|-------------------|--------------------|
| ml.t3.medium (2 vCPU, 4 GB) | Standard_DS1_v2 |
| ml.m5.large (2 vCPU, 8 GB) | Standard_DS2_v2 |
| ml.m5.xlarge (4 vCPU, 16 GB) | Standard_DS3_v2 |
| ml.m5.2xlarge (8 vCPU, 32 GB) | Standard_DS4_v2 |
| ml.m5.4xlarge (16 vCPU, 64 GB) | Standard_DS5_v2 |
| ml.g4dn.xlarge (4 vCPU, 16 GB, 1 GPU) | Standard_NC6s_v3 |

### Cost Saving Strategies

1. **Use compute clusters with autoscaling**
2. **Implement Azure spot instances** (similar to SageMaker Spot)
3. **Schedule cluster shutdowns** during non-business hours
4. **Use lower-cost storage tiers** for infrequently accessed data
5. **Implement data lifecycle policies**
6. **Monitor and analyze costs** using Azure Cost Management

```python
# Create compute cluster with autoscaling
from azure.ai.ml.entities import AmlCompute

compute_cluster = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=1800  # 30 minutes
)

ml_client.begin_create_or_update(compute_cluster).result()
```

---

This migration guide provides a comprehensive framework for transitioning your machine learning workloads from Amazon SageMaker to Azure Machine Learning. For assistance with complex migration scenarios, contact your Azure account team or Azure ML specialists.
