#!/usr/bin/env python
# coding: utf-8

"""
CIFAR-10 Pipeline Example using Azure ML SDK v2

This script demonstrates how to create and run a pipeline in Azure Machine Learning
that processes the CIFAR-10 dataset using PyTorch. The pipeline consists of three steps:
1. Get data: Download and extract the CIFAR-10 dataset
2. Train model: Train a convolutional neural network on the dataset
3. Evaluate model: Evaluate the trained model on the test dataset

Requirements:
- An Azure account with an active subscription
- An Azure ML workspace with computer cluster
- Azure Machine Learning Python SDK v2
"""

# 1. Import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# 2. Configure credential for authentication
def get_credential():
    """
    Get credential for Azure authentication with fallback to interactive login.
    Returns a credential object for Azure authentication.
    """
    try:
        # Try to use DefaultAzureCredential (supports multiple authentication methods)
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully
        credential.get_token("https://management.azure.com/.default")
        return credential
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential if DefaultAzureCredential doesn't work
        print(f"DefaultAzureCredential failed: {ex}")
        print("Falling back to InteractiveBrowserCredential")
        return InteractiveBrowserCredential()

# 3. Connect to Azure ML workspace
def get_ml_client():
    """
    Get a handle to the Azure ML workspace.
    Returns an MLClient object connected to the workspace.
    """
    credential = get_credential()
    ml_client = MLClient.from_config(credential=credential)
    return ml_client

# 4. Define the pipeline
def define_pipeline(ml_client):
    """
    Define the CIFAR-10 pipeline with three components:
    - get_data: Download and extract CIFAR-10 dataset
    - train_model: Train a CNN model on the dataset 
    - eval_model: Evaluate the trained model
    
    Args:
        ml_client: MLClient object connected to workspace
        
    Returns:
        A configured pipeline job object ready for submission
    """
    # Load the component definitions from YAML files
    parent_dir = "."
    get_data_func = load_component(source=parent_dir + "/get-data.yml")
    train_model_func = load_component(source=parent_dir + "/train-model.yml")
    eval_model_func = load_component(source=parent_dir + "/eval-model.yml")

    # Define pipeline function
    @pipeline()
    def train_cifar_10_with_pytorch():
        """CIFAR-10 Pipeline Example."""
        # define the job to get data
        get_data = get_data_func(
            cifar_zip=Input(
                path="wasbs://datasets@azuremlexamples.blob.core.windows.net/cifar-10-python.tar.gz",
                type="uri_file",
            )
        )
        get_data.outputs.cifar.mode = "upload"

        # define the job to train the model
        train_model = train_model_func(epochs=1, cifar=get_data.outputs.cifar)
        train_model.compute = "cpu-cluster"
        train_model.outputs.model_dir.mode = "upload"

        # define the job to evaluate the model
        eval_model = eval_model_func(
            cifar=get_data.outputs.cifar, model_dir=train_model.outputs.model_dir
        )
        eval_model.compute = "cpu-cluster"
        
        return eval_model

    # Create the pipeline job
    pipeline_job = train_cifar_10_with_pytorch()
    
    # Set pipeline level compute
    pipeline_job.settings.default_compute = "cpu-cluster"
    
    return pipeline_job

# 5. Submit and monitor the pipeline
def submit_pipeline_job(ml_client, pipeline_job):
    """
    Submit the pipeline job to Azure ML and monitor its progress.
    
    Args:
        ml_client: MLClient object connected to workspace
        pipeline_job: The pipeline job to submit
        
    Returns:
        The submitted pipeline job
    """
    # Submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="pipeline_samples"
    )
    print(f"Pipeline job submitted with name: {pipeline_job.name}")
    
    # Monitor the job until completion
    ml_client.jobs.stream(pipeline_job.name)
    
    return pipeline_job

# 6. Main function
def main():
    """
    Main function to execute the pipeline.
    """
    try:
        # Get ML client
        ml_client = get_ml_client()
        
        # Verify compute cluster is available
        cluster_name = "cpu-cluster"
        print("Retrieving compute cluster information...")
        compute = ml_client.compute.get(cluster_name)
        print(f"Using compute cluster: {compute.name} ({compute.type})")
        
        # Define the pipeline
        print("Defining pipeline...")
        pipeline_job = define_pipeline(ml_client)
        
        # Submit and monitor the pipeline
        print("Submitting pipeline job...")
        submitted_job = submit_pipeline_job(ml_client, pipeline_job)
        
        print(f"Pipeline job completed with status: {submitted_job.status}")
        
    except Exception as e:
        print(f"Error executing pipeline: {e}")
        raise

# Execute if run as a script
if __name__ == "__main__":
    main()
