"""
NYC Taxi Data Regression Pipeline

This script demonstrates how to create an Azure ML pipeline using reusable components
to process NYC taxi data, train a regression model, and evaluate its performance.
"""
# Import required libraries
import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component


def get_credentials():
    """Get Azure credentials with fallback authentication."""
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully
        credential.get_token("https://management.azure.com/.default")
        return credential
    except Exception:
        # Fall back to InteractiveBrowserCredential if DefaultAzureCredential fails
        return InteractiveBrowserCredential()


# Get workspace handle using appropriate credentials
credential = get_credentials()
ml_client = MLClient.from_config(credential=credential)

# Retrieve an already attached Azure Machine Learning Compute
cluster_name = "cpu-cluster"
print(f"Using compute cluster: {cluster_name}")
print(ml_client.compute.get(cluster_name))




def load_pipeline_components():
    """Load all pipeline components from YAML files."""
    # Use absolute paths to component YAML files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define component file paths
    component_files = {
        "prepare_data": os.path.join(current_dir, "prep.yml"),
        "transform_data": os.path.join(current_dir, "transform.yml"),
        "train_model": os.path.join(current_dir, "train.yml"),
        "predict_result": os.path.join(current_dir, "predict.yml"),
        "score_data": os.path.join(current_dir, "score.yml")
    }
    
    # Validate that files exist
    for name, file_path in component_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Component file not found: {file_path}")
        print(f"Found component file: {file_path}")
    
    # Load components
    components = {
        name: load_component(source=file_path) 
        for name, file_path in component_files.items()
    }
    
    return components


# Load all components
components = load_pipeline_components()


@pipeline()
def nyc_taxi_data_regression(pipeline_job_input):
    """NYC taxi data regression example.
    
    This pipeline performs the following steps:
    1. Data preparation - Clean and preprocess raw data
    2. Data transformation - Transform data for model training
    3. Model training - Train a regression model
    4. Prediction - Generate predictions using the trained model
    5. Evaluation - Score the model performance
    
    Args:
        pipeline_job_input: Input folder containing raw data
        
    Returns:
        Dictionary of pipeline outputs including model and evaluation results
    """
    # Step 1: Data preparation
    prepare_sample_data = components["prepare_data"](raw_data=pipeline_job_input)
    
    # Step 2: Data transformation
    transform_sample_data = components["transform_data"](
        clean_data=prepare_sample_data.outputs.prep_data
    )
    
    # Step 3: Model training
    train_with_sample_data = components["train_model"](
        training_data=transform_sample_data.outputs.transformed_data
    )
    
    # Step 4: Prediction
    predict_with_sample_data = components["predict_result"](
        model_input=train_with_sample_data.outputs.model_output,
        test_data=train_with_sample_data.outputs.test_data,
    )
    
    # Step 5: Evaluation
    score_with_sample_data = components["score_data"](
        predictions=predict_with_sample_data.outputs.predictions,
        model=train_with_sample_data.outputs.model_output,
    )
    
    # Return all pipeline outputs
    return {
        "pipeline_job_prepped_data": prepare_sample_data.outputs.prep_data,
        "pipeline_job_transformed_data": transform_sample_data.outputs.transformed_data,
        "pipeline_job_trained_model": train_with_sample_data.outputs.model_output,
        "pipeline_job_test_data": train_with_sample_data.outputs.test_data,
        "pipeline_job_predictions": predict_with_sample_data.outputs.predictions,
        "pipeline_job_score_report": score_with_sample_data.outputs.score_report,
    }


def configure_pipeline_job(pipeline_function, input_path):
    """Configure pipeline job with compute and datastore settings.
    
    Args:
        pipeline_function: The pipeline function to execute
        input_path: Path to input data
        
    Returns:
        Configured pipeline job ready for submission
    """
    # Create pipeline job with input data
    pipeline_job = pipeline_function(
        Input(type="uri_folder", path=input_path)
    )
    
    # Configure pipeline output settings
    pipeline_job.outputs.pipeline_job_prepped_data.mode = "rw_mount"
    
    # Set pipeline level compute
    pipeline_job.settings.default_compute = "cpu-cluster"
    
    # Set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    
    return pipeline_job


def submit_pipeline_job(ml_client, pipeline_job, experiment_name):
    """Submit pipeline job to workspace and monitor execution.
    
    Args:
        ml_client: Azure ML client
        pipeline_job: Configured pipeline job
        experiment_name: Name of the experiment
        
    Returns:
        The submitted pipeline job
    """
    print(f"Submitting pipeline job to experiment: {experiment_name}")
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )
    
    print(f"Pipeline job submitted. Job name: {submitted_job.name}")
    print(f"Job details available at: {submitted_job.studio_url}")
    
    # Wait for the job to complete
    print("Streaming job logs...")
    ml_client.jobs.stream(submitted_job.name)
    
    return submitted_job


def main():
    """Main execution function."""
    # Configure the pipeline job with absolute path to data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    
    # Verify data directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    print(f"Using data from: {data_path}")
    
    # List data files to verify content
    data_files = os.listdir(data_path)
    if not data_files:
        raise ValueError(f"No data files found in {data_path}")
    print(f"Found {len(data_files)} data files: {', '.join(data_files[:5])}")
    
    pipeline_job = configure_pipeline_job(
        nyc_taxi_data_regression, 
        input_path=data_path
    )
    
    # Submit and monitor the job
    submit_pipeline_job(
        ml_client=ml_client,
        pipeline_job=pipeline_job, 
        experiment_name="pipeline_samples"
    )


# Execute the script
if __name__ == "__main__":
    main()