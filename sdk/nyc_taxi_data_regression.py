# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component


try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()



    # Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "cpu-cluster"
print(ml_client.compute.get(cluster_name))




parent_dir = ""

# 1. Load components
prepare_data = load_component(source=parent_dir + "./prep.yml")
transform_data = load_component(source=parent_dir + "./transform.yml")
train_model = load_component(source=parent_dir + "./train.yml")
predict_result = load_component(source=parent_dir + "./predict.yml")
score_data = load_component(source=parent_dir + "./score.yml")

# 2. Construct pipeline
@pipeline()
def nyc_taxi_data_regression(pipeline_job_input):
    """NYC taxi data regression example."""
    prepare_sample_data = prepare_data(raw_data=pipeline_job_input)
    transform_sample_data = transform_data(
        clean_data=prepare_sample_data.outputs.prep_data
    )
    train_with_sample_data = train_model(
        training_data=transform_sample_data.outputs.transformed_data
    )
    predict_with_sample_data = predict_result(
        model_input=train_with_sample_data.outputs.model_output,
        test_data=train_with_sample_data.outputs.test_data,
    )
    score_with_sample_data = score_data(
        predictions=predict_with_sample_data.outputs.predictions,
        model=train_with_sample_data.outputs.model_output,
    )
    return {
        "pipeline_job_prepped_data": prepare_sample_data.outputs.prep_data,
        "pipeline_job_transformed_data": transform_sample_data.outputs.transformed_data,
        "pipeline_job_trained_model": train_with_sample_data.outputs.model_output,
        "pipeline_job_test_data": train_with_sample_data.outputs.test_data,
        "pipeline_job_predictions": predict_with_sample_data.outputs.predictions,
        "pipeline_job_score_report": score_with_sample_data.outputs.score_report,
    }


pipeline_job = nyc_taxi_data_regression(
    Input(type="uri_folder", path=parent_dir + "./data/")
)
# demo how to change pipeline output settings
pipeline_job.outputs.pipeline_job_prepped_data.mode = "rw_mount"

# set pipeline level compute
pipeline_job.settings.default_compute = "cpu-cluster"
# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"



# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)
pipeline_job




# Wait until the job completes
ml_client.jobs.stream(pipeline_job.name)