# Azure Machine Learning Pipeline Workshop

This repository demonstrates how to create and manage Azure Machine Learning pipelines using reusable components. It includes examples for both SDK and CLI approaches, focusing on a regression pipeline that processes NYC taxi data, trains a model, and evaluates its performance.

## Prerequisites

- **Azure Account**: An active Azure subscription ([Create one for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F))
- **Azure ML Workspace**: A configured Azure Machine Learning workspace with a compute cluster ([Configuration guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2))
- **Python Environment**: Python 3.7+ with Azure ML SDK v2 installed ([Installation instructions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment?view=azureml-api-2))

## Repository Structure

```
aml-pipeline-workshop/
├── README.md                         # Main repository documentation
├── cli/                              # CLI-based implementation
│   ├── prep.yml                      # Data preparation component
│   ├── transform.yml                 # Data transformation component
│   ├── train.yml                     # Model training component
│   ├── predict.yml                   # Prediction component
│   ├── score.yml                     # Evaluation component
│   ├── pipeline.yml                  # Pipeline definition
│   └── data/                         # Sample data
├── sdk/                              # SDK-based implementation
│   ├── nyc_taxi_data_regression.ipynb # Jupyter notebook example
│   ├── prep.yml                      # Data preparation component
│   ├── transform.yml                 # Data transformation component
│   ├── train.yml                     # Model training component
│   ├── predict.yml                   # Prediction component
│   ├── score.yml                     # Evaluation component
│   └── data/                         # Sample data
```

## Pipeline Overview

The pipeline consists of five reusable components:

1. **Data Preparation**: Cleans and preprocesses raw NYC taxi data.
2. **Data Transformation**: Transforms the cleaned data for model training.
3. **Model Training**: Trains a regression model using the transformed data.
4. **Prediction**: Uses the trained model to make predictions.
5. **Evaluation**: Evaluates the model's performance.

## Running the Pipeline

### Using the Azure CLI

1. **Register Components**:
   ```bash
   az ml component create --file ./cli/prep.yml
   az ml component create --file ./cli/transform.yml
   az ml component create --file ./cli/train.yml
   az ml component create --file ./cli/predict.yml
   az ml component create --file ./cli/score.yml
   ```

2. **Run the Pipeline**:
   ```bash
   az ml job create --file ./cli/pipeline.yml
   ```

3. **Monitor the Job**:
   ```bash
   az ml job stream -n <job-name>
   ```

### Using the Python SDK

1. Open the Jupyter notebook `sdk/nyc_taxi_data_regression.ipynb` or run the python file `nyc_taxi_data_regression.py`
2. Follow the steps to:
   - Connect to your Azure ML workspace.
   - Load components from YAML files.
   - Build and submit the pipeline.
   - Monitor the pipeline job.

## Best Practices

- **Version Components**: Always version your components for reproducibility.
- **Use Managed Identities**: Securely authenticate without hardcoding credentials.
- **Optimize Compute Resources**: Choose appropriate compute targets for each component.
- **Enable Logging**: Use logging to monitor and debug pipeline execution.
- **Set Timeouts**: Avoid resource overuse by setting timeouts for jobs.

## References

- [Azure ML Component Pipelines Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli?view=azureml-api-2)
- [Azure ML Python SDK Reference](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-ml-readme)
- [Component YAML Schema Reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-component](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-component-pipeline?view=azureml-api-2)

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
