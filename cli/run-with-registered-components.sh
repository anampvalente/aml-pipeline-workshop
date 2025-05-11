#!/bin/bash
# This script demonstrates how to register components and run a pipeline using registered components

# Step 1: Register all components
echo "Registering components..."
az ml component create --file ./prep.yml
az ml component create --file ./transform.yml
az ml component create --file ./train.yml
az ml component create --file ./predict.yml
az ml component create --file ./score.yml

# Step 2: Run the pipeline using the registered components
echo "Running pipeline with registered components..."
az ml job create --file ./pipeline_registered_components.yml

echo "Done! Check the Azure ML Studio for job progress."
