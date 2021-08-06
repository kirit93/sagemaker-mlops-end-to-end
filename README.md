# An end to end demo on MLOps on SageMaker

### Prerequisites
The following prerequisites must be completed to replicate this project.

From the SageMaker Studio UI, create a SageMaker Project using one of the 1st party supported templates. Please use one of the following templates - 
* *MLOps template for model building, training, and deployment*
* *MLOps template for model building, training, and deployment with third-party Git repositories using Jenkins*
* *MLOps template for model building, training, and deployment with third-party Git repositories using CodePipeline*

*Note - It is recommended to use the first template which leverages SageMaker native services for Source Version Control and CI/CD. This reduces the number of steps needed for an end to end demo. If choosing the other templates, please follow the documentation to complete the template specific prerequisites. 

For more information on SageMaker Projects, visit the AWS [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-whatis.html). 

Once the project is created, you will see 2 repositories created - one for the model building code and one for the model deployment code. The model deployment code will remain untouched, the model building code will be changed by replacing `pipelines/abalone/` with the code from this repo under `sagemaker-pipeline/`

### DataWrangler and Feature Store Setup

* Upload `datawrangler/abalone-dataset-header.csv` to S3 and note the S3 URI
* Replace the S3 URI in line 19 of `datawrangler/fs-ingest-wrangler-training-template.flow` with the S3 URI you uploaded the dataset to in the step above
* Uplaod this `.flow` file to SageMaker Studio and open it. This will open up the DataWrangler UI. 
* On the DW UI, click on `Export Step` and select `Feature Store`. This will generate a notebook. 
* Run the code in the notebook generated to ingest features in to Feature Store. 

### Repository Structure

This repository contains 2 folders -
* `sagemaker-pipeline/`
    This folder contains a code to create and run a SageMaker Pipeline to process data, run a hyperparameter tuning job, evaluate the top model from the HPO job, use Clarify to generate a bias report, and register the model into a model registry. 
* `model-monitor`
    This folder contains notebooks for creating an endpoint from a model registered in the model registry and setting up a Data Quality Monitor. 

### Demo Setup

#### End to End Pipeline with Feature Store integration
* If using Feature Store, the first step of the Pipeline will need to read data from Feature Store. 
* In the file `sagemaker-pipeline/pipeline-dw-fs.py` lines 131 to 178 need to be replaced with the code in the notebook created by the DataWrangler Export. 
* The first step of the Pipeline will be `step_read_train`
* Replace the first step in `sagemaker-pipeline/pipeline.py` with `step_read_train` and `step_process` from `sagemaker-pipeline/pipeline-dw-fs.py`. 

IF NOT USING FEATURE STORE, IGNORE THE STEPS ABOVE AND FOLLOW THE BELOW STEPS. 

* Navigate to the model build repo created by the SageMaker Project, replace 
the code in `pipelines/abalone/` with the code in `sagemaker-pipeline/`. 
* Trigger the pipeline by pushing the new code to the CodeCommit/Git repo (depending on the template selected)
* Once the pipeline has completed, find the model package group in the Model Registry and find the ARN of the model package created in the group
* Approve the model in the model registry, this will trigger the model deployment pipeline, you should see an endpoint being created in SageMaker
* This endpoint will have the suffix `-staging`. You can navigate to CodePipeline, and under Pipelines you will see one with your project name and `model-deploy`. Click on that Pipeline and you will see a manual approval option. When approved, a new endpoint will be created with the suffix `-prod`. 
* These endpoints are created by the default seed code in the 1st party template and do not have Data Capture enabled. 

To setup Model Monitor
* Navigate to `model-monitor/create_endpoint.ipynb` to create an endpoint with DataCapture enabled
* Run `model-monitor/data_quality_monitor.ipynb` to set up a Data Quality Monitoring schedule on the endpoint. 

Once all the Endpoints have been created, navigate to the Endpoint UI in SageMaker Studio. Click on the endpoint deployed using the notebook in the model monitor folder, 

Things to highlight in the demo -
* End to end lineage
    * View of the trial component from the model in the Model Registry
    * Lineage from the Endpoint to the Model Package Group and Version
* Pipelines integration with Experiments
* Debugging a Pipeline through the DAG view
* CI/CD for automatic training and deployment

