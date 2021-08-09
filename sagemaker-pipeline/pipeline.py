from sagemaker.estimator import Estimator  
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor
)
from sagemaker import Model
from sagemaker.xgboost import XGBoostPredictor

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline, PipelineExperimentConfig, ExecutionVariables
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig,
    TuningStep
)
from sagemaker.workflow.step_collections import RegisterModel, CreateModelStep
from sagemaker.debugger import ProfilerRule, Rule, rule_configs

from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
)

import sagemaker
import os
import boto3
import sagemaker.session

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="mpg-e2e-d96a4b92",
    pipeline_name="pipeline-e2e-d96a4b92",
    base_job_prefix="job-e2e-d96a4b92",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    bucket = default_bucket
    prefix = 'sm-end-to-end-d96a4b92'
    processed_data_path =  ParameterString(
        name="ProcessedData",
        default_value=f"s3://{bucket}/{prefix}/processed_data_sklearn",
    )
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    # Train XGBoost Model (use Debugger for Profiling)

    model_path = f"s3://{bucket}/{prefix}/{base_job_prefix}/AbaloneTrain"

    # Debugger rules
    rules=[
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
        ProfilerRule.sagemaker(rule_configs.CPUBottleneck()),
        ProfilerRule.sagemaker(rule_configs.OverallSystemUsage()),
        Rule.sagemaker(rule_configs.create_xgboost_report()),
        Rule.sagemaker(rule_configs.overfit())
    ]
    # the debugger rule output path is  estimator.output_path + estimator.latest_training_job.job_name + "/rule-output"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{prefix}/{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
        rules=rules
    )
    
    #### SageMaker Hyperparameter Tuning

    xgb_train.set_hyperparameters(
        eval_metric="rmse",
        objective="reg:squarederror",  # Define the object metric for the training job
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    objective_metric_name = "validation:rmse"

    hyperparameter_ranges = {
        "alpha": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
        "lambda": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
    }

    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy="Random",
        objective_type="Minimize",
    )

    step_train = TuningStep(
        name="HPTuning",
        tuner=tuner_log,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )
    
    model_bucket_key = f"{bucket}/{prefix}/{base_job_prefix}/AbaloneTrain"
    xgb_model = Model(
        image_uri=image_uri,
        model_data=step_train.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        sagemaker_session=sagemaker_session,
        role=role,
        predictor_cls=XGBoostPredictor,
    )

    step_create_model = CreateModelStep(
        name="CreateTopModelFromHPO",
        model=xgb_model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m4.large"),
        depends_on=[step_train.name]
    )
    
    analysis_config_path = f"s3://{bucket}/{prefix}/clarify_config"

    config_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{prefix}/{base_job_prefix}/clarify-config",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_clarify_config = ProcessingStep(
        name = "CreateClarifyConfig",
        processor = config_processor,
        job_arguments=["--model-name",step_create_model.properties.ModelName,],
        code = os.path.join(BASE_DIR, "create_config.py"),
        outputs = [
            ProcessingOutput(output_name="analysis_config", source="/opt/ml/processing/clarify", destination = analysis_config_path),
        ],
        depends_on = [step_create_model.name]
    )
    
    # # Detect Bias using Clarify

    bias_report_output_path = f"s3://{bucket}/{prefix}/clarify-output/bias"
    clarify_instance_type = 'ml.c5.xlarge'

    clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type=clarify_instance_type,
        sagemaker_session=sagemaker_session,
    )

    data_config = sagemaker.clarify.DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=bias_report_output_path,
        label=0,
        headers= [
            "rings",
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
            "sex_M",
            "sex_I",
            "sex_F"
          ],
        dataset_type="text/csv",
    )
    
    config_input = ProcessingInput(
        input_name="analysis_config",
        source= analysis_config_path + '/analysis_config.json',
        destination="/opt/ml/processing/input/config",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_compression_type="None",
    )

    data_input = ProcessingInput(
        input_name="dataset",
        source=data_config.s3_data_input_path,
        destination="/opt/ml/processing/input/data",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type=data_config.s3_data_distribution_type,
        s3_compression_type=data_config.s3_compression_type,
    )

    result_output = ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=data_config.s3_output_path,
        output_name="analysis_result",
        s3_upload_mode="EndOfJob",
    )

    step_clarify = ProcessingStep(
        name="ClarifyProcessingStep",
        processor=clarify_processor,
        inputs= [data_input, config_input],
        outputs=[result_output],
        cache_config = cache_config,
        depends_on = [step_clarify_config.name]
    )

    # Evaluate model performance

    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{prefix}/{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source = step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination = f"s3://{bucket}/{prefix}/evaluation_report"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
        cache_config = cache_config
    )

    # Register Model in Model Registry

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterAbaloneModel",
        estimator=xgb_train,
        model_data=step_train.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Define a Pipeline

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            processed_data_path,
            input_data,
        ],
        steps=[step_process, step_train, step_create_model, step_clarify_config, step_clarify, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
        pipeline_experiment_config=PipelineExperimentConfig(
          ExecutionVariables.PIPELINE_NAME,
          ExecutionVariables.PIPELINE_EXECUTION_ID
        )
    )
    return pipeline