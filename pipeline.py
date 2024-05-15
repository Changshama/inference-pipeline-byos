import os
import sagemaker
from sagemaker.utils import unique_name_from_base
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import (ParameterString,)
from sagemaker.workflow.steps import TrainingStep
from steps.sentenceTransformer import (transform,)
from steps.sklearnProcessor import sklearn_train
from steps.register import register
from sagemaker.huggingface import HuggingFace
from sagemaker.huggingface.model import HuggingFaceModel

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    # Define input data and default bucket
    bucket=sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()
    training_input_path = "s3://sagemaker-us-east-1-827930657850/sentencetransformer/input/"
    # Define the name of the model package group to host all the model versions in pending approval state
    model_pkg_group_name = "Accident Categorization"
    model_approval_status_param = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    # Define the steps of the SageMaker Pipeline, i.e. preprocess, train, evaluate, register
    
    # new pipeline SDK
    # sentence_transformer = step(train, name="sentence-transformer-model")(training_input_path)
    
    # old pipeline SDK
    estimator = HuggingFace(entry_point='train.py',
                            source_dir='./code',
                            instance_type="ml.p3.2xlarge",
                            instance_count=1,
                            role=role,
                            transformers_version='4.6',
                            pytorch_version='1.7',
                            py_version='py36',
                            hyperparameters = {'epochs': 1,
                             'train_batch_size': 8,
                             'model_name':'bert-base-uncased'
                            })
    
    training_input_file = training_input_path + "train.csv"
    estimator.fit(training_input_file)
    
    step_train = TrainingStep(
        name="sentence-transformer-train",
        estimator=estimator
    )    
    
    # create model (TO-DO: add as a step)     
    sentence_transformer = HuggingFaceModel(model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts, 
                        role = role, 
                        source_dir = './code',
                        entry_point = 'inference.py', 
                        transformers_version='4.6',
                        pytorch_version='1.7',
                        py_version='py36',)

    # batch transform
    transformed_file = step(transform, name="batch-transform-sentence-transformer")(sentence_transformer, training_input_path, training_input_path)

    # sklearn processor
    sklearn_model = step(sklearn_train, name="sklearn-model")(training_input_path, transformed_file)

    # linear learner ()
    
    # evaluation_result = step(evaluate, name="Model_Evaluation")(model=model, test_df=data[2])

    model_register = step(register, name="Model_Registration")(
        sentence_transformer=sentence_transformer,
        sklearn_processor=sklearn_model,
        model_approval_status=model_approval_status_param,
        model_package_group_name=model_pkg_group_name,
        bucket=bucket,
    )

    # Create the SageMaker Pipeline including name, parameters, and the output of the last step
    pipeline_building = Pipeline(
        name="accident-category-sm-pipeline-new-sdk",
        parameters=[model_approval_status_param],
        sagemaker_session=sagemaker_session,
        steps=[model_register],
    )

    # Deploy and start a SageMaker Pipeline execution
    pipeline_building.upsert(role_arn=sagemaker.get_execution_role())
    pipeline_building.start()