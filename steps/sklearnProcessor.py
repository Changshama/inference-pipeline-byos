from sagemaker.sklearn.estimator import SKLearn

def sklearn_train(train_input_path, transformed_file=''):
    FRAMEWORK_VERSION = "1.2-1"
    script_path = "../feature-processing.py"

    sklearn_preprocessor = SKLearn(
        entry_point=script_path,
        role=role,
        framework_version=FRAMEWORK_VERSION,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sess,
    )
    sklearn_preprocessor.fit({"train": train_input_path})
    return sklearn_preprocessor