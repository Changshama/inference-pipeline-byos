from sagemaker.huggingface import HuggingFace
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.workflow.function_step import step

# @step(instance_type="ml.p3.2xlarge")
# def train(training_input_path):
#     huggingface_estimator = HuggingFace(entry_point='../code/unsupervised.py',
#                             source_dir='../code',
#                             instance_type=instance_type, # GPU supported by Hugging Face
#                             instance_count=1,
#                             role=role,
#                             transformers_version='4.6',
#                             pytorch_version='1.7',
#                             py_version='py36',
#                             hyperparameters = {'epochs': 1,
#                              'train_batch_size': 8,
#                              'model_name':'bert-base-uncased'
#                             })
    
#     training_input_file = training_input_path + "train.csv"
    
#     huggingface_estimator.fit({'train': training_input_file})
    
#     sentence_transformer = HuggingFaceModel(model_data = huggingface_estimator.model_data, 
#                             role = role, 
#                             source_dir = '../code',
#                             entry_point = 'inference.py', 
#                             transformers_version='4.6',
#                             pytorch_version='1.7',
#                             py_version='py36',)
    
#     return sentence_transformer, training_input_path



@step(instance_type="ml.g4dn.xlarge")
def transform(sentence_transformer_model, input_path, output_path):
    
    #create model
    sentence_transformer = HuggingFaceModel(model_data = sentence_transformer_model.model_data, 
                        role = role, 
                        source_dir = '../code',
                        entry_point = 'inference.py', 
                        transformers_version='4.6',
                        pytorch_version='1.7',
                        py_version='py36',)
    
    batch_job = sentence_transformer.transformer(
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        strategy='SingleRecord'
    )
    
    input_file = input_path + 'incident-batch.jsonl'
    
    batch_job.transform(
        data=input_file,
        content_type='application/json',    
        split_type='Line'
    )
    
    transformed_file = output_path + "/incident-batch.jsonl.out"
    
    return transformed_file
    