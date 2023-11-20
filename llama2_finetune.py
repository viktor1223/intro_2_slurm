import getpass

import torch
import logging
import os
import yaml
from ludwig.api import LudwigModel
from datasets import load_dataset

import numpy as np; np.random.seed(42)
import pandas as pd
from sklearn.metrics import f1_score


# Initialize lists to hold the data
def data_get(data):
  instruction_list = []
  input_list = []
  output_list = []

  for i, row in enumerate(data):
    #print(row)
    instruction_list.append(f"""Below is an instruction that describes a task, paired with an input
                    that provides further context. Write a response that appropriately
                    completes the request.  # noqa: E501
                    You are a helpful, respectful and honest assistant. Always answer
                    as helpfully as possible, while being safe.  Your answers should
                    not include any harmful, unethical, racist, sexist, toxic,
                    dangerous, or illegal content. Please ensure that your responses
                    are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent,
                    explain why instead of answering something not correct. If you don't
                    know the answer to a question, please don't share false information.
                Use the following context to answer the question. Think step by step and explain your reasoning. {row['context']} """)
    input_list.append(f"Question: {row['question']}")
    output_list.append(row['answers'])
    # Create the DataFrame from the lists
  data_df = pd.DataFrame({"instruction": instruction_list, "input": input_list, "output": output_list})
  return data_df


assert os.environ["HUGGING_FACE_HUB_TOKEN"]
yaml_file = "llama_qa_ir.yaml"

print("Loading Data...")
dataset = load_dataset("squad_v2")
print("Partitioning Data...")
train = dataset['train']
val = dataset['validation']

train_data = data_get(train)
print("Training Dataframe")
print(train_data.head())

val_data = data_get(val)
print("Validation Dataframe")
print(val_data.head())

#Load Model
qlora_fine_tuning_config = yaml.safe_load(yaml_file)
model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)



#Train Model
results = model.train(dataset=train_data)
print("Model Trained!")


print("Getting F1 scores...")

gTruth_ans = val_data["output"]

print("OUTPUT FROM PRED ")

pred =  model.predict(val_df)
if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
    f1_metric = f1_score(predictions=pred, references=gTruth_ans)
    print(f"Llama2 SQuADv2 F1 Score {f1_metric}")
