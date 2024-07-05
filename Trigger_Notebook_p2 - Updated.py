# Databricks notebook source
# MAGIC %md
# MAGIC Method 1

# COMMAND ----------

!pip install adlfs
!set LLAMA_CUBLAS=1
!set CMAKE_ARGS=-DLLAMA_CUBLAS=on
!set FORCE_CMAKE=1
!pip install llama-cpp-python==0.1.78 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
!pip install langchain_community


# COMMAND ----------

import os
import tempfile
from adlfs.spec import AzureBlobFileSystem
from databricks.sdk.runtime import *
from pyspark.sql import SparkSession

class DBUtilConnectionCreator:
    """
    Helper class for creating a connection to Azure Blob Storage using DBUtils.
    """

    def __init__(self, dbutils):
        """
        Initialize the DBUtilConnectionCreator.

        Args:
            dbutils: DBUtils instance for accessing secrets and configurations.
        """
        self.storage_key = "pplz-key-adanexpr"
        self.storage_secret = "storage-account-adanexpr"
        self.dbutils = dbutils

    def get_abfs_client(self):
        """
        Create an Azure Blob Storage client for working with Azure Blob Storage

        Returns:
            AzureBlobFileSystem: The Azure Blob Storage client.
        """
        try:
            spark = SparkSession.builder.getOrCreate()

            ppl_tenant_id = self.dbutils.secrets.get(
                scope=self.storage_key,
                key="tenant-id-adanexpr"
            )
            adanexpr_storage_acct = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr", key="storage-account-adanexpr"
            )
            adanexpr_ds_dbricks_id = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr",
                key="Azure-SP-ADANEXPR-DS-DBricks-ID"
            )
            adanexpr_ds_dbricks_pwd = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr",
                key="Azure-SP-ADANEXPR-DS-DBricks-PWD"
            )

            spark.conf.set("fs.azure.enable.check.access", "false")

            acct = adanexpr_storage_acct
            config_key = f"fs.azure.account.auth.type.{acct}.dfs.core.windows.net"
            config_value = "OAuth"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.hns.enabled.{acct}.dfs.core.windows.net"
            config_value = "true"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.oauth.provider.type.{acct}.dfs.core.windows.net"
            config_value = "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.oauth2.client.id.{acct}.dfs.core.windows.net"
            config_value = adanexpr_ds_dbricks_id
            spark.conf.set(config_key, config_value)

            config_key_secret = f"fs.azure.account.oauth2.client.secret.{acct}.dfs.core.windows.net"
            config_value_secret = adanexpr_ds_dbricks_pwd
            spark.conf.set(config_key_secret, config_value_secret)

            endpoint = f"https://login.microsoftonline.com/{ppl_tenant_id}/oauth2/token"
            cke = f"fs.azure.account.oauth2.client.endpoint.{acct}.dfs.core.windows.net"
            spark.conf.set(cke, endpoint)

            return AzureBlobFileSystem(
                account_name=adanexpr_storage_acct,
                tenant_id=ppl_tenant_id,
                client_id=adanexpr_ds_dbricks_id,
                client_secret=adanexpr_ds_dbricks_pwd,
            )
        except Exception as e:
            print(f"Error in get_abfs_client: {str(e)}")
            return None

blob_directory = "datascience/data/ds/projects/2023_CS_PA_NL/llama"

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()

def download_blob_directory(abfsClient, blob_directory, local_directory):
    try:
        # List all blobs in the directory and download them
        blob_list = abfsClient.ls(blob_directory, detail=True)
        
        for blob in blob_list:
            blob_name = os.path.basename(blob['name'])  # Extract just the file name
            blob_path = blob['name']
            local_file_path = os.path.join(local_directory, blob_name)

            if blob['type'] == 'DIRECTORY':
                # Recursively create directories
                os.makedirs(local_file_path, exist_ok=True)
                download_blob_directory(abfsClient, blob_path, local_file_path)
            elif blob['type'] == 'FILE':
                with abfsClient.open(blob_path, "rb") as remote_file:
                    with open(local_file_path, "wb") as local_file:
                        local_file.write(remote_file.read())

        print(f"All blobs in '{blob_directory}' downloaded to '{local_directory}'")

    except Exception as e:
        print(f"Error downloading blob directory '{blob_directory}': {str(e)}")

# Create a temporary directory to store downloaded files
with tempfile.TemporaryDirectory() as temp_dir:
    download_blob_directory(abfsClient, blob_directory, temp_dir)

llm = LlamaCpp(
    model_path=temp_dir,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    #verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096
)


# COMMAND ----------

# MAGIC %md
# MAGIC Method 2 - dbfs

# COMMAND ----------

import os

os.environ['HF_HOME'] = '/FileStore/tables/cs_pa_nlp_gpu/data/output/llama'

hf_home_path = os.getenv('HF_HOME')
print("HF_HOME Path:", hf_home_path)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Distributed Inference

# COMMAND ----------

!set LLAMA_CUBLAS=1
!set CMAKE_ARGS=-DLLAMA_CUBLAS=on
!set FORCE_CMAKE=1
!pip install llama-cpp-python==0.1.78 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
!pip install langchain_community
!pip install ray


# COMMAND ----------

from langchain.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from transformers import pipeline, LlamaTokenizerFast, AutoTokenizer, AutoModelForTokenClassification
import time
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
import os
import nltk
from nltk.tokenize import word_tokenize
import ray
import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import regex as re
from nltk.tokenize import sent_tokenize
import ast
import time
import json
import os
nltk.download('punkt')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, 
                                               chunk_overlap=50, 
                                               length_function=len)

model_name_or_path = "TheBloke/Llama-2-7B-chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q5_0.bin"
model_path = hf_hub_download(repo_id=model_name_or_path,
                             filename=model_basename)

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    #n_threads=12,
    #verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096
)

def summary(x,
            text_splitter):
    prompt = """
[INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
<<SYS>> {} [/INST]
"""
    chunks = text_splitter.split_text(x)
    chunk_summaries = []

    for chunk in chunks:
        summary = llm.invoke(prompt.format(x))
        chunk_summaries.append(summary)

    combined_summary = "\n".join(chunk_summaries)

    return combined_summary

def actions_taken(x):
    prompt = f"""
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.
<<SYS>> {x} [/INST]
"""
    return llm.invoke(prompt)

def merger(anonymized_list):
    transcription = ""

    for trans_dict in anonymized_list:
        transcription+=trans_dict['text']

    return transcription 

def word_count(transcription):
    return len(word_tokenize(transcription))

today = datetime.now().date()
today_str = today.strftime('%Y_%m_%d')

# Define the base directory
base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"

# Construct the file path for reading
file_path = os.path.join(base_dir, f"gpu_transcriptions_redacted_2024_06_19.parquet")

# Read the Parquet file into a DataFrame
try:
    front_end = pd.read_parquet(file_path)
    print("Number of rows: "+ str(len(front_end)))
    print(f"Successfully loaded DataFrame from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")
# Applying the function to the 'transcription' column
start = time.time()
front_end = pd.read_parquet(file_path)

front_end['transcription'] = front_end['anonymized'].apply(lambda x: merger(x))
front_end['word_count'] = front_end['transcription'].apply(lambda x: word_count(x))

# Initialize Ray
ray.init()

# Function definitions for processing
def merger(anonymized_list):
    transcription = ""
    for trans_dict in anonymized_list:
        transcription += trans_dict['text']
    return transcription

def word_count(transcription):
    return len(word_tokenize(transcription))

def summary(x, 
            text_splitter, 
            llm):
    prompt = """
[INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
<<SYS>> {} [/INST]
"""
    chunks = text_splitter.split_text(x)
    chunk_summaries = []
    for chunk in chunks:
        summary = llm.invoke(prompt.format(x))
        chunk_summaries.append(summary)
    combined_summary = "\n".join(chunk_summaries)
    return combined_summary

def actions_taken(x, llm):
    prompt = f"""
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.
<<SYS>> {x} [/INST]
"""
    return llm.invoke(prompt)

# Define file paths
base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"
save_path = os.path.join(base_dir, "gpu_transcriptions_redacted_summary_2024_06_19.parquet")

# Function to process a single row
@ray.remote(num_gpus=2)  # Adjust the number of GPUs as needed
def process_row(row, 
                text_splitter, 
                llm):
    try:
        transcription = merger(row['anonymized'])
        word_count_val = word_count(transcription)
        if word_count_val > 15:
            actions_taken_val = actions_taken(transcription, llm)
            summary_val = summary(transcription, text_splitter, llm)
        else:
            actions_taken_val = "no actions taken generated"
            summary_val = "no summary generated"
        return {
            'transcription': transcription,
            'word_count': word_count_val,
            'actions_taken': actions_taken_val,
            'summary': summary_val
        }
    except Exception as e:
        return {
            'transcription': '',
            'word_count': 0,
            'actions_taken': f"Error: {str(e)}",
            'summary': f"Error: {str(e)}"
        }

# Process data using Ray
start = time.time()
processed_results = ray.get([process_row.remote(row, 
                                                text_splitter, 
                                                llm) for index, row in front_end.iterrows()])

# Convert results to Pandas DataFrame
processed_df = pd.DataFrame(processed_results)

# Save processed DataFrame to Parquet format
try:
    processed_df.to_parquet(save_path)
    print(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {save_path}. Exception: {e}")

# Shutdown Ray
ray.shutdown()

print("Job completed in " + str(time.time() - start))


# COMMAND ----------

# MAGIC %md
# MAGIC Mlflow Logging

# COMMAND ----------

import mlflow
import pandas as pd
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def __init__(self, llama_cpp_model):
        self.llama_cpp_model = llama_cpp_model

    def predict(self, context, model_input):
        output = self.llama_cpp_model.invoke(model_input['text'].iloc[0])
        return output

def _load_pyfunc(llm):
    return LlamaCppModel(llm)

model_name_or_path = "TheBloke/Llama-2-7B-chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q5_0.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=4096
)

model_instance = _load_pyfunc(llm)

# Define an input pandas DataFrame for inference example
model_input = pd.DataFrame(["Hi how are you doing today?"])
model_input.columns = ['text']

# Log the model and its artifacts with MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="llama_cpp_model",
        python_model=model_instance,
        signature=infer_signature(model_input),
        input_example=pd.DataFrame(model_input, columns=['text'])
    )

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/llama_cpp_model"
    print(f"Logged model URI: {model_uri}")


# COMMAND ----------

run_id = "9b01411d237a4a28ae7d1c82c2084912"

# Construct the model URI based on the run ID and artifact path
model_uri = f"runs:/{run_id}/llama_cpp_model"

# Load the model from MLflow
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Example input for inference
model_input = pd.DataFrame(["Hi how are you doing today?"], columns=['text'])

# Perform inference using the loaded model
output = loaded_model.predict(model_input)

print("Inference Output:")
print(output)


# COMMAND ----------

!pip install ray

# COMMAND ----------

from langchain.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from transformers import pipeline, LlamaTokenizerFast, AutoTokenizer, AutoModelForTokenClassification
import time
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
import os
import nltk
from nltk.tokenize import word_tokenize
import ray
import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import regex as re
from nltk.tokenize import sent_tokenize
import ast
import time
import json
import os
nltk.download('punkt')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, 
                                               chunk_overlap=50, 
                                               length_function=len)

def summary(x,
            text_splitter):
    prompt = """
[INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
<<SYS>> {} [/INST]
"""
    chunks = text_splitter.split_text(x)
    chunk_summaries = []

    for chunk in chunks:
        summary = llm.invoke(prompt.format(x))
        chunk_summaries.append(summary)

    combined_summary = "\n".join(chunk_summaries)

    return combined_summary

def actions_taken(x):
    prompt = f"""
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.
<<SYS>> {x} [/INST]
"""
    return llm.invoke(prompt)

def merger(anonymized_list):
    transcription = ""

    for trans_dict in anonymized_list:
        transcription+=trans_dict['text']

    return transcription 

def word_count(transcription):
    return len(word_tokenize(transcription))

today = datetime.now().date()
today_str = today.strftime('%Y_%m_%d')

# Define the base directory
base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"

# Construct the file path for reading
file_path = os.path.join(base_dir, f"gpu_transcriptions_redacted_2024_06_19.parquet")

# Read the Parquet file into a DataFrame
try:
    front_end = pd.read_parquet(file_path)
    print("Number of rows: "+ str(len(front_end)))
    print(f"Successfully loaded DataFrame from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")
# Applying the function to the 'transcription' column
start = time.time()
front_end = pd.read_parquet(file_path)

front_end['transcription'] = front_end['anonymized'].apply(lambda x: merger(x))
front_end['word_count'] = front_end['transcription'].apply(lambda x: word_count(x))

# Initialize Ray
ray.init()

# Function definitions for processing
def merger(anonymized_list):
    transcription = ""
    for trans_dict in anonymized_list:
        transcription += trans_dict['text']
    return transcription

def word_count(transcription):
    return len(word_tokenize(transcription))

def summary(x, 
            text_splitter, 
            loaded_model):
    prompt = """
[INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
<<SYS>> {} [/INST]
"""
    chunks = text_splitter.split_text(x)
    chunk_summaries = []
    for chunk in chunks:
        summary = loaded_model.invoke(prompt.format(x))
        chunk_summaries.append(summary)
    combined_summary = "\n".join(chunk_summaries)
    return combined_summary

def actions_taken(x, loaded_model):
    prompt = f"""
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.
<<SYS>> {x} [/INST]
"""
    return loaded_model.invoke(prompt)

# Define file paths
base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"
save_path = os.path.join(base_dir, "gpu_transcriptions_redacted_summary_2024_06_19.parquet")

# Function to process a single row
@ray.remote(num_gpus=2)  # Adjust the number of GPUs as needed
def process_row(row, 
                text_splitter, 
                loaded_model):
    try:
        transcription = merger(row['anonymized'])
        word_count_val = word_count(transcription)
        if word_count_val > 15:
            transcription = pd.DataFrame([transcription], columns=['text'])
            actions_taken_val = actions_taken(transcription, loaded_model)
            summary_val = summary(transcription, text_splitter, loaded_model)
        else:
            actions_taken_val = "no actions taken generated"
            summary_val = "no summary generated"
        return {
            'transcription': transcription,
            'word_count': word_count_val,
            'actions_taken': actions_taken_val,
            'summary': summary_val
        }
    except Exception as e:
        return {
            'transcription': '',
            'word_count': 0,
            'actions_taken': f"Error: {str(e)}",
            'summary': f"Error: {str(e)}"
        }

# Process data using Ray
start = time.time()
processed_results = ray.get([process_row.remote(row, 
                                                text_splitter, 
                                                loaded_model) for index, row in front_end.iterrows()])

# Convert results to Pandas DataFrame
processed_df = pd.DataFrame(processed_results)

# Save processed DataFrame to Parquet format
try:
    processed_df.to_parquet(save_path)
    print(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {save_path}. Exception: {e}")

# Shutdown Ray
ray.shutdown()

print("Job completed in " + str(time.time() - start))


# COMMAND ----------

# MAGIC %md
# MAGIC Topic Modelling With Ray Optimization 

# COMMAND ----------

class TopicModelling:
    def __init__(self, device):
        self.device = device
        torch.cuda.set_device(self.device)  # Explicitly setting device
        self.setup()

    def setup(self):
        self.classifier = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=self.device)
        
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5',
                                          device=self.device).to(self.device)
        
        with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topic_og.json", "r") as f:
            json_data = f.read()

            data = json.loads(json_data)
            
        self.ci = data
        self.topic_embeddings = self.model.encode(
            (list(self.ci.values())))
        
    def probability_assignment(self, summary, topic_list):
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            return self.classifier(summary, topic_list)
        except Exception as e:
            print(f"Error in probability_assignment: {str(e)}")
            return "ERROR"

    def apply_probability_assignment(self, topic_list, summary):
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            else:
                probabilities = self.probability_assignment(
                    summary, topic_list)
                return probabilities
        except Exception as e:
            print(f"Error in apply_probability_assignment: {str(e)}")
            return "ERROR"

    def parse_topic_with_probabilities(self, x):
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x):
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x):
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'
        
    def predict(self, summary):
        try:
            index = 0
            threshold = 0.4
            top_2_topics_per_cluster = pd.DataFrame(
                columns=[
                    'Sentence',
                    'Topic',
                    'Position',
                    'cos_sim',
                    'Chunking Strategy'])
            
            chunks = list(summary.split('.'))
            chunks = [sentence for sentence in summary.split(
                '.') if len(sentence.split()) >= 5]
            
            sentence_embeddings = self.model.encode(chunks)
            
            for i, sentence_embedding in enumerate(sentence_embeddings):
                for topic_num, topic_embedding in enumerate(self.topic_embeddings):
                    dot_product = np.dot(sentence_embedding, 
                                         topic_embedding)
                    norm_A = np.linalg.norm(sentence_embedding)
                    norm_B = np.linalg.norm(topic_embedding)
                    cosine_similarity = dot_product / (norm_A * norm_B)
                    if cosine_similarity > threshold:
                        top_2_topics_per_cluster.at[index,
                                                    'Sentence'] = str(
                                                        chunks[i])
                        top_2_topics_per_cluster.at[index,
                                                    'Topic'] = str(
                                                        list(
                                                            self.ci.keys())[
                                                                topic_num])
                        top_2_topics_per_cluster.at[index,
                                                    'Position'] = i
                        top_2_topics_per_cluster.at[index,
                                                    'cos_sim'] = float(
                                                        cosine_similarity)
                        top_2_topics_per_cluster.at[index,
                                                    'Chunking Strategy'] = str(
                                                        chunks)
                        index += 1

            if len(top_2_topics_per_cluster) == 0:
                print("Empty top topics df")

            position_wise = top_2_topics_per_cluster.sort_values(by=[
                'Position'], ascending=True)
            if len(position_wise) >= 10:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'].iloc[0:10])
            elif len(position_wise) > 0:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'])
            else:
                top_topics = []

        except Exception as e:
            print(f"Error in topic_modeller: {str(e)}")
            return []

        topic_dict = self.parse_topic_with_probabilities(
            self.apply_probability_assignment(
                topic_list = top_topics, 
                summary = summary))
        primary = self.get_primary_topic(x = topic_dict)
        secondary = self.get_secondary_topic(x =topic_dict)

        return [primary, secondary]

def safe_literal_eval(s):
    if isinstance(s, list):
        return s

    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except ValueError as e:
            print(f"Error parsing string with ast.literal_eval: {e}\nAttempting to parse with json.loads.")
            try:
                return json.loads(s.replace("'", '"'))
            except json.JSONDecodeError as je:
                print(f"Error parsing string with json: {je}\nInput data: {s}")

    print(f"Unsupported data type or malformed input: {type(s)}\nInput data: {s}")
    return None

@ray.remote(num_gpus=2)
def process_texts_on_gpu(device_id, texts):
    tm = TopicModelling(device=f'cuda:{device_id}')
    results = []
    for text in texts:
        start_time = time.time()
        topics = tm.predict(text)
        end_time = time.time()
        results.append({'summary': text,
                        'topics': topics, 
                        'time_taken': end_time - start_time})
    return results

def main():
    ray.init(num_gpus=2)

    today = datetime.now().date()
    today_str = today.strftime('%Y_%m_%d')
    base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"
    file_path = os.path.join(base_dir, f"gpu_transcriptions_redacted_summary_2024_06_19.parquet")

    try:
        transcriptions = pd.read_parquet(file_path)
        print(f"Successfully loaded DataFrame from: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")
        return

    file_names = transcriptions['summary'].tolist()
    start = time.time()
    print("Process Started")

    batch_size = len(file_names) // 2  # Splitting into two batches for two GPUs
    batches = [file_names[i * batch_size:(i + 1) * batch_size] for i in range(2)]

    results = []

    futures = [process_texts_on_gpu.remote(i, batches[i]) for i in range(2)]
    results.extend(ray.get(futures))

    for result in results:
        print(result)

    save_path = os.path.join(base_dir, f"gpu_transcriptions_redacted_topics_2024_06_19.parquet")
    try:
        pd.DataFrame(results).rename(columns=str).to_parquet(save_path)
        print(f"DataFrame successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error: Failed to save DataFrame. Exception: {e}")
    
    ray.shutdown()
    print("Ray shutdown complete.")

if __name__ == "__main__":
    main()

