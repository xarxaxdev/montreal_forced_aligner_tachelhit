from datasets import Features,Audio,Value,Dataset,load_dataset,concatenate_datasets #using huggingface's API
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch
import numpy as np
import random
from librosa import resample
from pathlib import Path
import os,sys
import urllib
from tqdm import tqdm
import unicodedata
import re 

THRESHOLD_MIN_SECONDS = 0.25
SEED = 42
DICTS = {}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dataset_alias={
    'common_voice_22_0':'cv22',
    'kabyle_asr':'tfnlab',
    'moroccan_amazigh_asr':'tfnlab',
}


def get_curr_folder():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0])

CACHE_DIR=os.path.join(get_curr_folder(),'.cache/huggingface/datasets')

regex = re.compile(r'[^\x00-\x7F]+')
def cleanup_text(row):
    return {"text":regex.sub('',unicodedata.normalize("NFC",row['text']))}

columns_relevant= ['text','audio']

def load_datasets_kabyle():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data = {}
    dataset = load_dataset("TutlaytAI/kabyle_asr",cache_dir=CACHE_DIR,streaming=True)
    dataset = concatenate_datasets([dataset['train'],dataset['test']])
    dataset = dataset.rename_column("Text","text")
    dataset = dataset.map(cleanup_text)
    #dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"kabyle_asr"})
    data['kabyle_asr'] = dataset

    # Don't want 500+ hours of data downloading
    dataset = load_dataset("fsicoli/common_voice_22_0", "kab", trust_remote_code=True, cache_dir=CACHE_DIR, streaming=True)
    # average sentence duration is 3.341
    # I want 15 hours, so 20k random samples should suffice
    dataset = dataset['train']
    dataset = dataset.shuffle(seed =SEED,buffer_size=100).take(20000)
    #dataset = dataset.shuffle(seed =SEED).take(1)
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(cleanup_text)
    #dataset = dataset.select_columns(columns_relevant)
    #print(dataset.take(1))
    #assert(False)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset

    return data



def load_datasets_zgh():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    ### LATINSCRIPT DATASET
    # https://aclanthology.org/2025.icnlsp-1.37.pdf#:~:text=We%20have%20also%20applied%20and%20validated%20the,of%20the%20utilized%20dataset%20for%20benchmarking%20and
    dataset = load_dataset("TutlaytAI/moroccan_amazigh_asr",cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['test']])
    # TODO: Why does the testing split not always have a sampling rate? look into whether this is fixable
    #dataset = dataset['train']
    #dataset = dataset.filter(lambda x: x["audio"]["sampling_rate"] is not None)
 
    print(dataset.features)
    dataset = dataset.rename_column("transcription","text")
    dataset = dataset.map(cleanup_text)
    #dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"moroccan_amazigh_asr"})
    data['moroccan_amazigh_asr'] = dataset

    ### TIFINAGH DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])

    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(cleanup_text)
    dataset = dataset.select_columns(columns_relevant)
    # Use the Audio's sampling rate rather than the
    # Dataset's schema for future concatenate_datasets()
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
    })
    dataset = dataset.cast(new_features)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset

    return data


def gen_project_folders():
    for folder in ['dicts','corpus','corpus/kab','corpus/shi','output','output/corpus_kab','output/corpus_shi']:
        cache_dir= os.path.join(get_curr_folder(),folder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clean_project_folders():
    for folder in ['dicts']:
        path = os.path.join(get_curr_folder(),folder)
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))


def prepare_project_structure():
    gen_project_folders()
    clean_project_folders() #in case they existed
    #download_dicts()


def load_dicts():
    path = os.path.join(get_curr_folder(),'dicts')
    for f in os.listdir(path):
        d_name = f.split('.')[0]
        DICTS[d_name] = {}
        with open(os.path.join(path,f)) as d:
            for line in  d:
                [left,right] = line.split('\t')
                DICTS[d_name][left]=right
    return DICTS

