from datasets import Dataset,load_dataset,Audio,concatenate_datasets #using huggingface's API
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
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_curr_folder():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0])

CACHE_DIR=os.path.join(get_curr_folder(),'.cache/huggingface/datasets')

    


def reshape_audio(row):
    audio = row["audio"]
    waveform = np.asarray(audio["array"], dtype=np.float32)
    waveform = resample(waveform,orig_sr=int(audio["sampling_rate"]),target_sr=16000)
    waveform = np.asarray(audio["array"], dtype=np.float32)
    text = row['text']
    regex = re.compile(r'[^\x00-\x7F]+')

    return {
        "waveform": waveform,
        "sr": 16000,
        "filename": str(audio["path"]),
        "origin": "tamasightASRDatasetV2",
        "text":regex.sub('',unicodedata.normalize("NFC",text)),
    }


def load_datasets_kabyle():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data = {}
    dataset = load_dataset("TutlaytAI/kabyle_asr",cache_dir=CACHE_DIR,streaming=True)
    dataset = concatenate_datasets([dataset['train'],dataset['test']])
    dataset = dataset.rename_column("Text","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
    data['kabyle_asr'] = dataset

    dataset = load_dataset("fsicoli/common_voice_22_0", "kab", trust_remote_code=True, cache_dir=CACHE_DIR, streaming=True)
    # average sentence duration is 3.341
    # I want 15 hours, so 20k random samples should suffice
    dataset = dataset['train']
    dataset = dataset.shuffle(seed =SEED).take(20000)
    #dataset = dataset.shuffle(seed =SEED).take(1)
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
    #print(dataset.take(1))
    #assert(False)
    data['common_voice_22_0'] = dataset
    #data['common_voice_22_0'] = Dataset.from_generator(lambda:dataset)

    return data



def load_datasets():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)


    #todo, normalize these

    ### LATINSCRIPT DATASET
    dataset = load_dataset("TutlaytAI/moroccan_amazigh_asr",cache_dir=CACHE_DIR)
    dataset = dataset['train']
    dataset = dataset.rename_column("transcription","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
    data['moroccan_amazigh_asr'] = dataset

    ### AMAZIGHSCRIPT DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = dataset['train']
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
    data['common_voice_22_0'] = dataset

    return data


def gen_project_folders():
    for folder in ['dicts','corpus','corpus_kabyl','output','output/corpus_aligned']:
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

