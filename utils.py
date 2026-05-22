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

NUM_PROC= max(os.cpu_count()-4,2)
BATCH_SIZE=500
# 8 threads are using 60~80GB of ram for me, beware
SAMPLES_TESTING=20



def get_curr_folder():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0])

CACHE_DIR=os.path.join(get_curr_folder(),'.cache/huggingface/datasets')

columns_relevant= ['text','audio','id']

def text_cleanup(row):
    text = row['text']
    text = text.replace("[]-", "")
    removable_chars = r"[?.,!\":;\'\t\*\n\“”’‘«»]"
    row['text'] = re.sub(removable_chars, "", text).lower()
    return row


def characters_valid(row):
    excluded_chars = r"(\d+|e-|…|é|ğ|ï|ⵒ|ⵠ|%|o|_|v|p|\(|\)|σ|\[|\])"
    return not bool(re.search(excluded_chars,row['text'].lower()))

def load_datasets_kab():
    data = {}

    print('Loading common_voice_22_0/kab dataset...')
    # TODO (maybe)
    # sort by text_len,text PRIOR to doing the .take(20000)
    # this makes the seeding useless, but prevents us 
    # from (unluckily) grabbing only short (and less 
    # representative) samples.


    # Don't want 500+ hours of data downloading
    #dataset = load_dataset("fsicoli/common_voice_22_0", "kab", trust_remote_code=True, cache_dir=CACHE_DIR, streaming=True)
    dataset = load_dataset("fsicoli/common_voice_22_0", "kab", trust_remote_code=True, streaming=True)
    # average sentence duration is 3.341
    # I want 30 hours, so 40k random samples should suffice
    samples = 40000
    subset_dataset = dataset['train'].shuffle(seed =SEED,buffer_size=samples).take(samples).remove_columns(["up_votes", "down_votes"])
    f = subset_dataset.features
    #print(f)
    dataset = Dataset.from_generator(lambda: (row for row in subset_dataset),features=f)
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    print("Casting dataset types...")
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    print("Cleaning up text...")
    dataset = dataset.filter(characters_valid,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.map(text_cleanup,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    print("Calculating per-transcrition text length...")
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])},num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.sort(['text_len','text'],reverse=True) #text added as a column for determinism order
    print("Localizing ids...")
    dataset = dataset.remove_columns("id")
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"},num_proc=NUM_PROC)
    data['common_voice_22_0'] = dataset
    print('Loaded common_voice_22_0/kab dataset.')

    #offset = len(dataset) #so we know which id to begin on next dataset

    return data



def load_datasets_zgh():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}
    shi = load_datasets_shi()['common_voice_22_0']
    len_shi = len(shi)
    shi = shi.select(range(SAMPLES_TESTING, len_shi))
    tzm = load_datasets_tzm()['common_voice_22_0']
    len_tzm = len(tzm)
    tzm = tzm.select(range(SAMPLES_TESTING, len_tzm))
    dataset = concatenate_datasets([tzm,shi])

    data['common_voice_22_0'] = dataset


    return data


def load_datasets_shi():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    ### TIFINAGH DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])
    dataset = dataset.rename_column("sentence","text")
    print(len(dataset))
    dataset = dataset.filter(lambda r: 'Tachelhit' in r['variant'])
    print(len(dataset))
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    print("Casting dataset types...")
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    print("Cleaning up text...")
    dataset = dataset.filter(characters_valid,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.map(text_cleanup,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    print("Calculating per-transcrition text length...")
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])},num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.sort(['text_len','text'],reverse=True) #text added as a column for determinism order
    print("Localizing ids...")
    dataset = dataset.remove_columns("id")
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"},num_proc=NUM_PROC)
    data['common_voice_22_0'] = dataset

    return data

def load_datasets_tzm():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])
    dataset = dataset.rename_column("sentence","text")
    print(len(dataset))
    dataset = dataset.filter(lambda r: 'Central Atlas Tamazight' in r['variant'])
    print(len(dataset))
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    print("Casting dataset types...")
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    print("Cleaning up text...")
    dataset = dataset.filter(characters_valid,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.map(text_cleanup,num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    print("Calculating per-transcrition text length...")
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])},num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    dataset = dataset.sort(['text_len','text'],reverse=True) #text added as a column for determinism order
    print("Localizing ids...")
    dataset = dataset.remove_columns("id")
    dataset = dataset.add_column("id", np.arange(len(dataset))) 
    dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"},num_proc=NUM_PROC)
    data['common_voice_22_0'] = dataset

    return data

autogen_folders = ['dicts','corpus','output','output_kab','output_nokab','plots','plots/output','plots/output_kab','plots/output_nokab']

def gen_project_folders():
    folders = [i for i in autogen_folders]
    for iso in ['kab','shi','zgh','tzm']:
        folders.append(f'corpus/{iso}')
        folders.append(f'output/{iso}')
        folders.append(f'output_kab/{iso}')
        folders.append(f'output_nokab/{iso}')

    for f in folders:
        cache_dir= os.path.join(get_curr_folder(),f)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clean_project_folders():
    for folder in autogen_folders:
        path = os.path.join(get_curr_folder(),folder)
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))


def prepare_project_structure():
    clean_project_folders() #in case they existed
    gen_project_folders()
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



def trim_trailing_silence(waveform, threshold=1e-2):
    # waveform: (channels, samples) or (samples,)
    if waveform.dim() > 1:
        energy = waveform.abs().max(dim=0).values
    else:
        energy = waveform.abs()

    # Find last index above threshold
    non_silent = torch.where(energy > threshold)[0]
    if len(non_silent) == 0:
        return waveform  # all silence

    start_idx = non_silent[0]
    end_idx = non_silent[-1]
    return waveform[..., start_idx:end_idx + 1]

