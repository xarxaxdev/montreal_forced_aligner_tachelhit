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

columns_relevant= ['text','audio','id']

def load_datasets_kab():
    data = {}

    print('Loading common_voice_22_0/kab dataset...')
    # TODO (maybe)
    # sort by text_len,text PRIOR to doing the .take(20000)
    # this makes the seeding useless, but prevents us 
    # from (unluckily) grabbing only short (and less 
    # representative) samples.


    # Don't want 500+ hours of data downloading
    dataset = load_dataset("fsicoli/common_voice_22_0", "kab", trust_remote_code=True, cache_dir=CACHE_DIR, streaming=True)
    # average sentence duration is 3.341
    # I want 15 hours, so 20k random samples should suffice
    subset_dataset = dataset['train'].shuffle(seed =SEED,buffer_size=len(dataset['train'])).take(20000).remove_columns(["up_votes", "down_votes"])
    f = subset_dataset.features
    #print(f)
    dataset = Dataset.from_generator(lambda: (row for row in subset_dataset),features=f)
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(cleanup_text)
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True) #text added as a column for determinism order
    dataset = dataset.map(lambda x, i: {"id": i}, with_indices=True)
    offset = len(dataset) #so we know which id to begin on next dataset
    dataset = dataset.select_columns(columns_relevant)
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset
    print('Loaded common_voice_22_0/kab dataset...')

 
    print('Loading kabyle_asr dataset...')
    dataset = load_dataset("TutlaytAI/kabyle_asr",cache_dir=CACHE_DIR)
    print(type(dataset['train']), type(dataset['test']))

    dataset = concatenate_datasets([dataset['train'],dataset['test']])
    dataset = dataset.rename_column("Text","text")
    dataset = dataset.map(cleanup_text)
    # Adding an id that depends on text length 
    #dataset = dataset.cast_column("audio", Audio(decode=False)) # We dont want to decode 10k+ audios while non-streaming
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True) #text added as a column for determinism order
    dataset = dataset.map(lambda x, i: {"id": i + offset}, with_indices=True)
    #dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"kabyle_asr"})
    data['kabyle_asr'] = dataset
    print('Loaded kabyle_asr dataset...')

   return data



def load_datasets_zgh():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    ### TIFINAGH DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])
    dataset = dataset.rename_column("sentence","text")
    # Deterministic ID assignment
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True)
    dataset = dataset.map(lambda x, i: {"id": i}, with_indices=True)
    offset = len(dataset) #so we know which id to begin on next dataset
    dataset = dataset.select_columns(columns_relevant)
    # Use the Audio's sampling rate rather than the
    # Dataset's schema for future concatenate_datasets()
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset


    ### LATINSCRIPT DATASET
    # https://aclanthology.org/2025.icnlsp-1.37.pdf#:~:text=We%20have%20also%20applied%20and%20validated%20the,of%20the%20utilized%20dataset%20for%20benchmarking%20and
    dataset = load_dataset("TutlaytAI/moroccan_amazigh_asr",cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['test']])
    dataset = dataset.rename_column("transcription","text")
    dataset = dataset.map(cleanup_text)
    # Deterministic ID assignment
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True)
    dataset = dataset.map(lambda x, i: {"id": i + offset}, with_indices=True)
    #dataset = dataset.select_columns(columns_relevant)
    dataset = dataset.map(lambda x: {"origin":"moroccan_amazigh_asr"})
    data['moroccan_amazigh_asr'] = dataset

    return data


# TODO:
# have to filter according to https://huggingface.co/datasets/fsicoli/common_voice_22_0/raw/main/transcript/zgh/validated.tsv

def load_datasets_shi():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    ### TIFINAGH DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])
    dataset = dataset.rename_column("sentence","text")
    # Deterministic ID assignment matching zgh
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True)
    dataset = dataset.map(lambda x, i: {"id": i }, with_indices=True)
    print(len(dataset))
    dataset = dataset.filter(lambda r: 'Tachelhit' in r['variant'])
    print(len(dataset))
    dataset = dataset.select_columns(columns_relevant)
    # Use the Audio's sampling rate rather than the
    # Dataset's schema for future concatenate_datasets()
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("int64"),
    })
    dataset = dataset.cast(new_features)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset


    return data

def load_datasets_tzm():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    data ={}

    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=CACHE_DIR)
    dataset = concatenate_datasets([dataset['train'],dataset['validation'],dataset['test']])

    dataset = dataset.rename_column("sentence","text")
    # Deterministic ID assignment matching zgh
    dataset = dataset.map(lambda x : {"text_len":len(x['text'])})
    dataset =  dataset.sort(['text_len','text'],reverse=True)
    dataset = dataset.map(lambda x, i: {"id": i }, with_indices=True)
    print(len(dataset))
    dataset = dataset.filter(lambda r: 'Central Atlas Tamazight' in r['variant'])
    print(len(dataset))
    dataset = dataset.select_columns(columns_relevant)
    # Use the Audio's sampling rate rather than the
    # Dataset's schema for future concatenate_datasets()
    new_features = Features({
        "audio": Audio(sampling_rate=None),
        "text": Value("string"),
        "id":Value("string"),
    })
    dataset = dataset.cast(new_features)
    dataset = dataset.map(lambda x: {"origin":"common_voice_22_0"})
    data['common_voice_22_0'] = dataset

    return data




def gen_project_folders():
    for folder in ['dicts','corpus','corpus/kab','corpus/shi','corpus/tzm','corpus/zgh','output','output/kab_corpus','output/shi_corpus','output/tzm_corpus','output/zgh_corpus']:
        cache_dir= os.path.join(get_curr_folder(),folder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clean_project_folders():
    for folder in ['dicts','corpus','output']:
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

