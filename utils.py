from datasets import Dataset,load_dataset,Audio,concatenate_datasets #using huggingface's API
import numpy as np
from librosa import resample
from pathlib import Path
import os,sys
import urllib
from tqdm import tqdm

THRESHOLD_MIN_SECONDS = 0.25
DATASETS = {
    'tamasightASRDatasetV2':'https://huggingface.co/datasets/SoufianeDahimi/Tamazight-ASR-Dataset-v2',

}
DATASET_DICTS = {
    'tamasightASRDatasetV2':'arabic_ipa', #unlikely to be useful
    'moroccan_amazigh_asr':'tzm_tfng2ipa',
    'common_voice_22_0':'tzm_tfng2ipa',
}

# Name format is {language}_{script_before}2{script_after}.dict
DICT_FILES = {
        'arabic_arabic2ipa.dict':'https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/763256cb0c04e9dbf0730b032d78ec9470e54188/dictionary/arabic/mfa/arabic_mfa.dict' , 
        'tzm_tfng2ipa.dict':'https://raw.githubusercontent.com/CUNY-CL/wikipron/refs/heads/master/data/scrape/tsv/tzm_tfng_broad.tsv' ,
    }

DICTS = {}
    

def get_curr_folder():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0])

def reshape_audio(row):
    audio = row["audio"]
    waveform = np.asarray(audio["array"], dtype=np.float32)
    waveform = resample(waveform,orig_sr=int(audio["sampling_rate"]),target_sr=16000)
    waveform = np.asarray(audio["array"], dtype=np.float32)

    return {
        "waveform": waveform,
        "sr": 16000,
        "filename": str(audio["path"]),
        "origin": "tamasightASRDatasetV2",
    }


def load_datasets():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)

    ### ARABSCRIPT DATASET
    cache_dir= os.path.join(get_curr_folder(),'.cache/huggingface/datasets')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    data = {}
    dataset = load_dataset("SoufianeDahimi/Tamazight-ASR-Dataset-v2",cache_dir=cache_dir)
    dataset = dataset['train'].take(3)
    #dataset = concatenate_datasets(dataset['train'],dataset['test'])# TODO LATER
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
                                                                         
    #dataset = dataset.filter(lambda r: len(r['waveform'])/r['sr']<THRESHOLD_MIN_SECONDS)
    data['tamasightASRDatasetV2'] = dataset

    #todo, normalize these

    ### LATINSCRIPT DATASET
    dataset = load_dataset("TutlaytAI/moroccan_amazigh_asr",cache_dir=cache_dir)
    dataset = dataset['train']
    dataset = dataset.rename_column("transcription","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])
    data['moroccan_amazigh_asr'] = dataset

    ### AMAZIGHSCRIPT DATASET 
    dataset = load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=cache_dir)
    dataset = dataset['train']
    dataset = dataset.rename_column("sentence","text")
    dataset = dataset.map(reshape_audio,remove_columns=[c for c in dataset.column_names if c != 'text'])

    #print(dataset)

    data['common_voice_22_0'] = dataset
    #assert(False)

    return data


def gen_project_folders():
    for folder in ['dicts','corpus','output']:
        cache_dir= os.path.join(get_curr_folder(),folder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clean_project_folders():
    for folder in ['dicts','corpus','output']:
        path = os.path.join(get_curr_folder(),folder)
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))


def download_dicts():
    for filename in DICT_FILES:
        page = urllib.request.urlretrieve(DICT_FILES[filename], f'dicts/{filename}')

def prepare_project_structure():
    gen_project_folders()
    clean_project_folders() #in case they existed
    download_dicts()

def load_dicts():
    path = os.path.join(get_curr_folder(),'dicts')
    for f in os.listdir(path):
        d_name = f.split('.')[0]
        print(d_name)
        DICTS[d_name] = {}
        with open(os.path.join(path,f)) as d:
            for line in  d:
                [left,right] = line.split('\t')
                DICTS[d_name][left]=right
    return DICTS


