from datasets import Dataset,load_dataset,Audio,concatenate_datasets #using huggingface's API
import numpy as np
from librosa import resample
from pathlib import Path
import os,sys
import urllib
from tqdm import tqdm

THRESHOLD_MIN_SECONDS = 0.25


    

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
    for folder in ['dicts','corpus','output','output/corpus_aligned']:
        cache_dir= os.path.join(get_curr_folder(),folder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clean_project_folders():
    for folder in ['dicts','corpus','output','output/corpus_aligned']:
        path = os.path.join(get_curr_folder(),folder)
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))


def prepare_project_structure():
    gen_project_folders()
    clean_project_folders() #in case they existed
    download_dicts()

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

