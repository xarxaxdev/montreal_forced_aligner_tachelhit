from datasets import load_dataset #using huggingface's API
from pathlib import Path
import os
import urllib



def get_curr_folder():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0])

def load_datasets():
    # Load each dataset (would be normally under ~/.cache/huggingface/datasets)
    cache_dir= os.path.join(get_curr_folder(),'.cache/huggingface/datasets')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    data = {}
    data['train'] = load_dataset("SoufianeDahimi/Tamazight-ASR-Dataset-v2",cache_dir=cache_dir)
    #todo, normalize these
    data['val'] = load_dataset("TutlaytAI/moroccan_amazigh_asr",cache_dir=cache_dir)
    data['test']= load_dataset("fsicoli/common_voice_22_0", "zgh",      trust_remote_code=True, cache_dir=cache_dir)
    return data


def gen_project_folders():
    for folder in ['dicts','tmp/tg_files','output']:
        cache_dir= os.path.join(get_curr_folder(),folder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

def download_dicts():
    dict_files= {
        'arabic_ipa.dict':'https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/763256cb0c04e9dbf0730b032d78ec9470e54188/dictionary/arabic/mfa/arabic_mfa.dict' , # ar -> IPA
    }
    
    for filename in dict_files:
        page = urllib.request.urlretrieve(dict_files[filename], f'dicts/{filename}')

def prepare_project_structure():
    gen_project_folders()
    download_dicts()

def load_dicts():
    dicts = {}
    #load
       
    return dicts


