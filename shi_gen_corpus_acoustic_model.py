from datasets import Audio,concatenate_datasets #using huggingface's API
import utils
import re
import os,sys,subprocess
from pathlib import Path
from scipy.io import wavfile
import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
from tqdm import tqdm


# TODO make filenames shorter
from utils import dataset_alias as dataset_alias
from utils import trim_trailing_silence


data = {}
tg_header = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {xmax}
tiers? <exists>
size = 1
item []:
item [1]:
class = "IntervalTier"
name = "{name}"
xmin = 0
xmax = {xmax}
intervals: size = {interval_size}"""

# One annotation = one utterance
def gen_naive_textgrid(wave,sr,transcript):
    t = len(wave)/sr
    #intervals at the utterance level
    tg_main =  tg_header.format(xmax=round(t,6),name='utt',interval_size=1)
    tg_main += f'\nintervals [1]:\nxmin = 0\nxmax = {t}\ntext = "{transcript}"'
    return tg_main


def transform_row(origin, waveform, sr, text, row_id):
    cur_path = utils.get_curr_folder()# must be run before huggingface
    ext = 'wav'
    filename = f'{origin}_{row_id}.{ext}'
    new_path = os.path.join(cur_path,'corpus','shi',filename)

    ### EXTRACT WAV ###
    new_sr=16000
    precision = torch.float16
    waveform = torch.tensor(waveform).to(precision)
    # Downsample and reduce precision to 16 bit
    resampler = T.Resample(orig_freq=sr, new_freq=new_sr,dtype=precision)
    waveform = resampler(waveform)
    waveform = trim_trailing_silence(waveform)
    torchaudio.save(new_path.replace(ext,'wav'), waveform, new_sr, encoding="PCM_F", bits_per_sample=16)

    ### GEN TEXTGRID ###
    raw_tg = gen_naive_textgrid(waveform,new_sr,text)
    tg = open(new_path.replace(ext, 'TextGrid'), 'w')
    tg.write(raw_tg)
    tg.close()

def main():
    print('Loading datasets and assigning ids...')
    data = utils.load_datasets_shi()
    print('Loaded.')
    cur =  concatenate_datasets([data['common_voice_22_0']])
    print(f'{"-"*10}Generating textgrid/wav files...{"-"*10}')
    for row in cur:
        row['text']= row['text'].replace('[]-','')
        words = re.sub(r"[?.,!\":;\'\t\*\n]",'', row['text']).lower().split(' ')
        text = " ".join(words)


        words = re.sub(r"[?.,!\":«»;\'\t\*]",'', text).split(' ')
        text = ' '.join(words)
        if bool(re.search(r'(\d+|%|p|o|_|v|\(|\)|σ)',text)):
            #skip rows with invalid symbols
            continue

        audio = row['audio']

        transform_row(origin=row['origin'], waveform = audio['array'],sr = audio['sampling_rate'], text = text, row_id=row["id"])





if __name__ == "__main__":
    main()
