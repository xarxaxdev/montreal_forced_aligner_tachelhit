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
from utils import NUM_PROC, BATCH_SIZE


data = {}
DICTS = {}
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

 
## All kabyle data is in latinscript 
#def latin2ipa(text):
#    transcript= []
#    for w in text.split(' '):
#        try:
#            transcript.append(DICTS['kab_latin2ipa'][w].replace(' ',''))
#        except:
#            print(f'word "{w}" not found in dictionary "latin2ipa_kab"')
#            transcript.append(w)
#    return ' '.join(transcript)
#
# One annotation = one utterance
def gen_naive_textgrid(wave,sr,transcript):
    #transcript= latin2ipa(transcript)
    t = len(wave)/sr
    #intervals at the utterance level
    tg_main =  tg_header.format(xmax=round(t,6),name='utt',interval_size=1)
    tg_main += f'\nintervals [1]:\nxmin = 0\nxmax = {t}\ntext = "{transcript}"'
    return tg_main


def transform_row(origin, waveform, sr, old_path,text):
    #print(f'row {utt}: text is "{row["text"]}"')
    cur_path = utils.get_curr_folder()# must be run before huggingface
    filename = os.path.split(old_path)[-1]
    ext = filename.split('.')[-1]
    filename = f'{origin}_{filename}'
    new_path = os.path.join(cur_path,'corpus','kab',filename)

    ### EXTRACT WAV ###
    new_sr=16000
    precision = torch.float16
    waveform = torch.tensor(waveform).to(precision)
    # Downsample and reduce precision to 16 bit
    resampler = T.Resample(orig_freq=sr, new_freq=new_sr,dtype=precision)
    waveform = resampler(waveform)
    torchaudio.save(new_path.replace(ext,'wav'), waveform, new_sr, encoding="PCM_F", bits_per_sample=16)

    ### GEN TEXTGRID ###
    raw_tg = gen_naive_textgrid(waveform,new_sr,text)
    tg = open(new_path.replace(ext, 'TextGrid'), 'w')
    tg.write(raw_tg)
    tg.close()
    #print(f'row {utt}: written TG in "{new_path}"')

def process_row(row):
    text = row['text'].lower()
    text = re.sub(r"[?.,!\":«»;\'\t\*]",'', text)

    audio = row['audio']

    audio['path'] = f'{row["id"]}.wav'

    transform_row(origin=row['origin'], waveform = audio['array'],sr = audio['sampling_rate'], old_path = audio['path'], text = text)


def main():
    data = utils.load_datasets_kab()
    #global DICTS
    #DICTS = utils.load_dicts()
    cur =  concatenate_datasets([data['common_voice_22_0']])
    # Remove rows with annoying cases
    cur.map(lambda row:{"text":re.sub(r"[?.,!\":;\'\t\*\n\“”’‘«»]", "", row["text"]).lower()},num_proc=NUM_PROC, batch_size=BATCH_SIZE)
    cur = cur.filter(lambda x: not bool(re.search(r"(\d+|…|é|ğ|ï|ⵒ|ⵠ|%|o|_|v|p|\(|\)|σ|\[|\])",x['text'])))
    print(f'{"-"*10}Generating textgrid/wav files...{"-"*10}')
    cur.map(process_row,num_proc=NUM_PROC, batch_size=BATCH_SIZE)






if __name__ == "__main__":
    main()
