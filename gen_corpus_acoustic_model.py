import utils
import re
import os 
from pathlib import Path
from scipy.io import wavfile
import numpy as np

cur_path = utils.get_curr_folder()# must be run before huggingface
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
 

def tifinagh2ipa(text):
    words = re.sub(r"[?.,!\":;\'\t]",'', text).split(' ')
    transcript= []
    for w in words:
        transcript.append(DICTS['tifinagh2ipa'][w].replace(' ',''))
    return ' '.join(transcript)

# One annotation = one utterance
def gen_naive_textgrid(wave,sr,transcript):
    transcript= tifinagh2ipa(transcript)
    t = len(wave)/sr
    #intervals at the utterance level
    tg_main =  tg_header.format(xmax=round(t,6),name='utt',interval_size=1)
    tg_main += f'\nintervals [1]:\nxmin = 0\nxmax = {t}\ntext = "{transcript}"'
    return tg_main


def main():
    cur_path = utils.get_curr_folder()# must be run before huggingface
    data = utils.load_datasets()
    global DICTS
    DICTS = utils.load_dicts()
    print(len(DICTS))
    cur =  data['common_voice_22_0']
    #cur = cur.take(500) # only 5 rows for debugging
    utt=1
    print(f'{"-"*10}Generating textgrid/wav files...{"-"*10}')
    for row in cur :
        print(f'row {utt}: text is "{row["text"]}"')
        waveform = row['waveform']
        sr = row['sr']
        filename = row['filename']
        filename = filename.replace('.mp3',f'_{utt}.mp3')
        filename = os.path.split(filename)[-1]
        utt+=1
        filename = os.path.join(cur_path,'corpus',filename)
        ### EXTRACT WAV ###
        #wavfile.write(filename,sr,waveform.astype(np.int16))
        print(filename)
        waveform = np.asarray(waveform, dtype=np.float32)
        wavfile.write(filename,sr,waveform)
        print(f'row {utt}: written wavfile in "{filename}"')
        ### GEN TEXTGRID ###
        raw_tg = gen_naive_textgrid(waveform,sr,row['text'])
        tg = open(filename.replace('.mp3', '.TextGrid'), 'w')
        tg.write(raw_tg)
        tg.close()
        print(f'row {utt}: written TG in "{filename}"')




if __name__ == "__main__":
    main()
