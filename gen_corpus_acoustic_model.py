import utils
import re
import os,sys,subprocess
from pathlib import Path
from scipy.io import wavfile
import numpy as np

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
 

def tifinagh2ipa(pron_dict,text):
    words = re.sub(r"[?.,!\":;\'\t]",'', text).split(' ')
    transcript= []
    for w in words:
        transcript.append(DICTS['tifinagh2ipa'][w].replace(' ',''))
    return ' '.join(transcript)

# One annotation = one utterance
def gen_naive_textgrid(pron_dict,wave,sr,transcript):
    transcript= tifinagh2ipa(pron_dict,transcript)
    t = len(wave)/sr
    #intervals at the utterance level
    tg_main =  tg_header.format(xmax=round(t,6),name='utt',interval_size=1)
    tg_main += f'\nintervals [1]:\nxmin = 0\nxmax = {t}\ntext = "{transcript}"'
    return tg_main


def transform_row(pron_dict,waveform, sr, old_path,text):
    #print(f'row {utt}: text is "{row["text"]}"')
    cur_path = utils.get_curr_folder()# must be run before huggingface
    filename = os.path.split(old_path)[-1]
    ext = filename.split('.')[-1]
    new_path = os.path.join(cur_path,'corpus',filename)
    ### EXTRACT WAV ###
    # Downsample and reduce precision to 16 bit
    command = ['sox', old_path, '-t', 'wav', '-r', '16000', '-b', '16', new_path.replace(ext,'wav')]
    subprocess.check_call(command)

    #print(f'row {utt}: written wavfile in "{new_path}"')
    ### GEN TEXTGRID ###
    raw_tg = gen_naive_textgrid(pron_dict,waveform,sr,text)
    tg = open(new_path.replace(ext, 'TextGrid'), 'w')
    tg.write(raw_tg)
    tg.close()
    #print(f'row {utt}: written TG in "{new_path}"')

def main():
    data = utils.load_datasets()
    global DICTS
    DICTS = utils.load_dicts()
    print(len(DICTS))
    cur =  data['common_voice_22_0']
    cur = cur.take(20) # debugging
    utt=1
    print(f'{"-"*10}Generating textgrid/wav files...{"-"*10}')
    for row in cur :
        #TODO ADD A CSV WITH ORIGINAL UTTERANCE, TRANSLATED UTTERANCE, ORIGINAL DATASET, FILENAME
        transform_row(pron_dict = 'tifinagh2ipa',waveform = row['waveform'],sr =row['sr'],old_path=row['filename'], text =row['text'])

        utt+=1




if __name__ == "__main__":
    main()
