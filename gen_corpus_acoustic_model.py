import utils
import os 
from pathlib import Path
from scipy.io import wavfile

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
xmax =  {xmax}
intervals: size = {interval_size}"""
 


# One annotation = one phone
def gen_textgrid(wave,sr,transcript):
    #per-file textgrid generation
    t = len(wave)/sr
    #intervals at the phone level
    tg_main =  tg_header.format(xmax=round(t,6),name='phon',interval_size=len(transcript))

    time_per_phon = round(t, 6) / len(transcript)
    phon_start = 0
    interval_counter = 1
    for phon in transcript:
        tg_entry = f'intervals [{interval_counter}]:\nxmin = {phon_start}\nxmax = {phon_start+time_per_phon}\ntext = "{phon}"'
        phon_start += time_per_phon
        interval_counter +=1
        tg_main += '\n' + tg_entry

    return tg_main

# One annotation = one utterance
def gen_naive_textgrid(wave,sr,transcript):
    t = len(wave)/sr
    #intervals at the utterance level
    tg_main =  tg_header.format(xmax=round(t,6),name='utt',interval_size=1)
    tg_main += f'\nintervals [1]:\nxmin = 0\nxmax = {t}\ntext = "{transcript}"'
    return tg_main



def main():
    utils.prepare_project_structure()
    data = utils.load_datasets()
    cur =  data['train']['train']
    cur = cur.take(2) # only 5 rows for debugging
    cur_path = utils.get_curr_folder()
    utt=1
    print(f'{"-"*10}Generating textgrid/wav files...{"-"*10}')
    for row in cur :
        #print(row)
        audio= row['audio']
        filename = audio['path']
        filename = filename.replace('.wav',f'_{utt}.wav')
        utt+=1
        filename = os.path.join(cur_path,'corpus',filename)
        waveform = audio['array']
        sr = audio['sampling_rate']
        ### EXTRACT WAV ###
        wavfile.write(filename,sr,waveform)
        ### GEN TEXTGRID ###
        raw_tg = gen_naive_textgrid(waveform,sr,row['text'])
        tg = open(filename.replace('.wav', '.TextGrid'), 'w')
        tg.write(raw_tg)
        tg.close()




if __name__ == "__main__":
    main()
