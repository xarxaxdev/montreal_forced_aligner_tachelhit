import utils
import os 
from pathlib import Path
import soundfile as sf


data = {}
tg_header = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = W
tiers? <exists>
size = 1
item []:
item [1]:
class = "IntervalTier"
name = "{name}"
xmin = 0
xmax =  {xmax}
intervals: size = {interval_size}"""
 
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


def main():
    utils.prepare_project_structure()
    data = utils.load_datasets()
    cur =  data['train']['train']
    cur = cur.take(2)
    
    cur_path = utils.get_curr_folder()
    print(f'{"-"*10}Generating textgrid files...{"-"*10}')
    for row in cur :
        #print(row)
        audio= row['audio']
        #print(audio.keys())
        waveform = audio['array']
        filename= audio['path']
        ### EXTRACT WAV ###
        sf.write(os.path.join(cur_path,'corpus',filename),audio['array'],audio['sampling_rate'])
        ### GEN TEXTGRID ###
        raw_tg = gen_textgrid(audio['array'],audio['sampling_rate'],row['text'])
        # write .TextGrid in target directory
        tg = open(os.path.join( cur_path,'corpus',filename.replace('wav', 'TextGrid')), 'w')
        tg.write(raw_tg)
        tg.close()
    




if __name__ == "__main__":
    main()
