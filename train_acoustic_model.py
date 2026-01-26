import utils
import os 
from pathlib import Path


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
 
def gen_textgrid(name,wave,sr,transcript):
    #per-file textgrid generation
    t = len(wave)/sr
    #intervals at the phone level
    tg_main =  tg_header.format(xmax=round(t,6),name='phon',interval_size=len(transcript))

    time_per_phon = round(t, 6) / len(transcript)
    phon_start = 0
    interval_counter = 1
    for phon in transcript:
        # DO SOMETHING SMARTER
        tg_entry = f'intervals [{interval_counter}]:\nxmin = {phon_start}\nxmax = {phon_start+time_per_phon}\ntext = "{phon}"'
        phon_start += time_per_phon
        interval_counter +=1
        tg_main += '\n' + tg_entry

    return tg_main


def main():
    utils.prepare_project_structure()
    data = utils.load_datasets()
    cur =  data['train']['train']
    #{path:,array:,sampling_rate:}
    cur = cur.take(2)
    cur_path = utils.get_curr_folder()
    #print(cur['audio'])
    print(f'{"-"*10}Generating textgrid files...{"-"*10}')
    for row in cur :
        filename= row['audio']['path']
        #print(row)
        #print(cur_path)
        #print(filename)
        raw_tg = gen_textgrid(filename, row['audio']['array'],row['audio']['sampling_rate'],row['text'])
        Path(os.path.join(cur_path, 'tg_files')).mkdir(parents=True, exist_ok=True)
        # write .TextGrid in target directory
        tg = open(os.path.join( cur_path,'tg_files',filename.replace('wav', 'TextGrid')), 'w')
        tg.write(raw_tg)
        tg.close()
    




if __name__ == "__main__":
    main()
