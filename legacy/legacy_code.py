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


