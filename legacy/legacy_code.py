# According to:
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
arabic2ipa_zgh = {
    ### VOWELS AND GLIDES
    'ا':'a',# according to wiki is 'æ'
    'أ':'a',
    ' َ':'a',
    'ي':'i',
    ' ِ':'i',
    'و':'u',
    ' ُ':'u',
    'ۍ':'e',
    'ـ':'e',
    'ع':'ɛ',
    'و':'w',
    'ي':'j', # and 'j':'ʒ'

    # Bilabials
    'م':'m',
    'ب':'b',# or 'β'

    # Labiodental
    'ف':'f',

    # Dental

    # Alveolar
    'ن':'n',

    'س':'s',
    'ص':'sˤ',
    'ز':'z',
    'ژ':'zˤ',

    'ث':'t',# or 'θ' 
    'ت':'t',# or 'θ' 
    'ط':'tˤ',
    'ذ':'d',# or 'ð' 
    'د':'d',# or 'ð' 
    'ظ':'ðˤ',
    'ض':'ðˤ',

    'ل':'l', # or 'ɫ'
    'ر':'r', # or 'rˤ'
    'ر':'ɺ',# 
    'ڕ':'rˤ',# 

    # Post Alveolar
    'ش':'ʃ',
    'ت':'t͡ʃ',
    'چ':'t͡ʃ',
    'ج':'ʒ',# and 'y':'j' #CONFLICT WITH TIFINAGH2IPA
    #'ج':'d͡ʒ', #ambiguity here.

    # Palatal 

    # Velar 		
    'خ':'x',# or 'χ' 
    'غ':'ɣ',# or 'ʁ'
    'گ':'g',
    'ݣ':'g',
    #'ɡʷ':'ɡʷ',# North-berber, no proper writing in arabic
    'ک':'k',
    #'kʷ':'kʷ',#  North-berber, no proper writing in arabic

    # Uvular
    'ق':'q',# or 'qʷ' or 'ɢ'
 
    # Pharyngeal
    'ح':'ħ',# CONSENSUS 1,2,3

    # Glottal 
    'ه':'h',
}

ipa2arabic = {
    ### VOWELS AND GLIDES
    'a':'ا',# according to wiki is 'æ'
    'a':'أ',
    'a':' َ',
    'i':'ي',
    'i':' ِ',
    'u':'و',
    'u':' ُ',
    'e':'ۍ',
    'e':'ـ',
    'w':'و',
    'j':'ي', # and 'j':'ʒ'

    # Bilabials
    'm':'م',
    'b':'ب',# or 'β'

    # Labiodental
    'f':'ف',

    # Dental

    # Alveolar
    'n':'ن',

    's':'س',
    'sˤ':'ص',
    'z':'ز',
    'zˤ':'ژ',

    't':'ث',# or 'θ' 
    't':'ت',# or 'θ' 
    'tˤ':'ط',
    'd':'ذ',# or 'ð' 
    'd':'د',# or 'ð' 
    'ðˤ':'ظ',
    'ðˤ':'ض',

    'l':'ل', # or 'ɫ'
    'r':'ر', # or 'rˤ'
    'ɺ':'ر',# 
    'rˤ':'ڕ',# 

    # Post Alveolar
    'ʃ':'ش',
    't͡ʃ':'ت',
    't͡ʃ':'چ',
    'ʒ':'ج',
    #'ج':'d͡ʒ', #ambiguity here.

    # Palatal 

    # Velar 		
    'x':'خ',# or 'χ' 
    'ɣ':'غ',# or 'ʁ'
    'g':'گ',
    'g':'ݣ',
    #'ɡʷ':'ɡʷ',# North-berber, no proper writing in arabic
    'k':'ک',
    #'kʷ':'kʷ',#  North-berber, no proper writing in arabic

    # Uvular
    'q':'ق',# or 'qʷ' or 'ɢ'
 
    # Pharyngeal
    'ħ':'ح',# CONSENSUS 1,2,3
    'ع':'ɛ',

    # Glottal 
    'h':'ه',
}

# According to:
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
arabic2ipadict_kab = {
    ### VOWELS AND GLIDES
    'ا':'a',# according to wiki is 'æ'
    'أ':'a',
    ' َ':'a',
    'ي':'i',
    ' ِ':'i',
    'و':'u',
    ' ُ':'u',
    'ۍ':'e',
    'ـ':'e',
    'ع':'ɛ',
    'و':'w',
    'ي':'j', # and 'j':'ʒ'

    # Bilabials
    'م':'m',
    'ب':'b',# or 'β'

    # Labiodental
    'ف':'f',

    # Dental

    # Alveolar
    'ن':'n',

    'س':'s',
    'ص':'sˤ',
    'ز':'z',
    'ژ':'zˤ',

    'ث':'t',# or 'θ' 
    'ت':'t',# or 'θ' 
    'ط':'tˤ',
    'ذ':'d',# or 'ð' 
    'د':'d',# or 'ð' 
    'ظ':'ðˤ',
    'ض':'ðˤ',

    'ل':'l', # or 'ɫ'
    'ر':'r', # or 'rˤ'
    'ر':'ɺ',# 
    'ڕ':'rˤ',# 

    # Post Alveolar
    'ش':'ʃ',
    'ت':'t͡ʃ',
    'چ':'t͡ʃ',
    'ج':'ʒ',# and 'y':'j' #CONFLICT WITH TIFINAGH2IPA
    #'ج':'d͡ʒ', #ambiguity here.

    # Palatal, 

    # Velar 		
    'خ':'x',# or 'χ' 
    'غ':'ɣ',# or 'ʁ'
    'گ':'g',
    'ݣ':'g',
    #'ɡʷ':'ɡʷ',# North-berber, no proper writing in arabic
    'ک':'k',
    #'kʷ':'kʷ',#  North-berber, no proper writing in arabic

    # Uvular
    'ق':'q',# or 'qʷ' or 'ɢ'
 
    # Pharyngeal
    'ح':'ħ',# CONSENSUS 1,2,3

    # Glottal 
    'ه':'h',

}

#TODO
def arabic2ipa(text):
    orig = []
    trans = []
    i = 0
    while i< len(text):
        if text[i] :
            pass
    return [orig,trans]




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


