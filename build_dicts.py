import utils
import re
import os 
from pathlib import Path
 

"""
Based on:
- https://polyglotclub.com/wiki/Language/Standard-moroccan-tamazight/Pronunciation/Alphabet-and-Pronunciation
- https://en.wikipedia.org/wiki/Central_Atlas_Tamazight#Phonology
- https://en.wikipedia.org/wiki/Shilha_language#Phonology
- https://en.wikivoyage.org/wiki/Tashelhit_phrasebook


Other word sources:
- https://www.livelingua.com/peace-corps/Tashelhit/tashelhit-dictionary-2011.pdf

Options: 
Tachelhit/Shilha OR Tamaziɣt/Central Atlas Tamazight
Tachelhit is #1 choice so far

For common voice:
https://huggingface.co/datasets/fsicoli/common_voice_22_0/raw/main/transcript/zgh/validated.tsv



https://universeofmemory.com/tashelhit-language-resources/

# Possible extension dictionaries
https://en.wiktionary.org/w/index.php?title=Category:Tashelhit_lemmas&pageuntil=IGIDR%0Aigidr#mw-pages
https://www.livelingua.com/peace-corps/Tashelhit/tashelhit-dictionary-2011.pdf
https://friendsofmorocco.org/Docs/Tashlheet/tashlheettextbook2011.pdf


# 4000 WORDS
https://friendsofmorocco.org/Docs/Dict/Tamazizght%20T-E.htm
# 22k SENTENCES
https://tatoeba.org/en/downloads
https://downloads.tatoeba.org/exports/per_language/shi/shi_sentences.tsv.bz2
# Wikipedia - Nuclear option; 14k pages however major cleanup needed
https://shi.wikipedia.org/wiki/Tasna_Tamzwarut
https://dumps.wikimedia.org/other/mediawiki_content_current/shiwiki/2026-02-01/xml/bzip2/
https://medium.com/@evan.frank/accessing-and-cleaning-bulk-wikipedia-text-data-bfde3b550474
"""




# REVISE
# according to "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
# by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
# only voiced h "murmured glottal fricative (‘voiced h’)."
#
# SOURCE 1
# Which is the correct writing according to: https://en.wikipedia.org/wiki/Shilha_language#Writing_systems
# From: https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters
# SOURCE 2
# https://en.wiktionary.org/wiki/Module:Tfng-translit
# SOURCE 3
# https://www.mdpi.com/2078-2489/16/7/600
# SOURCE 4
# https://ieeexplore.ieee.org/abstract/document/8284715
# SOURCE 5
# https://commons.wikimedia.org/wiki/Tifinagh

# SOURCE 6 (for specific berber variants)
#https://en.wikipedia.org/wiki/Berber_Latin_alphabet
tifinagh2ipa_dict = {
    ### VOWELS AND GLIDES
    #'ⴰ':'æ',# SRC 1
    'ⴰ':'a',# SRC 2,3,4
    #'ⴻ':'ə',# SRC 1  CONFLICT!
    'ⴻ':'e',# SRC 2,3,4; in normal Berber is 'ə' but not in Tachelhit
    #https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    'ⵉ':'i',# CONSENSUS SRC 1,2,3,4
    #'ⵄ':'ʕ',# SRC 1  CONFLICT!
    'ⵄ':'ɛ',# SRC 2,3,4  CONFLICT!
    'ⵓ':'u', #SRC 2,3,4  CONFLICT!
    #'ⵓ':'ʊ', in latin alphabet
    #'ⵓ':'w', #SRC 1   CONFLICT!
    'ⵡ':'w',# CONSENSUS 1,2,3,4
    'ⵢ':'j',# SRC 1,3,4
    'ⵊ':'ʒ',# SRC 1 CONFLICT!
    #'ⵊ':'j',# SRC 2,3,4 CONFLICT!
    'ⵋ':'j',# SRC 2
    'ⵌ':'j',# SRC 2
    'ⵘ':'j',# SRC 2

    'ⵧ':'o', # SRC 1,2


    # Bilabials
    'ⴱ':'b',# CONSENSUS 1,2,3,4
    'ⴱⵯ':'bʷ',# SRC 6; velarization
    'ⴲ':'β',# IRCAM EXTENDED fricative; SRC 1,2
    'ⵒ':'p', # IRCAM EXTENDED; SRC 1,2
    'ⵎ':'m',# CONSENSUS 1,2,3,4
    'ⵎⵯ':'mʷ',# SRC 6;velarization

    # Labiodental
    'ⴼ':'f',# CONSENSUS 1,2,3,4
    'ⵠ':'v', # IRCAM EXTENDED

    # Dental
    'ⵝ':'θ',# IRCAM EXTENDED fricative; SRC 1,2
    'ⴺ':'ðˤ',# IRCAM EXTENDED fricative; SRC 1,2 
    'ⴸ':'ð',# SRC 2

 
    # Alveolar
    'ⵏ':'n',# CONSENSUS 1,2,3,4
    
    'ⵙ':'s',# CONSENSUS 1,2,3,4
    'ⵚ':'sˤ',# CONSENSUS 1,2,3,4
    'ⵣ':'z',# CONSENSUS 1,2,3,4
    'ⵥ':'zˤ',# CONSENSUS 1,2,3,4
    
    'ⵜ':'t',# CONSENSUS 1,2,3,4
    'ⵟ':'tˤ',# CONSENSUS 1,2,3,4
    'ⴷ':'d', # CONSENSUS 1,2,3,4
    'ⴹ':'dˤ',# CONSENSUS 1,2,3,4
    
    'ⵍ':'l', # CONSENSUS 1,2,3,4
    'ⵔ':'r', # CONSENSUS 1,2,3,4
    'ⵕ':'rˤ',# CONSENSUS 1,2,3,4

    # Post Alveolar
    'ⵛ':'ʃ',# CONSENSUS 1,2,3,4 
    'ⴵ':'d͡ʒ',# SRC 1,2
    'ⴶ':'d͡ʒ',# SRC 2
    'ⵞ':'t͡ʃ',# SRC 1,2 

    # Palatal 
    'ⵐ':'ny',# SRC 2

    # Velar 		
    'ⵆ':'x',# SRC 2
    'ⴿ':'x',# IRCAM EXTENDED fricative; SRC 1 
    #'ⵅ':'χ',# SRC 1 CONFLICT!
    'ⵅ':'x',# SRC 2,3,4 CONFLICT!
    'ⵅⵯ':'xʷ',# SRC 2,3,4 CONFLICT!
    'ⵖ':'ɣ',# CONSENSUS 1,2,3,4
    'ⵖⵯ':'ɣʷ',# SRC 6; velarization
    'ⵗ':'ɣ',# SRC 2

    'ⴳ':'g',# CONSENSUS 1,2,3,4
    'ⴴ':'g',# SRC 2 (SRC 5 points to this being aproximant);   CONFLICT!
    #'ⴴ':'ʝ',# IRCAM EXTENDED fricative;SRC 1 CONFLICT!
    'ⴳⵯ':'ɡʷ',# CONSENSUS 1,2,3,4
    # SRC 6; velarization
    'ⴽ':'k',# CONSENSUS 1,2,3,4
    'ⴾ':'k',# SRC 2
    'ⴿ':'k',# SRC 2 
    'ⴽⵯ':'kʷ',# CONSENSUS 1,2,3,4
    # SRC 6; velarization

    # Uvular
    'ⵇ':'q',# CONSENSUS 1,2,3,4
    'ⵇⵯ':'qʷ',# SRC 6;velarization
    'ⵈ':'q',# SRC 2

    # Pharyngeal
    'ⵃ':'ħ',# CONSENSUS 1,2,3

    # Glottal 
    'ⵀ':'h',# CONSENSUS 1,2,3,4; SRC 2 does not mention if for shi
    'ⵂ':'h',# SRC 2
    'ⵁ':'h',# IRCAM EXTENDED; SRC 1,2
    
    # multi-letter
    'ⵑ':'ng',# SRC 2
    'ⵐ':'ny',# SRC 2
}

ipa2tifinagh_dict = {} #the inverse dictionary
for k in tifinagh2ipa_dict:
    ipa2tifinagh_dict[tifinagh2ipa_dict[k]] = k


def tifinagh2ipa(text):
    # Consider supporting diphtongs 
    # /ts/ and /dz/
    # /d͡ʒ/ and /t͡ʃ/
    orig = []
    trans = []
    i = 0
    while i< len(text):
        #check for multi-character phones
        if text[i] in ['ⴽ','ⴳ'] and i+1 < len(text) and text[i+1] =='ⵯ':
            k = ''.join([text[i],text[i+1]])
            trans.append(tifinagh2ipa_dict[k])
            i += 1
        else :
            trans.append(tifinagh2ipa_dict[text[i]])
        i += 1
    return [orig,trans]

# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
# and IRCAM Tifinagh~latin~arabic alphabet equivalence   
latin2ipadict = { 
    ### VOWELS AND GLIDES
    # https://en.wikipedia.org/wiki/Shilha_language#Vowels
    'a':'a',# according to wiki is 'æ'
    'i':'i',
    'u':'u',# according to wiki is 'ʊ'
    'e':'e',# in normal Berber is 'ə' but not in Tachelhit
    #https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    'ɛ':'ɛ',#according to wiki 'ʕ',
    'â':'ɛ',#according to wiki 'ʕ',
    'w':'w',
    'y':'j', # and 'j':'ʒ'

    # Bilabials
    'm':'m',
    'mʷ':'mʷ',
    'mᵒ':'mᵒ',
    'b':'b',# or 'β'
    'bʷ':'bʷ',
    'bᵒ':'bᵒ',

    # Labiodental
    'f':'f',

    # Dental

    # Alveolar
    'n':'n',

    's':'s',
    'ṣ':'sˤ',
    'z':'z',
    'ẓ':'zˤ',

    't':'t',# or 'θ' 
    'ţ':'t͡s',
    'ṭ':'tˤ',
    'd':'d',# or 'ð' 
    'ḍ':'ðˤ',
    'z̧':'d͡z',

    'l':'l', # or 'ɫ'
    'r':'r', # or 'rˤ'
    'ṛ':'rˤ',# 
    'ř':'ɺ',# between r and l, Rif Berber
    # according to "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # r should be IPA ɾ or r depending on context

    # Post Alveolar
    'c':'ʃ',
    'č':'t͡ʃ',
    'j':'ʒ',# and 'y':'j' #CONFLICT WITH TIFINAGH2IPA
    'dj':'d͡ʒ',
    #'ll':'d͡ʒ',# Rif-BERBER Only
    'ǧ':'d͡ʒ',
    'ǧǧ':'d͡ʒ',

    # Palatal 

    # Velar 		
    'x':'x',# or 'χ' 
    'xʷ':'xʷ',
    'xᵒ':'xᵒ',
    'ɣ':'ɣ',# or 'ʁ'
    'ɣʷ':'ɣʷ',
    'ɣᵒ':'ɣᵒ',
    'g':'g',
    'ɡʷ':'ɡʷ',
    'ɡᵒ':'ɡᵒ',
    'k':'k',
    'kʷ':'kʷ',
    'kᵒ':'kᵒ',

    # Uvular
    'q':'q',# or 'qʷ' or 'ɢ'
    'qʷ':'qʷ',
    'qᵒ':'qᵒ',
 
    # Pharyngeal
    'ḥ':'ħ',# CONSENSUS 1,2,3

    # Glottal 
    'h':'h',
    
    # My annotations with 
    # https://huggingface.co/datasets/TutlaytAI/tamazight_asr
    #'t':'θ',
    #'ṭ':'t',
    #'s':'s',
    #'tt':'ts',
    #'z':'z',
    #'ṭṭ':'tˤ',
    #'y':'j',
    #'x':'χ',
    #'ʕ':'ʕ',
    #'ṛ':'rˤ',
    #'ṣ':'sˤ',
    #'ẓ':'zˤ',
}


def latins2ipa(text):
    orig = []
    trans = []
    i = 0
    while i< len(text):
        if text[i] :
            pass
    return [orig,trans]

# According to:
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
arabic2ipadict = {
    ### VOWELS AND GLIDES
    'ا':'a',# according to wiki is 'æ'
    'أ':'a',# according to wiki is 'æ'
    ' َ':'a',# according to wiki is 'æ'
    'ي':'i',
    ' ِ':'i',
    'و':'u',# according to wiki is 'ʊ'
    ' ُ':'u',# according to wiki is 'ʊ'
    'ۍ':'e',# according to wiki is 'ə'
    'ـ':'e',# according to wiki is 'ə'
    'ع':'ɛ',# according to wiki 'ʕ',
    'و':'w',
    'ي':'j', # and 'j':'ʒ'

    # Bilabials
    'م':'m',
    'ب':'b',# or 'β'
    #'p':'p',

    # Labiodental
    'ف':'f',
    #'v':'v',

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


def arabic2ipa(text):
    # TO DO 
    # Consider GEMMINATES: jj 
    orig = []
    trans = []
    i = 0
    while i< len(text):
        if text[i] :
            pass
    return [orig,trans]


def main():
    utils.prepare_project_structure()
    data = utils.load_datasets()
    cur =  data['common_voice_22_0']
    # we want the dictionary in both directions:
    # tifinagh2ipa: to transcribe our datasets
    # ipa2tifinagh: to take note of homophones
    # ambigous pronunciation(IRCAM extended only):    
    #'ⵁ', 'ⵀ':'h' #AMBIGUOUS IN IPA
    # 'ⵓ' , 'ⵡ' : 'w' #AMBIGUOUS IN IPA
    # 'ⵜ', 'ⵝ':'t'  
    # 'ⴽ', 'ⴿ': 'k'
    # 'ⴱ', 'ⴲ' : 'b'
    # 'ⴳ','ⴴ' :'g'
    dicts = {}
    dicts['tifinagh2ipa'] = {}
    dicts['ipa2tifinagh'] = {}

    # We want to collect the totality of words that exist in Tashelhit
    vocab = {}# a set is more fitting, but lists do not have a builtin way to get hashed for sets.
    for row in cur:
        words = re.sub(r"[?.,!\":;\'\t\*]",'', row['text']).split(' ')
        for w in words:
            #function returns them as array of symbols
            orig,trans = tifinagh2ipa(w)
            w_trans = ''.join(trans)
            # Standard Pronunciation dictionary format
            vocab[w_trans] = ' '.join(trans)
            dicts['tifinagh2ipa'][w]= ' '.join(w_trans)
            # We keep this as a safety check, so the format is designed for that
            if not w_trans in dicts['ipa2tifinagh']:
                dicts['ipa2tifinagh'][w_trans] = [w]
            # homophone check
            elif not (w in dicts['ipa2tifinagh'][w_trans]):
                dicts['ipa2tifinagh'][w_trans].append(w)

    print('-' *15)
    print('SAVING DICTS')
    print('-' *15)
    cur_path = utils.get_curr_folder()
    #del vocab['']
    # Write ipa-to-else dicts
    for d in dicts:
        filename = os.path.join(cur_path,'dicts',d+'.dict')
        with open(filename,'w') as f:
            f.write('<unk>\tspn\n')
            for key in dicts[d]:
                f.write(f'{key}\t{dicts[d][key]}\n')
            f.close()
    # Write ipa-to-ipa dict
    with open(os.path.join(cur_path,'dicts','vocab.dict'),'w') as f:
        f.write('<unk>\tspn\n')
        for w in vocab:
            f.write(f'{w}\t{vocab[w]}\n')
        f.close()


if __name__ == "__main__":
    main()

