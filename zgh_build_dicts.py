from datasets import Audio, concatenate_datasets #using huggingface's API
import utils
import re
import os 
from pathlib import Path
import sys

 
# pronunciation dictionary?
# https://huggingface.co/datasets/prothmane/amawal-dataset
# https://huggingface.co/datasets/omarkamali/wikipedia-monthly
# https://huggingface.co/datasets/Tamazight-NLP/IRCAM-CORPUS

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
tifinagh2ipa = {
    ### VOWELS AND GLIDES
    #'ⴰ':'æ',# SRC 1
    'ⴰ':'a',# SRC 2,3,4
    #'ⴻ':'ə',# SRC 1  CONFLICT!
    'ⴻ':'e',# SRC 2,3,4; in normal Berber is 'ə' but not in Tachelhit
    #https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    'ⵉ':'i',# CONSENSUS SRC 1,2,3,4
    'ⵓ':'u', #SRC 2,3,4  CONFLICT!
    #'ⵓ':'ʊ', in latin alphabet
    #'ⵓ':'w', #SRC 1   CONFLICT!
    'ⵡ':'w',# CONSENSUS 1,2,3,4
    'ⵢ':'j',# SRC 1,3,4
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
    'ⵊ':'ʒ',# SRC 1 CONFLICT!
    #'ⵊ':'j',# SRC 2,3,4 CONFLICT!
    'ⴵ':'d͡ʒ',# SRC 1,2
    'ⴶ':'d͡ʒ',# SRC 2
    'ⵞ':'t͡ʃ',# SRC 1,2 

    # Palatal 
    'ⵐ':'ny',# SRC 2

    # Velar 		
    #'ⵅ':'χ',# SRC 1 CONFLICT!
    'ⵅ':'x',# SRC 2,3,4 CONFLICT!
    'ⵅⵯ':'xʷ',# SRC 2,3,4 CONFLICT!
    'ⵆ':'x',# SRC 2
    'ⴿ':'x',# IRCAM EXTENDED fricative; SRC 1 
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
    'ⵄ':'ʕ',# SRC 1  CONFLICT! #Tinifagh
    #'ⵄ':'ɛ',# SRC 2,3,4  CONFLICT! #Latin, more common

    # Glottal 
    'ⵀ':'h',# CONSENSUS 1,2,3,4; SRC 2 does not mention if for shi
    'ⵂ':'h',# SRC 2
    'ⵁ':'h',# IRCAM EXTENDED; SRC 1,2
    
    # multi-letter
    'ⵑ':'ng',# SRC 2

    #clitics
    '-':'-',
}


ipa2tifinagh = {
    ### VOWELS AND GLIDES
    'a':'ⴰ',
    'e':['ⴻ',''],
    'i':'ⵉ',
    'u':'ⵓ',
    'o':['ⵧ',''],

    'w':'ⵡ',
    'j':'ⵢ',

    # Bilabials
    'b':'ⴱ',
    'bʷ':'ⴱⵯ',
    'β':'ⴲ',
    'p':'ⵒ',
    'm':'ⵎ',
    'mʷ':'ⵎⵯ',

    # Labiodental
    'f':'ⴼ',
    'v':'ⵠ',

    # Dental
    'θ':'ⵝ',
    'ðˤ':'ⴺ',
    'ð':'ⴸ',
 
    # Alveolar
    'n':'ⵏ',
    
    's':'ⵙ',
    'sˤ':'ⵚ',
    'z':'ⵣ',
    'zˤ':'ⵥ',
    
    't':'ⵜ',
    'tˤ':'ⵟ',
    'd':'ⴷ',
    'dˤ':'ⴹ',
    
    'l':'ⵍ',
    'r':'ⵔ', 
    'rˤ':'ⵕ',

    # Post Alveolar
    'ʃ':'ⵛ',
    'ʒ':'ⵊ',
    'd͡ʒ':'ⴵ',
    't͡ʃ':'ⵞ',

    # Velar 		
    'x':'ⵅ',
    'xʷ':'ⵅⵯ',
    'ⴿ':'x',
    'ɣ':'ⵖ',
    'ɣʷ':'ⵖⵯ',

    'g':'ⴳ',
    'ɡʷ':'ⴳⵯ',
    'k':'ⴽ',
    'kʷ':'ⴽⵯ',

    # Uvular
    'q':'ⵇ',
    'qʷ':'ⵇⵯ',

    # Pharyngeal
    'ħ':'ⵃ',
    'ʕ':['ⵄ','ɛ'],


    # Glottal 
    'h':['ⵀ','ⵁ'],

    #clitics
    '-':'-',
    '-':'-',
}



# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
# and IRCAM Tifinagh~latin~arabic alphabet equivalence   
latin2ipa = { 
    ### VOWELS AND GLIDES
    # https://en.wikipedia.org/wiki/Shilha_language#Vowels
    'a':'a',# according to wiki is 'æ'
    'e':'e',# in normal Berber is 'ə' but not in Tachelhit
    #https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    'i':'i',
    'u':'u',# according to wiki is 'ʊ'
    'w':'w',
    'y':'j', # and 'j':'ʒ'

    # Bilabials
    'b':'b',# or 'β'
    'bʷ':'bʷ',
    'bᵒ':'bʷ',
    'm':'m',
    'mʷ':'mʷ',
    'mᵒ':'mʷ',

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
    'ṭ':'tˤ',
    'ţ':'t͡s',
    'd':'d',# or 'ð' 
    'ḍ':'ðˤ',# or 'dˤ'
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
    'j':'ʒ',# and 'y':'j' 
    'dj':'d͡ʒ',
    #'ll':'d͡ʒ',# Rif-BERBER Only
    'ǧ':'d͡ʒ',
    'ǧǧ':'d͡ʒ',

    # Palatal 

    # Velar 		
    'x':'x',# or 'χ' 
    'xʷ':'xʷ',
    'xᵒ':'xʷ',
    'ɣ':'ɣ',# or 'ʁ'
    'ɣʷ':'ɣʷ',
    'ɣᵒ':'ɣʷ',

    'g':'g',
    'ɡʷ':'ɡʷ',
    'ɡᵒ':'ɡʷ',
    'k':'k',
    'kʷ':'kʷ',
    'kᵒ':'kʷ',

    # Uvular
    'q':'q',# or 'qʷ' or 'ɢ'
    'qʷ':'qʷ',
    'qᵒ':'qʷ',
 
    # Pharyngeal
    'ḥ':'ħ',# CONSENSUS 1,2,3
    'ɛ':'ʕ',
    'â':'ʕ',

    # Glottal 
    'h':'h',
    
    #clitics
    '-':'-',
}

ipa2latin = { 
    ### VOWELS AND GLIDES
    # https://en.wikipedia.org/wiki/Shilha_language#Vowels
    'a':'a',# according to wiki is 'æ'
    'e':['e',''],
    'i':'i',
    'u':'u',# according to wiki is 'ʊ'

    'w':'w',
    'j':'y', # and 'j':'ʒ'

    # Bilabials
    'b':'b',# or 'β'
    'bʷ':'bʷ',
    'm':'m',
    'mʷ':'mʷ',

    # Labiodental
    'f':'f',

    # Alveolar
    'n':'n',

    's':'s',
    'sˤ':'ṣ',
    'z':'z',
    'zˤ':'ẓ',

    't':'t',
    'θ':'t',
    'tˤ':'ṭ',
    't͡s':'ţ',
    'd':'d',
    'dˤ':'ḍ',#had to add manually
    'ð':'d',
    'ðˤ':'ḍ',
    'd͡z':'z̧',

    'l':'l', # or 'ɫ'
    'r':'r', # or 'rˤ'
    'rˤ':'ṛ',# 
    'ɺ':'ř',# between r and l, Rif Berber
    # according to "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # r should be IPA ɾ or r depending on context

    # Post Alveolar
    'ʃ':'c',
    't͡ʃ':'č',
    'ʒ':'j',# and 'y':'j' 
    'd͡ʒ':'dj',
    'd͡ʒ':'ǧ',
    'd͡ʒ':'ǧǧ',

    # Palatal 

    # Velar 		
    'x':'x',# or 'χ' 
    'xʷ':'xʷ',
    'ɣ':'ɣ',# or 'ʁ'
    'ɣʷ':'ɣʷ',

    'g':'g',
    'ɡʷ':'ɡʷ',
    'k':'k',
    'kʷ':'kʷ',

    # Uvular
    'q':'q',# or 'qʷ' or 'ɢ'
    'qʷ':'qʷ',
 
    # Pharyngeal
    'ħ':'ḥ',# CONSENSUS 1,2,3
    'ʕ':'ɛ',

    # Glottal 
    'h':'h',
    
    #clitics
    '-':'-',

    'p':'p',#talpidzat
}


# According to:
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
arabic2ipa = {
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



# Any->IPA == always unambiguous transliteration
# IPA->Any == may have multiple transliterations
def transliterate(text,my_dict):
    trans = [[]] # list will all possible transliterations
    i = 0
    while i< len(text):
        print('----')
        print(text[i])
        if i+1< len(text):
            print(text[i+1])
        # Find out character-matching in our dicts
        k = text[i] 
        ## check for 2-character phones
        if i+1 < len(text) and ''.join([text[i],text[i+1]]) in my_dict:
            k = ''.join([text[i],text[i+1]])
            i += 1

        # Update our transliterations
        if type(my_dict[k]) == type(['a','b']):
            # a fork in transliteration
            new_trans = [] 
            for t in trans:
                for val in my_dict[k]:
                    print(t)
                    print(val)
                    new_trans.append(t + [val])
            trans = new_trans
            print(trans)
        else:
            trans = [t+[my_dict[k]] for t in trans]

        i += 1
    return trans


def main():
    data = utils.load_datasets_zgh()
    cur =  concatenate_datasets([data['common_voice_22_0'],data['moroccan_amazigh_asr']])
    #cur =  data['common_voice_22_0']

    dicts = {}
    dicts['all2ipa'] = {}
    # We want to collect the totality of words that exist, 
    # and do so to create an IPA->IPA pronunciation dictionary
    vocab = {}

    for row in cur:
        row['text']= row['text'].replace('[]-','')
        words = re.sub(r"[?.,!\":;\'\t\*\n]",'', row['text']).lower().split(' ')
        for w in words:
            #print('-'*10)
            #print(w)
            #print(len(w))
            #if re.search(r'[\d%po_v()\[\]{}|σ]', w) or len(w) == 0 or w=='-':
            if bool(re.search(r'(\d+|%|p|o|_|v|\(|\)|σ|\[|\])',w)) or len(w)== 0 or w=='-':
                continue
                #assert(False)
            if 'talpidzat' in w:
                print(f' adding {w}')
                assert(False)
            if 'common_voice_22_0' == row['origin']:
                trans = transliterate(w,tifinagh2ipa)
            else:
                trans = transliterate(w,latin2ipa)
            for t in trans:
                #if 'nekkni' in w:
                #    print(w)
                #    print(t)
                w_trans = ''.join(t)
                vocab[w_trans] = ' '.join(t)
                dicts['all2ipa'][w]= ' '.join(w_trans)


    for w in vocab:
        print('-'*10)
        print(f'w:{w};')
        #for d in [ipa2latin,ipa2tifinagh,ipa2arabic]:
        for d in [ipa2latin,ipa2tifinagh]:
            trans = transliterate(w,d)
            for t in trans:
                dicts['all2ipa'][''.join(t)] = w
            

    print('-' *15)
    print('SAVING DICTS')
    print('-' *15)
    cur_path = utils.get_curr_folder()
    #del vocab['']
    # Write ipa-to-else dicts
    for d in dicts:
        filename = os.path.join(cur_path,'dicts',f'zgh_{d}.dict')
        with open(filename,'w') as f:
            f.write('<unk>\tspn\n')
            for key in dicts[d]:
                f.write(f'{key}\t{dicts[d][key]}\n')
            f.close()
    # Write ipa-to-ipa dict
    with open(os.path.join(cur_path,'dicts','zgh_vocab.dict'),'w') as f:
        f.write('<unk>\tspn\n')
        for w in vocab:
            f.write(f'{w}\t{vocab[w]}\n')
        f.close()


if __name__ == "__main__":
    main()

