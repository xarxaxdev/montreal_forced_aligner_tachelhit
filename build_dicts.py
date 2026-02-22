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
#https://en.wiktionary.org/wiki/Module:Tfng-translit
#
# technically neo-tifinagh
# Which is the correct writing according to: https://en.wikipedia.org/wiki/Shilha_language#Writing_systems
# From: https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters
tifinagh2ipa_dict = {
    # Vowels and glides
    'ⴰ':'æ',
    'ⴻ':'ə',
    'ⵉ':'i',
    'ⵡ':'w',#VTV= voiced transitional vocoids

    # Bilabials
    'ⴱ':'b',
    'ⵒ':'p', # IRCAM EXTENDED

    # Labiodental
    'ⴼ':'f',
    'ⵠ':'v', # IRCAM EXTENDED

    # Dental
    'ⵝ':'θ',#fricative

    # Alveolar
    'ⵜ':'t',
    'ⵟ':'tˤ',
    'ⴷ':'d', # I worry about aproximant ð
    'ⴹ':'dˤ',

    # Post Alveolar

    # Retro flex 

    # Palatal 

    # Velar 
    
    # Uvular

    # Pharyngeal

    # Glottal 


    'ⴳ':'g',
    'ⴳⵯ':'ɡʷ',
    'ⴽ':'k',
    'ⴽⵯ':'kʷ',
    # according to "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # only voiced h "murmured glottal fricative (‘voiced h’)."
    'ⵀ':'h',
    'ⵃ':'ħ',
    'ⵄ':'ʕ',
    'ⵅ':'χ',
    'ⵇ':'q',
    'ⵊ':'ʒ',
    'ⵍ':'l',
    'ⵎ':'m',
    'ⵏ':'n',
    'ⵓ':'w',
    'ⵔ':'r',
    'ⵕ':'rˤ',
    'ⵖ':'ɣ',
    'ⵙ':'s',
    'ⵚ':'sˤ',
    'ⵛ':'ʃ',
    'ⵢ':'j',
    'ⵣ':'z',
    'ⵥ':'zˤ',
    #IRCAM extended
    'ⴲ':'β',#fricative
    'ⴴ':'ʝ',#fricative
    'ⴺ':'ðˤ',#fricative
    'ⴿ':'x',#fricative
    #'ⵧ':['o','ɔ'], #this is proper, but let me worry about it later
    'ⵧ':'o',
    'ⴵ':'d͡ʒ',
    'ⵁ':'h',
    'ⵞ':'t͡ʃ',
    ' ':' '#spacing exists
}

ipa2tifinagh_dict = {} #the inverse dictionary
for k in tifinagh2ipa_dict:
    ipa2tifinagh_dict[tifinagh2ipa_dict[k]] = k


def tifinagh2ipa(text):
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

# https://huggingface.co/datasets/TutlaytAI/tamazight_asr/viewer/default/train?p=1
latin2ipadict = {
    # Vowels and glides
    'a':'a',
    'i':'i',
    'u':'u',
    'e':'ə',#VTV= voiced transitional vocoids

    # Bilabials
    'b':'b',
    'p':'p',

    # Labiodental
    'f':'f',
    'v':'v',

    # Dental
    't':'θ',#fricative


    # Alveolar
    'ṭ':'t',
    's':'s',
    'tt':'ts',
    'z':'z',
    'ṭṭ':'tˤ',
    'd':'d',# Can also be approximated ð
    'ḍ':'dˤ',
    'ẓ':'z',

    # Post Alveolar
    'c':'ʃ',
    'čč':'',

    # Retro flex 

    # Palatal 
    'y':'j',

    # Velar 
    
    # Uvular
    'x':'χ',

    # Pharyngeal

    # Glottal 



    'ḥ':'hˤ',
    'ʕ':'ʕ',
    'ṛ':'rˤ',
    'ṣ':'sˤ',
    'ẓ':'zˤ',
    #using writing from "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    'š':'ʃ',
    'ž':'ʒ',
    # Other uncommon symbols from IPA seem to already be ok in the datasets I have seen so far  
    # Ambiguity between the book and IPA for symbols that appear in both IPA and latinscript berber:
    # x,ɣ  has the IPA realizations χ and ʁ 
    # Neotifinagh has 'ⵅ':'χ' ,'ⵖ':'ɣ',
    # y should be IPA j 
    'y':'j',
    'w':'w',
    # r should be IPA ɾ or r depending on context
    # ! could mean next consonant is emphasized
    # w and j are glides
}


def latins2ipa(text):
    # TO DO 
    # Consider GEMMINATES: jj 
    orig = []
    trans = []
    i = 0
    while i< len(text):
        if text[i] :
            pass
    return [orig,trans]

# https://www.livelingua.com/peace-corps/Tamazight/Tamazight%20Textbook%202007.pdf
# This needs heavy reviewing
# This is probably a better source: 
# https://en.wikipedia.org/wiki/Help:IPA/Arabic

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

