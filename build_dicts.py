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
"""




# REVISE
#https://en.wiktionary.org/wiki/Module:Tfng-translit
#
# technically neo-tifinagh
# Which is the correct writing according to: https://en.wikipedia.org/wiki/Shilha_language#Writing_systems
# From: https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters
tifinagh2ipa_dict = {
    'ⴰ':'æ',
    'ⴱ':'b',
    'ⴳ':'g',
    'ⴳⵯ':'ɡʷ',
    'ⴷ':'d',
    'ⴹ':'dˤ',
    'ⴻ':'ə',
    'ⴼ':'f',
    'ⴽ':'k',
    'ⴽⵯ':'kʷ',
    'ⵀ':'h',
    'ⵃ':'ħ',
    'ⵄ':'ʕ',
    'ⵅ':'χ',
    'ⵇ':'q',
    'ⵉ':'i',
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
    'ⵜ':'t',
    'ⵟ':'tˤ',
    'ⵡ':'w',
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
    'ⵒ':'p',
    'ⵝ':'θ',#fricative
    'ⵠ':'v',
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

# https://en.wikivoyage.org/wiki/Tashelhit_phrasebook#Pronunciation_guide
# Letters not in here are directly transcribed to IPA
latin2ipadict = {
    'ḍ':'dˤ',
    'ḥ':'hˤ',
    'ʕ': 'ʕ',
    'ṛ':'rˤ',
    'ṣ':'sˤ',
    'ṭ':'tˤ',
    'ẓ':'zˤ',
    'e':'ə',
    #using writing from "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    'š':'ʃ',
    'ž':'ʒ',
    # Other uncommon symbols from IPA seem to already be ok in the datasets I have seen so far  
    # Ambiguity between the book and IPA for symbols that appear in both IPA and latinscript berber:
    # x,ɣ Should be IPA χ and ʁ 
    # y should be IPA j 
    # r should be IPA ɾ or r depending on context
    # ! could mean next consonant is emphasized
}


def latins2ipa(text):
    # TO DO 
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
        #print('-' *15)
        #print(row['text'])
        words = re.sub(r"[?.,!\":;\'\t]",'', row['text']).split(' ')
        #print(words)
        for w in words:
            #print('-' *15)
            #print(w)
            #print('-' *15)
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

