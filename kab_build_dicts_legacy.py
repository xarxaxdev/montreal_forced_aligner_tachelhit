from datasets import concatenate_datasets #using huggingface's API
import utils
import re
import os 
from pathlib import Path
 



# https://en.wikipedia.org/wiki/Kabyle_language#Phonology
latin2ipadict = { 
    ### VOWELS AND GLIDES
    # vowel allophones depend on surrounding consonants
    'a':'a',# 'æ' is an allophone
    'e':'ə',
    #https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    'i':'i',# 'ɪ' is an allophone
    'u':'u',# 'ʊ' is an allophone
    'w':'w',
    'y':'j', # and 'j':'ʒ'

    # Bilabials
    'b':'b',# 'bʷ' is an allophone
    'm':'m',

    # Labiodental
    'f':'f',

    # Dental
    't':'θ',

    # Alveolar
    'n':'n',

    's':'s',
    'ṣ':'sˤ',
    'z':'z',
    'ẓ':'zˤ',

    't':'t̪',
    'ṭ':'tˤ',
    'ţ':'t͡s',
    'tt':'t͡s',
    'd':'d̪',# or 'ð' 
    'ḍ':'ðˤ',
    'z':'z',
    'ẓ':'zˤ',
    'zz':'d͡z',

    'l':'l', # 'lˤ' is an allophone
    'r':'r', 
    'ṛ':'rˤ',# 

    # Post Alveolar
    'c':'ʃ', # or 'ʃˤ'
    'č':'t͡ʃ',
    'j':'ʒ',# or 'ʒˤ'
    'dj':'d͡ʒ',
    'ġ':'ɣ',
    'Γ':'ɣ',
    'γ':'ɣ',
    'ǧ':'d͡ʒ',
    'ǧǧ':'d͡ʒ',

    # Palatal 

    # Velar 		
    'ɣ':'ɣ',
    'g':'g',# or 'ʝ','ʝᶣ','ɡʷ'
    'k':'k',# or 'ç','çᶣ','çʷ'

    # Uvular
    'x':'χ',# or 'χʷ' 
    'ɣ':'ʁ',# or 'ʁʷ'
    'q':'q',# or 'qʷ' or 'ɢ'
 
    # Pharyngeal
    'ḥ':'ħ',
    'ɛ':'ʕ',#  Multiple unicode values for this symbol
    'ε':'ʕ',
    'ԑ':'ʕ',
    'Σ':'ʕ',# https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages
    'â':'ʕ',# https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages

    # Glottal 
    'h':'h',

    # clitic:
    '-':'-',
    '‑':'-',
    'ḅ':'bʷ', #not standard, but in dataset


    
}


#https://nantes-universite.hal.science/hal-03682791/document

exceptions = {
    'firefox':'fairfoks',
    'tom':'tom',
    'lpizza':'lpid͡za',
    'purtugal':'purtugal',
    'paris':'paris',
    'york':'jork',
    'boston':'boston',
    'shakespeare':'ʃeikspir',
    'facebook':'feisbuk',
    'jackson':'d͡ʒakson',
    'tokyo':'tokjo'
}

ipa2realization = {
    ### VOWELS AND GLIDES
    'e':['','ə'],
    'a':['a','æ'],
    'i':['i','ɪ','j','ɨ','e','ɪj','ɪˤ'],
    # shi: i,ɪ,j 
    # tzm: i,ɨ,ɪ,e,ɪj (ɪj as absolute end)
    # rif: i,ɪ,ɪˤ (ɪˤ in vicinity of pharyngealized cons.)

    'u':['u','ɤ','w','ʊ','o','ʊw','ʊˤ'],
    # shi: u,ɤ,w,ʊ
    # tzm: u,ʊ,o,ʊw  (ʊw as absolute end)
    # rif: u,ʊ,ʊˤ (ʊˤ in vicinity of pharyngealized cons.)

    'j':['j','i','ʝ'],
    # shi: j,i (i between consonants)
    # tzm: j
    # rif: j,ʝ (ʝ in central rifian is j)

    'w':['w','u'],
    # shi: w,u (u between consonants)
    # tzm: w
    # rif: w


    # Bilabials
    'b':['b','β'], 
    'bʷ':['bʷ','βʷ'],
    # shi: bʷ occurs sporadically in loandwords
    # tzm: b spirantizes to β

    'p':['p','pˤ'],# 
    # shi: p occurs sporadically in loandwords
    # tzm: p explicitly lacking
    # rif: p may be pharyngealized(as a result of spreading)

    'm':'m',
    'mʷ':'mʷ', #non-existant in shi/tzm/rif

    # Labiodental
    'f':'f',

    # Alveolar
    'n':['n','nˤ','n̪','ŋ'],
    # shi: n̪
    # tzm: n,nˤ
    # rif: n,ŋ assimilation exclusively before /w/

    # CONSONANT SET: t,d,s,z,l,r
    # all: have a contrastive pharyngealized equivalent (s-sˤ)
    # shi: t,d,s,z,l,r are dental t̪,d̪,s̪,z̪,l̪,r̪
    # tzm/rif: t,d,s,z,l,r are alveolar

    's':['s','s̪'],
    'sˤ':['sˤ','s̪ˤ'],
    'z':['z','z̪'],
    'zˤ':['zˤ','z̪ˤ'],

    't':['t','θ','tʰ','t̪'],
    'tˤ':['tˤ','θˤ','t̪ˤ','θˤ'],
    # tzm: t spirantizes to θ
    #'t͡s':'t͡s',
    'ts':'ts',
    'd':['d','ð','d̪'],
    'dˤ':['dˤ','ðˤ','d̪ˤ'],
    # tzm: d spirantizes to ð
    #'d͡z':'d͡z',
    'dz':'dz',

    'l':'l',
    'r':['r','r̪','ɾ','ɾ̪'],
    # shi: r̪,ɾ̪
    # tzm: r,ɾ
    # rif: r,ɾ
    'rˤ':['rˤ','r̪ˤ'],
    #'ɺ':'ɺ',# Rif Berber only, written ř
    # "Syllables in Tashlhiyt Berber and in Moroccan Arabic"by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # r should be IPA ɾ or r depending on context

    # Post Alveolar
    'ʃ':['ʃ','ʃˤ'],
    # rif:  ʃ may be pharyngealized ʃˤ(as a result of spreading)
    #       /ç/ has mostly become /ʃ/ in Central Riffian 
    #       Note: I see no explicit writing of /ç/, so I presume it
    #       uses the same orthography c.

    #'t͡ʃ':'t͡ʃ',
    'tʃ':'tʃ',
    'ʒ':'ʒ',
    #'d͡ʒ':'d͡ʒ',
    'dʒ':'dʒ',
    # Palatal 

    # Velar 		
    'g':'g',
    'ɡʷ':'ɡʷ',
    'k':'k',
    'kʷ':'kʷ',
    # ASK AREA EXPERT
    # shi/rif: have labialized g/k
    # tzm: does not have labialized g/k

    # Uvular
    # shi:  clearly contrasts χʷ/χ and ʁʷ/ʁ
    # tzm:  /χʷ/ and /ʁʷ/ rare(native speakers can freely substitute /χʁ/), however labialization is supposedly contrastive
    #       x/ɣ can be a realization of k/g in specific dialect
    # rif: no labialization; x,ɣ orthography should be x,ɣ (which is equivalent to χ,ʁ); ɣ gemminate is ʁ;
    'χ':['χ','x'],
    'χʷ':['χʷ','xʷ'],
    'ʁ':['ʁ','ɣ'],
    'ʁʷ':['ʁʷ','ɣʷ'],

    'q':'q',
    'qʷ':'qʷ',
    # rif: qʷ is not mentioned
 
    # Pharyngeal
    'ħ':['ʜ','ħ'],# CONSENSUS 1,2,3
    # shi: ʜ
    # tzm,rif: ħ
    'ʕ':['ʢ','ʕ'],
    # shi: ʢ
    # tzm,rif: ʕ

    # Glottal 
    'h':['h','ɦ'],
    # shi,rif: ɦ
    # tzm: h
    
    #clitics
    '-':'-',

}



# Any->IPA == always unambiguous transliteration
# IPA->Any == may have multiple transliterations
def transliterate(text,my_dict):
    trans = [[]] # list will all possible transliterations
    i = 0
    while i< len(text):
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
                    new_trans.append(t + [val])
            trans = new_trans
        else:
            trans = [t+[my_dict[k]] for t in trans]

        i += 1
    return trans



def main():
    # Careful, this nukes previously generated daaset
    utils.prepare_project_structure()
    data = utils.load_datasets_kab()
    #for i in data:
        #print(f'i:{i}, data[i]:{data[i]}')
    cur =  concatenate_datasets([data['common_voice_22_0'],data['kabyle_asr']])
    # we want the dictionary in both directions:
    # tifinagh2ipa: to transcribe our datasets
    # ipa2tifinagh: to take note of homophones

    dicts = {}
    dicts['latin2ipa'] = {}
    dicts['ipa2latin'] = {}

    vocab = {}# a set is more fitting, but lists do not have a builtin way to get hashed for sets.
    i_row=0
    #TODO:
    # Make this parallel (not high priority, still takes under a min)
    for row in cur:
        #print(f'row={i_row}' )
        i_row+=1
        #if i_row < 20000:
            #pass
            #continue
        text = row['text'].lower()
        words = re.sub(r"[?.,!\":«»;\'\t\*]",'', text).split(' ')
        for w in words:
            if bool(re.search(r'(\d+|%|p|o|_|v|\(|\)|σ)',w)):
                continue
            orig,trans = latin2ipa(w)
            w_trans = ''.join(trans)
            # Standard Pronunciation dictionary format
            vocab[w_trans] = ' '.join(trans)
            dicts['latin2ipa'][w]= ' '.join(w_trans)
            # We keep this as a safety check, so the format is designed for that
            if not w_trans in dicts['ipa2latin']:
                dicts['ipa2latin'][w_trans] = [w]
            # homophone check
            elif not (w in dicts['ipa2latin'][w_trans]):
                dicts['ipa2latin'][w_trans].append(w)

    print('-' *15)
    print('SAVING DICTS')
    print('-' *15)
    cur_path = utils.get_curr_folder()
    #del vocab['']
    # Write ipa-to-else dicts
    for d in dicts:
        filename = os.path.join(cur_path,'dicts',f'kab_{d}.dict')
        with open(filename,'w') as f:
            f.write('<unk>\tspn\n')
            for key in dicts[d]:
                f.write(f'{key}\t{dicts[d][key]}\n')
            f.close()
    # Write ipa-to-ipa dict
    with open(os.path.join(cur_path,'dicts','kab_vocab.dict'),'w') as f:
        f.write('<unk>\tspn\n')
        for w in vocab:
            f.write(f'{w}\t{vocab[w]}\n')
        f.close()


if __name__ == "__main__":
    main()

