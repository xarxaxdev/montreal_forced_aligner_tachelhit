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

def latin2ipa(text):
    text = text.lower()
    if text in exceptions:
        return [exceptions[text].split(),exceptions[text].split() ]
    orig = []
    trans = []
    #print(text)
    i = 0
    while i< len(text):
        #print(text[i])
        if i < len(text) -1 and f'{text[i]}{text[i+1]}' in latin2ipadict:
            orig.append(text[i] + text[i+1])
            trans.append(latin2ipadict[text[i]+text[i+1]])
            i+=1
            pass
        else: 
            orig.append(text[i])
            trans.append(latin2ipadict[text[i]])
        i += 1
    return [orig,trans]


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

