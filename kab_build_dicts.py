from datasets import Audio, concatenate_datasets  # using huggingface's API
import utils
import re
import os
from pathlib import Path
import sys

"""
Built from zgh_build_dicts.py

Information cross-referenced from:
https://en.wikipedia.org/wiki/Kabyle_language#Assimilation
A lot of information in this table mentions how some assimilations 
are present in some dialects and some are global. To be usable, it 
would be nice to know which are global and which are not. Or at
least some source.

Given these, I contacted a native speaker of Kabyle from Tizi_Ouzou
and asked him how he would pronounce (or if he had heard other 
pronunciations) of different assimilations. 
I will summarize my findings here:

-   e may be pronounced as an epenthetic vowel with a wide range, 
    or  even not be pronounced


-   ţ, z̧ can be just added (and I will use them for simplicity)
    SRC: https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages

Now, some discrepancies:
Regarding gh as an alternative spelling for ɣ 
- wiki says it is sometimes used
- Native speaker said it is understandable, but informal and rarely-if-ever used
DISMISSED 

Regarding kkw/ggw  as an alternative spelling for kʷ/gʷ  (SRC: https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages)
- wiki says many authors use kkw for  kʷ    
- my native speaker pronounced kkw /k:w/. /kʷ/ was exclusively used when it was word-terminating
IMPLEMENTED (according to speaker), replacement rule 

Spelling /ts/ as tt or ss (SRC https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages)
- wiki  mentions  both tt and ss  spellings for /ts/
- my native speaker did never pronounce ss as /ts/ and pronounced tt always as /ts/ (including cross-word assimilations such as d+t)
IMPLEMENTED (according to speaker) 

Assimilation n+w 
- wiki says : n+w = /bb/ or /pp/
- my native speaker said, all these feel correct: n+w = /nbw/ /npw/ /nw/
- https://en.wikipedia.org/wiki/Kabyle_language#Dialects -> The native speaker never heard of n+w= /ggʷ/
IMPLEMENTED (according to speaker) 

Assimilation n+y:
- wiki says: n+w = /gg/ or /yy/
- native speaker only would pronounce /yy/
IMPLEMENTED ACCORDING TO SPEAKER

Assimilation i+y= /ig/
- wiki does not specify if dialectal.
- My native speaker used it with specific words, but not as a general case (no pattern)
DISMISSED


Several dialectal pronunciations of geminates are listed in 
- https://en.wikipedia.org/wiki/Kabyle_language#Dialects
- https://en.wikipedia.org/wiki/Kabyle_language#Phonology
However, no source is available.These have been confirmed by my native speaker:
- ww -> bbʷ/ww (dialectal-dependent) (could not confirm ggʷ from dialects)
- yy -> gg/yy (dialectal-dependent)
- ɣɣ -> ɣɣ/qq (dialectal-dependent)
IMPLEMENTED (according to speaker) 

Once again a section without sources: https://en.wikipedia.org/wiki/Kabyle_language#Fricatives_vs._stops 
My native speakers pronunciation matched this table.
IMPLEMENTED (according to wiki) 
"""

#https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages
std_lat = {
    "bᵒ": "bʷ",
    "mᵒ": "mʷ",
    "š": "c",  # rif # Arbitrary choice, both c/š are used interchangeably
    "ṣ̌": "c",  # rif
    "ɡᵒ": "ɡʷ",
    "kᵒ": "kʷ",
    "xᵒ": "xʷ",
    "ɣᵒ": "ɣʷ",
    "qᵒ": "qʷ",
    "â": "ɛ",
    "Σ":"ɛ",
    "ԑ":"ɛ",
    "ε":"ɛ",# Greek epsylon (not IPA epsylon)
    "Γ":"ɣ",# Greek gammas
    "γ":"ɣ",
    "tt":"ţ", # Technically an old writing
    "dt":"ţ", # Technically an old writing
    #"ss":"ţ", # Native speaker did not recognize this
    "zz":"z̧",
    "ṛ":"ṛ",
    "ṛ":"ṛ",
    "ẓ":"ẓ",
    "ṣ":"ṣ",
    "ḍ":"ḍ",
    "ṭ":"ṭ",
    "ḥ": "ḥ",  # CONSENSUS 1,2,3
    "ﬀ":"ff",
    "‑":"-",
    "f̣":"f",

}




#https://www.omniglot.com/writing/kabyle.php
lat2ipa = {
    ### VOWELS AND GLIDES
    "a": "a",
    "e": ["ə",""],
    "i": "i",
    "u": "u",
    "w": "w",
    "y": "j",
    # Bilabials
    "b": "β",
    "bʷ": "bʷ",
    "m": "m",
    "mʷ": "mʷ",
    # Labiodental
    "f": "f",
    # Alveolar
    "n": "n",
    "s": "s",
    "ṣ": "sˤ",
    "z": "z",
    "ẓ": "zˤ",
    "t": "θ",
    "ṭ": "tˤ",
    #'ţ':'t͡s',
    "ţ": "ts",
    "d": "ð",
    "ḍ": "dˤ",
    #'z̧':'d͡z',
    "z̧": "dz",
    "l": "l",
    "r": "r",
    "ṛ": "rˤ",
    #'ř':'ɺ',# between r and l, Rif Berber
    # Post Alveolar
    "c": "ʃ",
    "č": "tʃ",
    "j": "ʒ",
    "ǧ": "dʒ",
    # Palatal
    # Velar
    "g": "ʝ", # I have seen examples in common voice that sound
    # /ɣ/
    "ɡʷ": "ɡʷ",
    "k": "ç",
    "kʷ": "kʷ",
    # Uvular
    "x": "χ",
    "xʷ": "χʷ",
    "ɣ": "ʁ",
    "ɣʷ": "ʁʷ",
    "q": "q",
    "qʷ": "qʷ",
    # Pharyngeal
    "ḥ": "ħ",  # CONSENSUS 1,2,3
    "ɛ": "ʕ",
    # Glottal
    "h": "h",
    # clitics
    "-": "",
}


#https://en.wikipedia.org/wiki/Kabyle_language#Fricatives_vs._stops
lat2ipa[f'mb'] = [f'm b']# no fricative b
for c in 'ln':
    lat2ipa[f'{c}d'] = [f'{c} d']
for c in 'brz':
    lat2ipa[f'{c}g'] = [f'{c} g']
lat2ipa[f'ɛg'] = [f'ʕ g']
lat2ipa[f'jg'] = [f'ʒ g']
for c in 'fbslrn':
    lat2ipa[f'{c}k'] = [f'{c} k']
lat2ipa[f'ḥk'] = [f'ħ k']
lat2ipa[f'ɛk'] = [f'ʕ k']
lat2ipa[f'ck'] = [f'ʃ k']
for c in 'lmn':
    lat2ipa[f'{c}t'] = [f'{c} t']

lat2ipa['nw']= ['nw','nbw','npw']
lat2ipa['ny'] = 'y:'

## GEMINATES
lat2ipa['ww'] = ['bʷ:','w:'] 
lat2ipa['yy'] = ['y:','g:']
lat2ipa['ɣɣ'] = ['ʁ:','q:']

"""
From: 
- https://en.wikipedia.org/wiki/Kabyle_language#Fricatives_vs._stops
"note that gemination turns fricatives into stops)."
- not true for tt (becomes affricate /ts/)
- not true for most geminates (checked dataset manually)


"""
lat2ipa[f'bb'] = [f'b:'] # True for b
lat2ipa[f'dd'] = [f'd:'] # True for d
lat2ipa[f'gg'] = [f'g:'] # True for g
lat2ipa[f'kk'] = [f'k:'] # True for g
# normal gemminates (do not change from affricate to stop)
for c in 'mfnszlrqh':
    lat2ipa[f'{c}{c}'] = [f'{c}:'] 
lat2ipa['cc']=f'ʃ:'
lat2ipa['jj']=f'ʒ:'
lat2ipa['xx']=f'χ:'
lat2ipa['ḥḥ']=f'ħ:'
lat2ipa['ṣṣ']=f'sˤ:'
lat2ipa['ẓẓ']=f'zˤ:'
lat2ipa['ṭṭ']=f'tˤ:'
lat2ipa['ḍḍ']=f'dˤ:'
lat2ipa['ţţ']=f'ts:'
lat2ipa['z̧z̧']=f'dz:'
lat2ipa['ṛṛ']=f'rˤ:'
lat2ipa['ḍḍ']=f'dˤ:'
lat2ipa['čč']=f'tʃˤ:'
lat2ipa['ǧǧ']=f'dʒˤ:'
lat2ipa['ḥḥ']=f'ħˤ:'
lat2ipa['ɛɛ']=f'ʕ:'

# Suspicious that in common voice we cannot see emphatic:
# b,m,g,k,x,ɣ,q
 


def transliterate(text, my_dict):
    trans = [[]]  # list will all possible transliterations
    i = 0
    while i < len(text):
        # Find out character-matching in our dicts
        k = text[i]
        ## check for 2-character phones
        if i + 1 < len(text) and "".join([text[i], text[i + 1]]) in my_dict:
            k = "".join([text[i], text[i + 1]])
            i += 1

        # Update our transliterations
        if type(my_dict[k]) == type(["a", "b"]):
            # a fork in transliteration
            new_trans = []
            for t in trans:
                for val in my_dict[k]:
                    new_trans.append(t + [val])
            trans = new_trans
        else:
            trans = [t + [my_dict[k]] for t in trans]

        i += 1
    return trans


def standardize(text, std_dict):
    out = text
    for k in std_dict:
        out = out.replace(k, std_dict[k])
    return out


def main():
    data = utils.load_datasets_kab()
    cur = concatenate_datasets([data["common_voice_22_0"]])
    vocab = {}
    for row in cur:
        row["text"] = row["text"].replace("[]-", "")
        words = re.sub(r"[?.,!\":;\'\t\*\n\“”’‘«»]", "", row["text"]).lower().split(" ")
        for w in words:
            if (
                bool(re.search(r"(\d+|…|é|ğ|ï|ⵒ|ⵠ|%|o|_|v|p|\(|\)|σ|\[|\])", w))
                or len(w) == 0
                or w == "-"
            ):
                continue  # skip ambiguous pronunciation cases

            w_std_lat = standardize(w, std_lat)
            trans = transliterate(w_std_lat, lat2ipa)
            for t in trans:  # e may be 'ə',''
                pron = " ".join(t).replace('  ',' ')       #e being '' causes double spacing
                if not (w in vocab):
                    vocab[w] = set()
                vocab[w].add(pron)
                if not (w_std_lat in vocab):
                    vocab[w_std_lat] = set()
                vocab[w_std_lat].add(pron)

    print("-" * 15)
    print("SAVING DICTS")
    print("-" * 15)
    cur_path = utils.get_curr_folder()
    # Write all-spelling to ipa dict
    with open(os.path.join(cur_path, "dicts", "kab_vocab.dict"), "w") as f:
        f.write("<unk>\tspn\n")
        for w in vocab:
            for pron in vocab[w]:
                f.write(f"{w}\t{pron}\n")
        f.close()


if __name__ == "__main__":
    main()
