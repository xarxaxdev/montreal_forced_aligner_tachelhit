from datasets import Audio, concatenate_datasets  # using huggingface's API
import utils
import re
import os
from pathlib import Path
import sys

"""
Based on zgh_build_dicts.py


https://en.wikipedia.org/wiki/Kabyle_language#Assimilation:
- (I see 3-4 listed with no account of when are they dialectal).
- Gemmination turns fricatives to stops? The table makes no sense.
- epenthethic e (is it impredictably properly pronounced?)

https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Kabyle-Berber_local_usages
- ţ, z̧ can be just added.
- kkw may be  kʷ for k,g,d,t,b. How to discern those from gemminates.
- gh may be ɣ(so how is ɣʷ written)
- kͦkͦ may be kʷ

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
    "ε":"ɛ",# Greek epsylon (not IPA epsylon)
    "Γ":"ɣ",# Greek gammas
    "γ":"ɣ",
    "tt":"ţ",
    "ss":"ţ",
    "zz":"z̧",

}




#https://www.omniglot.com/writing/kabyle.php
lat2ipa = {
    ### VOWELS AND GLIDES
    "a": "a",
    "e": ["", "ə"],
    "i": "i",
    "u": "u",
    "w": "w",
    "y": "j",
    # Bilabials
    "b": "b",
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
    "t": "t",
    "ṭ": "tˤ",
    #'ţ':'t͡s',
    "ţ": "ts",
    "d": "d",
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
    # rif has some uncommon orthography:
    #'ll':['ll','dʒ']
    #'lt':['lt','tʃ']
    # Palatal
    # Velar
    "g": "g",
    "ɡʷ": "ɡʷ",
    "k": "k",
    "kʷ": "kʷ",
    # Uvular
    "x": "χʷ",
    "xʷ": "χʷ",
    "ɣ": "ʁ",#SRC 7 disagrees: it suggests /ɣ/
    "ɣʷ": "ʁʷ",
    "q": "q",
    "qʷ": "qʷ",
    # Pharyngeal
    "ḥ": "ħ",  # CONSENSUS 1,2,3
    "ɛ": "ʕ",
    # Glottal
    "h": "h",
    # clitics
    "-": "-",
}


tif2lat = {
    ### VOWELS AND GLIDES
    # According to Phon[1-3] only a,u,i exist in the language (with sometimes an ə) that may or may not be written/pronounced
    "ⴰ": "a",
    "ⴻ": "e",
    "ⵉ": "i",
    "ⵓ": "u",
    "ⵡ": "w",
    "ⵢ": "y",
    "ⵧ": "o",
    # Bilabials
    "ⴱ": "b",
    "ⴱⵯ": "bʷ",
    "ⵎ": "m",
    "ⵎⵯ": "mʷ",
    # Labiodental
    "ⴼ": "f",
    # Dental
    # Alveolar
    "ⵏ": "n",
    "ⵙ": "s",
    "ⵚ": "ṣ",
    "ⵣ": "z",
    "ⵥ": "ẓ",
    "ⵜ": "t",
    "ⵟ": "ṭ",
    "ⴷ": "d",
    "ⴹ": "ḍ",
    "ⵍ": "l",
    "ⵔ": "r",
    "ⵕ": "ṛ",
    "ⵜⵙ": "ţ",#ts
    "ⴷⵣ": "z̧",#dz
    # Post Alveolar
    "ⵛ": "c",
    "ⵜⵛ": "č",  # tsh
    "ⵊ": "j",
    "ⴷⵊ": "ǧ", # dj 
    # Velar
    "ⴳ": "g",
    "ⴳⵯ": "ɡʷ",
    "ⴽ": "k",
    "ⴽⵯ": "kʷ",
    # Uvular
    "ⵅ": "x",
    "ⵅⵯ": "xʷ",
    "ⵖ": "ɣ",
    "ⵖⵯ": "ɣʷ",
    "ⵇ": "q",
    "ⵇⵯ": "qʷ",
    # Pharyngeal
    "ⵃ": "ḥ",  # CONSENSUS 1,2,3
    "ⵄ": "ɛ",  # SRC 1  CONFLICT!
    # Glottal
    "ⵀ": "h",
    # clitics
    "-": "-",
}

lat2tif = {}

for k in tif2lat:
    lat2tif[tif2lat[k]] = k


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
        words = re.sub(r"[?.,!\":;\'\t\*\n]", "", row["text"]).lower().split(" ")
        for w in words:
            if (
                bool(re.search(r"(\d+|ⵒ|ⵠ|%|o|_|v|p|\(|\)|σ|\[|\])", w))
                or len(w) == 0
                or w == "-"
            ):
                continue  # skip ambiguous pronunciation cases
            w_std_lat = standardize(w, std_lat)
            trans = transliterate(w_std_lat, lat2ipa)
            for t in trans:  # e may be 'ə',''
                vocab[w] = " ".join(t)
                vocab[w_std_lat] = " ".join(t)

    print("-" * 15)
    print("SAVING DICTS")
    print("-" * 15)
    cur_path = utils.get_curr_folder()
    # Write all-spelling to ipa dict
    with open(os.path.join(cur_path, "dicts", "kab_vocab.dict"), "w") as f:
        f.write("<unk>\tspn\n")
        for w in vocab:
            f.write(f"{w}\t{vocab[w]}\n")
        f.close()


if __name__ == "__main__":
    main()
