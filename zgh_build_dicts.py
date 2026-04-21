from datasets import Audio, concatenate_datasets  # using huggingface's API
import utils
import re
import os
from pathlib import Path
import sys

"""
In this script, 2 types of dictionary are created:
1. {language_iso}_all2ipa.dict: creates Tifinagh->IPA and Latinscript-> IPA entries for later transforming our base dataset into usable textgrids. Each IPA word only appears once per writing system(no multiple realizations).

2. {language_iso}_vocab.dict: creates IPA-> IPA entries, allowing for multiple realizations of a word.


Transliteration relevant links:
- (Phon1) https://en.wikipedia.org/wiki/Shilha_language#Phonology
- (Phon2) https://en.wikipedia.org/wiki/Central_Atlas_Tamazight#Phonology
- (Phon3) https://en.wikipedia.org/wiki/Tarifit#Phonology (An Introduction To Tarifiyt Berber)
- (Phon4) https://en.wikipedia.org/wiki/Kabyle_language#Phonology

Note that:
- shi & tzm are similar
- rif & kab are similar
Therefore for {language_iso}_all2ipa.dict I will be prioriziting shi/tzm, then use rif as a tierbreaker, then use kab as a tierbreaker.
{language_iso}_vocab.dict will include all allophones.

"""


"""
CONSIDER TODO
Other word sources:
https://universeofmemory.com/tashelhit-language-resources/
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


"""
# (SRC 1) https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters
# Which is the correct writing according to: https://en.wikipedia.org/wiki/Shilha_language#Writing_systems
# (SRC 2) https://en.wiktionary.org/wiki/Module:Tfng-translit
# (SRC 3) https://www.mdpi.com/2078-2489/16/7/600
# (SRC 4) https://ieeexplore.ieee.org/abstract/document/8284715
# (SRC 5) https://commons.wikimedia.org/wiki/Tifinagh
# (SRC 6)  https://en.wikipedia.org/wiki/Berber_Latin_alphabet (for specific berber variants)
"""

"""
 Many Tifinagh symbols are uncommon/not officially recognized by IRCAM:
- Some are for foreign phonemes (v,o) and I will leave for the sake of completion (but will not be used as training).
- Some are just specific realizations of existing phonemes(e.g. ⴲ is β). I will transform them to their specific realization (so long they are not in the corpus common_voice_22_0/zgh) for the sake of completion.
- In case a symbol is in common_voice_22_0 I will consider it a standard symbol and adjust it to the best fitting phoneme in Phon[1-4]
- A glottal stop ʔ is mentioned in the rif literature. Upon inspection on the sources, it has not dedicated Tifinagh symbol and in latin writing a ' is used. While ' appears in the data, seeing as ʔ is not part tzm,shi or even kab'; we will filter it out.
- mʷ is exclusively mentioned in https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Berber_Latin_alphabet_and_the_Tifinagh_Berber_alphabet ;  probably non-existant in tzm/shi/rif but present in some other Berber variant

"""
""" 
Standardizing several symbols are archaich or extremely rare:
- Some symbols are dialect-exclusive(not in rif/shi/tzm)
- Spirantized symbols exist of consonants, but they are rarely used (and I will be handling spirantization as a replacement rule). 
"""

std_tif = {
    "ⵋ": "ⵢ",  # SRC 2
    "ⵌ": "ⵢ",  # SRC 2
    "ⵘ": "ⵢ",  # SRC 2
    # Leaving 'ⵋ','ⵌ','ⵘ'; doesn't appear in common_voice_22_0/zgh
    #'ⴲ':'β',# IRCAM EXTENDED fricative; SRC 1,2
    "ⴲ": "b",  # b spirantizes to β
    "ⵠ": "v",  # IRCAM EXTENDED; SRC 1,2
    # Leaving 'ⵠ'; doesn't appear in common_voice_22_0/zgh
    #'ⵝ':'θ',# IRCAM EXTENDED fricative; SRC 1,2
    "ⵝ": "ⵜ",  # t spirantizes to θ
    #'ⴸ':'ð',# SRC 2
    "ⴸ": "ⴷ",  # d aspirantizes to ð
    #'ⴺ':'ðˤ',# IRCAM EXTENDED fricative; SRC 1,2
    "ⴺ": "ⴹ",  # dˤ aspirantizes to ðˤ
    # Leaving these 3 ; don't appear in common_voice_22_0/zgh
    "ⴶ": "ⴵ",  # SRC 2
    #'ⴴ':'ʝ',# SRC 2 (SRC 5 points to this being aproximant);   CONFLICT!
    "ⴴ": "ⴳ",  # g aspirantizes to ʝ
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh
    #'ⴴ':'ʝ',# IRCAM EXTENDED fricative;SRC 1 CONFLICT!
    "ⴾ": "ⴽ",  # SRC 2
    #'ⴿ':'ⴽ',# SRC 2  I suspect this is a mixup with SRC 1 and Neo-tifinagh's writing.
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh
    # x ~ χ
    "ⵆ": "ⵅ",  # SRC 2
    "ⴿ": "ⵅ",  # IRCAM EXTENDED fricative; SRC 1 ; I suspect this is a case of aspirantization
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh
    "ⵗ": "ⵖ",  # SRC 2
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh
    "ⵈ": "ⵇ",  # SRC 2
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh
    "ⵂ": "ⵀ",  # SRC 2
    "ⵁ": "ⵀ",  # IRCAM EXTENDED; SRC 1,2
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh
    # Palatal
    #'ⵐ':'ny',# SRC 2
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh
    # multi-letter
    #'ⵑ':'ng',# SRC 2
    # Leaving this  1; doesn't appear in common_voice_22_0/zgh
}


tif2ipa = {
    ### VOWELS AND GLIDES
    # According to Phon[1-3] only a,u,i exist in the language (with sometimes an ə) that may or may not be written/pronounced
    #'ⴰ':'æ',# SRC 1
    "ⴰ": "a",  # SRC 2,3,4
    # According to all Phon[1-4] /a/ is the right phoneme(the other being a common realization)
    #'ⴻ':'ə',# SRC 1  CONFLICT!
    "ⴻ": ["", "ə"],
    #'ⴻ':'e',# SRC 2,3,4
    # rarely written in South-mid Morocco (shi/tzm) https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    # Epenthetic vowel that may exist or not depending on language
    # when written usually represents /ə/
    # Appears in common_voice_22_0/zgh, unlike the /o/ equivalent
    "ⵉ": "i",  # CONSENSUS SRC 1,2,3,4
    "ⵓ": "u",  # SRC 2,3,4  CONFLICT!
    #'ⵓ':'w', #SRC 1   CONFLICT! (this is true only between vowels)
    "ⵡ": "w",  # CONSENSUS 1,2,3,4
    "ⵢ": "j",  # SRC 1,3,4
    "ⵧ": "o",  # SRC 1,2
    # Leaving 'ⵧ'; doesn't appear in common_voice_22_0/zgh
    # Bilabials
    "ⴱ": "b",  # CONSENSUS 1,2,3,4
    "ⴱⵯ": "bʷ",  # SRC 6; velarization
    "ⵒ": "p",  # IRCAM EXTENDED; SRC 1,2
    # Leaving 'ⵒ'; doesn't appear in common_voice_22_0/zgh
    # Note that /p/ exists only in rif (according to literature)
    # According to literature can also be pharyngealized  pˤ
    "ⵎ": "m",  # CONSENSUS 1,2,3,4
    "ⵎⵯ": "mʷ",  # SRC 6;velarization
    # Leaving 'ⵎⵯ'; doesn't appear in common_voice_22_0/zgh
    # Labiodental
    "ⴼ": "f",  # CONSENSUS 1,2,3,4
    # Alveolar
    "ⵏ": "n",  # CONSENSUS 1,2,3,4
    "ⵙ": "s",  # CONSENSUS 1,2,3,4
    "ⵚ": "sˤ",  # CONSENSUS 1,2,3,4
    "ⵣ": "z",  # CONSENSUS 1,2,3,4
    "ⵥ": "zˤ",  # CONSENSUS 1,2,3,4
    "ⵜ": "t",  # CONSENSUS 1,2,3,4
    "ⵟ": "tˤ",  # CONSENSUS 1,2,3,4
    "ⴷ": "d",  # CONSENSUS 1,2,3,4
    "ⴹ": "dˤ",  # CONSENSUS 1,2,3,4
    "ⵍ": "l",  # CONSENSUS 1,2,3,4
    "ⵔ": "r",  # CONSENSUS 1,2,3,4
    "ⵕ": "rˤ",  # CONSENSUS 1,2,3,4
    # Post Alveolar
    "ⵛ": "ʃ",  # CONSENSUS 1,2,3,4
    "ⵊ": "ʒ",  # SRC 1 CONFLICT!
    #'ⵊ':'j',# SRC 2,3,4 CONFLICT!
    "ⴵ": "dʒ",  # SRC 1,2
    "ⵞ": "tʃ",  # SRC 1,2
    # Leaving these 3 ; doesn't appear in common_voice_22_0/zgh
    # Velar
    "ⴳ": "g",  # CONSENSUS 1,2,3,4
    "ⴳⵯ": "ɡʷ",  # CONSENSUS 1,2,3,4
    "ⴽ": "k",  # CONSENSUS 1,2,3,4
    "ⴽⵯ": "kʷ",  # CONSENSUS 1,2,3,4
    # Uvular
    # x ~ χ
    # These 2 phonemes get always written the same way (x and ⵅ),
    # velarization is contrastive, and change depending on dialect.
    "ⵅ": "x",  # SRC 1,2,3,4 CONFLICT!
    "ⵅⵯ": "xʷ",  # SRC 2,3,4 CONFLICT!
    "ⵖ": "ɣ",  # CONSENSUS 1,2,3,4
    "ⵖⵯ": "ɣʷ",  # SRC 6; velarization
    "ⵇ": "q",  # CONSENSUS 1,2,3,4
    "ⵇⵯ": "qʷ",  # SRC 6;velarization
    # Pharyngeal
    "ⵃ": "ħ",  # CONSENSUS 1,2,3
    "ⵄ": "ʕ",  # SRC 1  CONFLICT!
    #'ⵄ':'ɛ',# SRC 2,3,4  CONFLICT! # This is the latinscript equivalent
    # Glottal
    "ⵀ": "h",  # CONSENSUS 1,2,3,4; SRC 2 does not mention if for shi
    # clitics
    "-": "-",
}

std_lat = {
    "bᵒ": "bʷ",
    "mᵒ": "mʷ",
    "š": "c",  # rif # Arbitrary choice, both c/š are used interchangeably
    "ṣ̌": "c",  # rif
    "ɡᵒ": "ɡʷ",
    "kᵒ": "kʷ",
    "xᵒ": "xʷ",
    "qᵒ": "qʷ",
    "â": "ɛ",
}

# Used to generate cross-script dictionary for zgh
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
# and IRCAM Tifinagh~latina alphabet equivalence
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
    "p": "p",
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
    "x": "x",
    "xʷ": "xʷ",
    "ɣ": "ɣ",
    "ɣʷ": "ɣʷ",
    "ɣᵒ": "ɣʷ",
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
    "ⵒ": "p",
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
    "ⵜⵙ": "ţ",
    "ⴷⵣ": "z̧",
    # Post Alveolar
    "ⵛ": "c",
    "ⵞ": "č",  # tsh
    "ⵊ": "j",
    "ⴵ": "ǧ",  # dj
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
    data = utils.load_datasets_zgh()
    cur = concatenate_datasets(
        [data["common_voice_22_0"], data["moroccan_amazigh_asr"]]
    )
    vocab = {}
    for row in cur:
        row["text"] = row["text"].replace("[]-", "")
        words = re.sub(r"[?.,!\":;\'\t\*\n]", "", row["text"]).lower().split(" ")
        for w in words:
            if (
                bool(re.search(r"(\d+|%|o|_|v|\(|\)|σ|\[|\])", w))
                or len(w) == 0
                or w == "-"
            ):
                continue  # skip ambiguous pronunciation cases
            if "common_voice_22_0" == row["origin"]:
                w_std_tif = standardize(w, std_tif)
                w_std_lat = ''.join(transliterate(w_std_tif, tif2lat)[0])
                trans = transliterate(w_std_tif, tif2ipa)
            else:
                w_std_lat = standardize(w, std_lat)
                w_std_tif = ''.join(transliterate(w_std_lat, lat2tif)[0])
                trans = transliterate(w_std_lat, lat2ipa)
            for t in trans:  # e may be 'ə',''
                vocab[w] = " ".join(t)
                vocab[w_std_tif] = " ".join(t)
                vocab[w_std_lat] = " ".join(t)

    print("-" * 15)
    print("SAVING DICTS")
    print("-" * 15)
    cur_path = utils.get_curr_folder()
    # Write all-spelling to ipa dict
    with open(os.path.join(cur_path, "dicts", "zgh_vocab.dict"), "w") as f:
        f.write("<unk>\tspn\n")
        for w in vocab:
            f.write(f"{w}\t{vocab[w]}\n")
        f.close()


if __name__ == "__main__":
    main()
