from datasets import Audio, concatenate_datasets  # using huggingface's API
import utils
import re
import os
from pathlib import Path
import sys

"""
# (SRC 1) https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters
# Which is the correct writing according to: https://en.wikipedia.org/wiki/Shilha_language#Writing_systems
# (SRC 2) https://en.wiktionary.org/wiki/Module:Tfng-translit
# (SRC 3) https://www.mdpi.com/2078-2489/16/7/600
# (SRC 4) https://ieeexplore.ieee.org/abstract/document/8284715
# (SRC 5) https://commons.wikimedia.org/wiki/Tifinagh
# (SRC 6)  https://en.wikipedia.org/wiki/Berber_Latin_alphabet (for specific berber variants)
"

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
https://analytics.hplt-project.org/viewer/HPLT-v3-zgh_Tfng.yaml -> Misslabeled
# Wikipedia - Nuclear option; 14k pages however major cleanup needed
https://shi.wikipedia.org/wiki/Tasna_Tamzwarut
https://dumps.wikimedia.org/other/mediawiki_content_current/shiwiki/2026-02-01/xml/bzip2/
https://medium.com/@evan.frank/accessing-and-cleaning-bulk-wikipedia-text-data-bfde3b550474
"""



"""
Neo-Tifinagh: https://en.wikipedia.org/wiki/Tifinagh#Neo-Tifinagh_letters

Many Tifinagh symbols are uncommon/not officially recognized by IRCAM. Roughly speaking the 33 symbols are the ones that should be used for almost all Tamazight; whereas the extended cover language edge cases:
- ⵁ is just an alternative writing for ⵀ
- /o/(ⵧ) is for the Tuareg dialect, it is a variant of /u/(ⵓ).
- /p/(ⵒ) and /v/(ⵠ)  are  intended for foreign words. I will leave them in-code (for completion's sake), but sentences with foreign words will be skipped due to their more volatile pronunciation.
- /β/(ⴲ) /ʝ/(ⴴ) /ð/(ⴸ, from SRC 2) /ðˤ/(ⴺ) /θ/(ⵝ) /x/(ⴿ) are just specific aspirantizations of existing phonemes. These are sometimes used when the writer wants to emphasize some pronunciation in the specific dialect, but not the standard. I will transform them to their specific unaspired consonant(/β/->/b/,/ʝ/->/g/,/ð/->/d/,/ðˤ/->/dˤ/,/θ/->/t/,/x/->/k/)
- /tʃ/(ⵞ) /dʒ/(ⴵ) are equivalent to ⵜⵛ and ⴷⵊ respectively; and are interchangeable. I will handle this as a phonological rule.

Notes: 
- "ʃˁ", "ʒˁ","nˁ"  exist, but is not lexically relevant (happens due to environemnt/in loanwords) and has no explicit orthography.
- "H" has been suggested as the pronunciation in IPA for tachelhit, but a lot more literature (in both tachelhit and other dialects) suggests it to be "ħ". 
- shi is in particular non-spirantizing (https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS1999/papers/p14_0603.pdf) but supposedly some dialects in it do spirantize (https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/tashlhiyt-berber/D5C8F16C425A89314D833DDE0ACF83D4)



"""

std_tif = {
    # UNCOMMON SYMBOLS AND ALTERNATIVE WRITINGS
    "ⵋ": "ⵢ",  # SRC 2
    "ⵌ": "ⵢ",  # SRC 2
    "ⵘ": "ⵢ",  # SRC 2
    "ⵗ": "ⵖ",  # SRC 2
    "ⵈ": "ⵇ",  # SRC 2
    "ⵂ": "ⵀ",  # SRC 2
    "ⵁ": "ⵀ",  # IRCAM EXTENDED; SRC 1,2
    "ⴾ": "ⴽ",  # SRC 2
    "ⵧ": "o",  # SRC 1,2; Touareg-only symbol
    # Leaving these; don't appear in common_voice_22_0/zgh
    #'ⴿ':'ⴽ',# SRC 2  I suspect this is a mixup with SRC 1 and Neo-tifinagh's writing.
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh
    #"ⵆ": "ⵅ",  # SRC 2 # I am uncertain of this symbol, leaving it commented for posterity

    # FOREIGN PHONEMES(OR VERY DIALECTAL
    #"ⵠ": "v",  # IRCAM EXTENDED; SRC 1,2
    #"ⵒ": "p",  # IRCAM EXTENDED; SRC 1,2
    # Leaving 'ⵒ'; uncommon in common_voice_22_0/zgh
    # Note that /p/ exists only in rif (according to literature)
    # According to literature can also be pharyngealized  pˤ

    # SPIRANTIZATIONS
    #'ⴲ':'β',# IRCAM EXTENDED fricative; SRC 1,2
    "ⴲ": "ⴱ",  # b spirantizes to β
    #'ⵝ':'θ',# IRCAM EXTENDED fricative; SRC 1,2
    "ⵝ": "ⵜ",  # t spirantizes to θ
    #'ⴸ':'ð',# SRC 2
    "ⴸ": "ⴷ",  # d aspirantizes to ð
    #'ⴺ':'ðˤ',# IRCAM EXTENDED fricative; SRC 1,2
    "ⴺ": "ⴹ",  # dˤ aspirantizes to ðˤ
    #'ⴴ':'ʝ',# SRC 2 (SRC 5 points to this being g aspirantized);   CONFLICT!
    "ⴴ": "ⴳ",  # g aspirantizes to ʝ
    #'ⴴ':'ʝ',# IRCAM EXTENDED fricative;SRC 1 CONFLICT!
    "ⴿ": "ⴽ",  # IRCAM EXTENDED fricative; SRC 1 ; I suspect this is a case of aspirantization
    # Leaving these; don't appear in common_voice_22_0/zgh

    # MULTI-SYMBOL
    # Palatal
    #'ⵐ':'ny',# SRC 2,5
    #'ⵑ':'ng',# SRC 2,5
    "ⴶ": "ⴷⵊ",  # SRC 2,5
    "ⴵ": "ⴷⵊ",  # SRC 1,2
    "ⵞ": "ⵜⵛ",  # SRC 1,2
    # Leaving these 5 in-code; don't appear in common_voice_22_0/zgh
    # Pondered a lot whether these are worth simplifying. At the end, since 'ⵜⵛ' appears in common voice corpus but not 'ⵞ' it is probably a better standardization. Same applies to 'ⴷⵊ'. 
    # The more basic the representation; the easier our work becomes.
    # The other cover dialects very dissimilar to shi

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
    # Appears in common_voice_22_0/zgh
    "ⵉ": "i",  # CONSENSUS SRC 1,2,3,4
    "ⵓ": "u",  # SRC 2,3,4  CONFLICT!
    #'ⵓ':'w', #SRC 1   CONFLICT! (this is true only between vowels)
    "ⵡ": "w",  # CONSENSUS 1,2,3,4
    "ⵢ": "j",  # SRC 1,3,4
    # Bilabials
    "ⴱ": "b",  # CONSENSUS 1,2,3,4
    "ⴱⵯ": "bʷ",  # SRC 6; velarization
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
    # Velar
    "ⴳ": "g",  # CONSENSUS 1,2,3,4
    "ⴳⵯ": "ɡʷ",  # CONSENSUS 1,2,3,4
    "ⴽ": "k",  # CONSENSUS 1,2,3,4
    "ⴽⵯ": "kʷ",  # CONSENSUS 1,2,3,4
    # Uvular
    # velarization is contrastive, and change depending on dialect.
    "ⵅ": "χ",  # SRC 1,2,3,4 CONFLICT!
    "ⵅⵯ": "χʷ",  # SRC 2,3,4 CONFLICT!
    "ⵖ": "ʁ",  # CONSENSUS 1,2,3,4
    "ⵖⵯ": "ʁʷ",  # SRC 6; velarization
    "ⵇ": "q",  # CONSENSUS 1,2,3,4
    "ⵇⵯ": "qʷ",  # SRC 6;velarization
    # Pharyngeal
    "ⵃ": "ħ",  # CONSENSUS 1,2,3
    "ⵄ": "ʕ",  # SRC 1  CONFLICT!
    #'ⵄ':'ɛ',# SRC 2,3,4  CONFLICT! # This is the latinscript equivalent
    # Glottal
    "ⵀ": "h",  # CONSENSUS 1,2,3,4; SRC 2 does not mention if for shi
    # clitics
    "-": "",
}

non_geminated = "ⴰⴻⵉⵓ-"#aeiu
# Gemminates
keys =list( tif2ipa.keys())
for k in keys:
    if k not in non_geminated:
        if 'ⵯ' in k:
            # https://huggingface.co/datasets/fsicoli/common_voice_22_0/blob/main/transcript/zgh/validated.tsv 
            # many cases of ⴽⴽⵯ; this must be the correct way to write the geminate
            tif2ipa[f'{k[0]}{k}']= f'{tif2ipa[k]}:'
        else:
            tif2ipa[f'{k}{k}']= f'{tif2ipa[k]}:'

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
non_geminated = "aeiučǧz̧ţ-"
# Geminates
keys = list(lat2ipa.keys())
for k in keys:
    if k not in non_geminated:
        lat2ipa[f'{k}{k}']= f'{lat2ipa[k]}:'


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
        if i + 1 < len(text) and "".join([text[i], text[i + 1]]) in my_dict: # match 2-character
            if i+2 == len(text) or text[i+2] not in 'ʷⵯ':
                # unless the 3rd character labializes 2nd
                k = "".join([text[i], text[i + 1]])
                i += 1

        # Update our transliterations
        if type(my_dict[k]) == type(["a", "b"]):
            # a fork in transliteration
            # this handles even ⵜⵓⵜⵜ (tett) well
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
        [data["common_voice_22_0"]]
    )
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
            

            if "common_voice_22_0" == row["origin"]:
                w_std_tif = standardize(w, std_tif)
                w_std_lat = ''.join(transliterate(w_std_tif, tif2lat)[0])
                trans = transliterate(w_std_tif, tif2ipa)
            #else:
            #    w_std_lat = standardize(w, std_lat)
            #    w_std_tif = ''.join(transliterate(w_std_lat, lat2tif)[0])
            #    trans = transliterate(w_std_lat, lat2ipa)


            for t in trans:  # e may be 'ə',''
                pron = " ".join(t).replace('  ',' ')       #e being '' causes double spacing
                if not (w in vocab):
                    vocab[w] = set()
                vocab[w].add(pron)
                if not (w_std_tif in vocab):
                    vocab[w_std_tif] = set()
                vocab[w_std_tif].add(pron)
                if not (w_std_lat in vocab):
                    vocab[w_std_lat] = set()
                vocab[w_std_lat].add(pron)
    #  "Syllables in tashlhiyt berber and in moroccan arabic" p. 46
    # says "the genitive preposition /n/ completely assimilates to the initial segment of the following word"
    # therefore we not hear /n/ before /uinwyrlm/
    # before w,y becomes u,i respectively
    # before u,i becomes u,i respectively and transforms the following u,i to w,y
    # before l,r(pharing or not) become l or r
    #(not with geminates)
    vocab['n'] = {'n','u','i','r','l'}
    vocab['ⵏ'] = {'n','u','i','r','l'}

    # "Syllables in tashlhiyt berber and in moroccan arabic" p.48
    # (R)AD's final consonant: sometimes /ad/ or /rad/ drop (or assimilates)
    # the d at the end of the word
    vocab['ⴰⴷ'] = {'a d','a'}
    vocab['ⵔⴰⴷ'] = {'r a d','r a'}
    vocab['ad'] = {'a d','a'}
    vocab['rad'] = {'r a d','r a'}


    # "Syllables in tashlhiyt berber and in moroccan arabic" p.59
    # Comments on context-dependant realizations of vowels 
    # (near emphatic pronunciations). However I see unclear how
    # syllabization influences this, I find it therefore better 
    # not to implement
    #


    # "Syllables in tashlhiyt berber and in moroccan arabic" p. 94
    # Sonority scale: 
    # /a/ > high vocoids > liquids > nasals > fricatives > stops
    # a > iuyw > rl > mn > sxzʃʒh > tkqbdg
    # r is liquid
    # higher sonorant = nuclei of syllable = epenthesis before it
    # coda cannot have higher sonority than nucleus
    # ambiguous syllabification = VC~CCV= V.C~C.CV = VC~C.CV
    # (where first C and second C merge through gemination)

    #  "Syllables in tashlhiyt berber and in moroccan arabic" p. 139
    # when does a release happen(not only epethetic e) between any
    #  noncontinuant(b,t,d,k,g,q) consonant c1 and c2
    #  book considers n and m noncontinuants as well.
    # vocoids are voiced 
    #   if (articulation_place(c1)  != articulation_place(c2)) :
    #       release() # /tb//kt//mn/ /lm/ MUST release audibly
    #               # may have voicoid
    #   else:
    #       if (sonority(c1) != sonority(c2)):
    #           no_release() # /nd/ /tl/ /bm/ never release
    #                       # guaranteed no vocoid
    #       else:
    #           if (cannot_fusion(c1,c2)):
    #               optional_release() # /tt +t/ /nn + n/
    #                   # may have vocoid
    #           else :#prohibited release
    #               gemination(c1,c2) #/t+t/ /n+n/
    #
    # VTV's (which are voice) cannot happen between voiceless 
    # obstruents. 
    # p.145
    # VTV never near a vowel. Always next to voiced segment
    # After consonant

    # p.154 : a geminate articulation can never be fused to another
    # (closures are not articulations, so ttt ddd kkk  ddd is ok)
    # consistent with the previous, mm + m is fine as /mmm/ (p.158)
    # such as xmm'm
    #

    #p. 160 regressive devoicing exists (across word boundaries too)
    # Kernel(syllable) dependant, so not implementing this.
    #




    print("-" * 15)
    print("SAVING DICTS")
    print("-" * 15)
    cur_path = utils.get_curr_folder()
    # Write all-spelling to ipa dict
    with open(os.path.join(cur_path, "dicts", "zgh_vocab.dict"), "w") as f:
        f.write("<unk>\tspn\n")
        for w in vocab:
            for pron in vocab[w]:
                f.write(f"{w}\t{pron}\n")
        f.close()


if __name__ == "__main__":
    main()
