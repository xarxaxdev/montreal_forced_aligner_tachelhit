from datasets import Audio, concatenate_datasets #using huggingface's API
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
- (Phon3) https://en.wikipedia.org/wiki/Tarifit#Phonology
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

"""
tifinagh2ipa = {
    ### VOWELS AND GLIDES
    # According to Phon[1-3] only a,u,i exist in the language (with sometimes an ə) that may or may not be written/pronounced
    #'ⴰ':'æ',# SRC 1
    'ⴰ':'a',# SRC 2,3,4
    # According to all Phon[1-4] /a/ is the right phoneme(the other being a common realization)

    'ⴻ':'ə',# SRC 1  CONFLICT! 
    #'ⴻ':'e',# SRC 2,3,4
    # rarely written in South-mid Morocco (shi/tzm) https://en.wikipedia.org/wiki/Berber_Latin_alphabet#Souss-Berber_local_usage
    # Epenthetic vowel that may exist or not depending on language
    # when written usually represents 'ə'
    # Appears in common_voice_22_0/zgh, unlike the o equivalent

    'ⵉ':'i',# CONSENSUS SRC 1,2,3,4
    'ⵓ':'u', #SRC 2,3,4  CONFLICT!
    #'ⵓ':'w', #SRC 1   CONFLICT!
    'ⵡ':'w',# CONSENSUS 1,2,3,4
    'ⵢ':'j',# SRC 1,3,4
    'ⵋ':'j',# SRC 2
    'ⵌ':'j',# SRC 2
    'ⵘ':'j',# SRC 2
    # Leaving 'ⵋ','ⵌ','ⵘ'; doesn't appear in common_voice_22_0/zgh

    'ⵧ':'o', # SRC 1,2  
    # Leaving 'ⵧ'; doesn't appear in common_voice_22_0/zgh
    

    # Bilabials
    'ⴱ':'b',# CONSENSUS 1,2,3,4
    'ⴱⵯ':'bʷ',# SRC 6; velarization
    'ⴲ':'β',# IRCAM EXTENDED fricative; SRC 1,2
    'ⵒ':'p', # IRCAM EXTENDED; SRC 1,2
    # Leaving 'ⵒ'; doesn't appear in common_voice_22_0/zgh
    # Note that /p/ exists only in rif (according to literature)
    # According to literature can also be pharyngealized  pˤ
    'ⵎ':'m',# CONSENSUS 1,2,3,4
    'ⵎⵯ':'mʷ',# SRC 6;velarization
    # Leaving 'ⵎⵯ'; doesn't appear in common_voice_22_0/zgh

    # Labiodental
    'ⴼ':'f',# CONSENSUS 1,2,3,4
    'ⵠ':'v', # IRCAM EXTENDED
    # Leaving 'ⵠ'; doesn't appear in common_voice_22_0/zgh

    # Dental
    'ⵝ':'θ',# IRCAM EXTENDED fricative; SRC 1,2
    'ⴺ':'ðˤ',# IRCAM EXTENDED fricative; SRC 1,2 
    'ⴸ':'ð',# SRC 2
    # Leaving these 3 ; don't appear in common_voice_22_0/zgh


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
    # Leaving these 3 ; doesn't appear in common_voice_22_0/zgh

    # Palatal 
    'ⵐ':'ny',# SRC 2
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh

    # Velar 		
    'ⴳ':'g',# CONSENSUS 1,2,3,4
    'ⴴ':'g',# SRC 2 (SRC 5 points to this being aproximant);   CONFLICT!
    # Leaving this 1 ; doesn't appear in common_voice_22_0/zgh

    #'ⴴ':'ʝ',# IRCAM EXTENDED fricative;SRC 1 CONFLICT!
    'ⴳⵯ':'ɡʷ',# CONSENSUS 1,2,3,4
    # SRC 6; velarization
    'ⴽ':'k',# CONSENSUS 1,2,3,4

    'ⴾ':'k',# SRC 2
    'ⴿ':'k',# SRC 2 
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh

    'ⴽⵯ':'kʷ',# CONSENSUS 1,2,3,4

    # Uvular
    'ⵅ':'χ',# SRC 1 CONFLICT!
    #'ⵅ':'x',# SRC 2,3,4 CONFLICT!

    'ⵅⵯ':'χʷ',# SRC 2,3,4 CONFLICT!
    'ⵆ':'χ',# SRC 2
    'ⴿ':'χ',# IRCAM EXTENDED fricative; SRC 1 
    # Leaving these 3 ; doesn't appear in common_voice_22_0/zgh

    'ⵖ':'ʁ',# CONSENSUS 1,2,3,4

    'ⵖⵯ':'ʁʷ',# SRC 6; velarization
    'ⵗ':'ʁ',# SRC 2
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh

    'ⵇ':'q',# CONSENSUS 1,2,3,4

    'ⵇⵯ':'qʷ',# SRC 6;velarization
    'ⵈ':'q',# SRC 2
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh

    # Pharyngeal
    'ⵃ':'ħ',# CONSENSUS 1,2,3
    'ⵄ':'ʕ',# SRC 1  CONFLICT! 
    #'ⵄ':'ɛ',# SRC 2,3,4  CONFLICT! #Latin, more common

    # Glottal 
    'ⵀ':'h',# CONSENSUS 1,2,3,4; SRC 2 does not mention if for shi
    
    'ⵂ':'h',# SRC 2
    'ⵁ':'h',# IRCAM EXTENDED; SRC 1,2
    # Leaving these 2 ; doesn't appear in common_voice_22_0/zgh
    
    # multi-letter
    'ⵑ':'ng',# SRC 2
    # Leaving this  1; doesn't appear in common_voice_22_0/zgh

    #clitics
    '-':'-',
}

# Leaving ipa symbols that should not appear commented for simplicity
ipa2tifinagh = {
    ### VOWELS AND GLIDES
    'a':'ⴰ',
    #'ə':['ⴻ',''],# 
    # Though it is epenthethic, it is used in Tifinagh text.
    # appears in moroccan_amazigh_asr
    # and ⴻ in common_voice_22_0/zgh; Should not be discarded for generating word orthography 
    'ə':'ⴻ',
    #'o':'ⵧ',

    'i':'ⵉ',
    'u':'ⵓ',

    'w':'ⵡ',
    'j':'ⵢ',

    # Bilabials
    'b':'ⴱ',
    'bʷ':'ⴱⵯ',
    #'β':'ⴲ',
    'p':'ⵒ',
    'm':'ⵎ',
    'mʷ':'ⵎⵯ',

    # Labiodental
    'f':'ⴼ',
    #'v':'ⵠ',

    # Dental
    #'θ':'ⵝ',
    #'ðˤ':'ⴺ',
    #'ð':'ⴸ',
 
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
    'g':'ⴳ',
    'ɡʷ':'ⴳⵯ',
    'k':'ⴽ',
    'kʷ':'ⴽⵯ',

    # Uvular
    'χ':'ⵅ',
    'χʷ':'ⵅⵯ',
    'ʁ':'ⵖ',
    'ʁʷ':'ⵖⵯ',
    'q':'ⵇ',
    'qʷ':'ⵇⵯ',

    # Pharyngeal
    'ħ':'ⵃ',
    #'ʕ':['ⵄ','ɛ'],# ɛ is latinscript
    'ʕ':'ⵄ',


    # Glottal 
    #'h':['ⵀ','ⵁ'],# trusting common_voice_22_0/zgh as standard
    'h':'ⵀ',

    #clitics
    '-':'-',
    '-':'-',
}


# Used to generate cross-script dictionary for zgh
# https://en.wikipedia.org/wiki/Berber_Latin_alphabet
# and IRCAM Tifinagh~latin~arabic alphabet equivalence   
latin2ipa = { 
    ### VOWELS AND GLIDES
    'a':'a',
    'e':'ə',
    'i':'i',
    'u':'u',
    'w':'w',
    'y':'j', 

    # Bilabials
    'b':'b',
    'bʷ':'bʷ',
    'bᵒ':'bʷ',
    'm':'m',
    'mʷ':'mʷ',
    'mᵒ':'mʷ',
    'p':'p',

    # Labiodental
    'f':'f',

    # Dental
    # Alveolar
    'n':'n', # ŋ exists in Tuareg but in rif/kab/shi/tzm is an assimilation

    's':'s',
    'ṣ':'sˤ',
    'z':'z',
    'ẓ':'zˤ',

    't':'t',
    'ṭ':'tˤ',
    #'ţ':'t͡s',
    'ţ':'ts',
    'd':'d',
    'ḍ':'dˤ',
    #'z̧':'d͡z',
    'z̧':'dz',

    'l':'l', 
    'r':'r', 
    'ṛ':'rˤ',
    #'ř':'ɺ',# between r and l, Rif Berber

    # Post Alveolar
    'c':'ʃ',
    'š':'ʃ', # rif
    'ṣ̌':'ʃˤ', # rif
    #'č':'t͡ʃ',
    'č':'tʃ',#can be simple or double, same pronunciation
    'čč':'tʃ',
    'j':'ʒ',
    #'dj':'d͡ʒ',
    #'ǧ':'d͡ʒ',
    'ǧ':'dʒ',#can be simple or double, same pronunciation
    #'ǧǧ':'d͡ʒ',
    'ǧǧ':'dʒ',

    # rif has some uncommon orthography:
    #'ll':['ll','dʒ'] 
    #'lt':['lt','tʃ']
    # Palatal 

    # Velar 		
    'g':'g',
    'ɡʷ':'ɡʷ',
    'ɡᵒ':'ɡʷ',
    'k':'k',
    'kʷ':'kʷ',
    'kᵒ':'kʷ',

    # Uvular
    'x':'χ',
    'xʷ':'χʷ',
    'xᵒ':'χʷ',
    'ɣ':'ʁ',
    'ɣʷ':'ʁʷ',
    'ɣᵒ':'ʁʷ',
    'q':'q',
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
    'a':'a',
    'ə':'e',
    'i':'i',
    'u':'u',
    'w':'w',
    'j':'y', 

    # Bilabials
    'b':'b',
    'br':'bʷ',
    'p':'p',
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
    #'t͡s':'ţ',
    'ts':'ţ',
    'd':'d',
    'ð':'d',
    'dˤ':'ḍ',#had to add manually
    'ðˤ':'ḍ',
    #'d͡z':'z̧',
    'dz':'z̧',

    'l':'l', # or 'ɫ'
    'r':'r', # or 'rˤ'
    'rˤ':'ṛ',# 
    'ɺ':'ř',# between r and l, Rif Berber
    # according to "Syllables in Tashlhiyt Berber and in Moroccan Arabic"
    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # r should be IPA ɾ or r depending on context

    # Post Alveolar
    'ʃ':'c',
    #'t͡ʃ':'č',
    'tʃ':'č',
    'ʒ':'j',
    #'d͡ʒ':'dj',
    #'d͡ʒ':'ǧ',
    'dʒ':'ǧ',
    #'d͡ʒ':'ǧǧ',
    'dʒ':'ǧǧ',

    # Palatal 

    # Velar 		
    'g':'g',
    'ɡʷ':'ɡʷ',
    'k':'k',
    'kʷ':'kʷ',

    # Uvular
    'χ':'x',
    'χʷ':'xʷ',
    'ʁ':'ɣ',
    'ʁʷ':'ɣʷ',
    'q':'q',# or 'qʷ' or 'ɢ'
    'qʷ':'qʷ',
 
    # Pharyngeal
    'ħ':'ḥ',# CONSENSUS 1,2,3
    'ʕ':'ɛ',

    # Glottal 
    'h':'h',
    
    #clitics
    '-':'-',

    #'p':'p',#talpidzat
}

# Multiple allophones, adding every option from Phon[1-3] so the aligner can choose the best match.
# Ignoring gemminates as of now
# RIF Vowels [Phon3]
# https://www.universiteitleiden.nl/en/research/research-output/humanities/an-introduction-to-tarifiyt-berber
# i-> i,ɪ,ɪˤ
# a-> a,æ,ɑˤ
# u-> u,ʊ,ʊˤ
# vocalized r replacements:
# iɾ -> ɛa
# uɾ -> ɔa
# aɾ -> a/æ
# iɾˤ-> ɪˤɑ
# uɾˤ-> ʊˤa
# aɾˤ-> ɑˤ

# DENTALIZATION (e.g. n -> n̪) 
# shi dentalizes 
# tzm/rif dont 

# LABIALIZATION: contrastive
# rif: kʷ,gʷ
# tzm: xʷ,ɣʷ,qʷ,χʷ,ʁʷ
# shi: kʷ,gʷ,qʷ,χʷ,ʁʷ

# PHARINGEALIZATION: contrastive
# rif: dˤ,zˤ,rˤ,ʃˤ
# tzm: tˤ,dˤ,sˤ,zˤ,lˤ,nˤ,rˤ
# shi: tˤ,dˤ,sˤ,zˤ,lˤ,rˤ

# Generally a huge amount of vowel realizations due to Berber's limited vowel set
# shi does not allow hiats, therefore : i->j/u->w/a->ʕ
# conversely, between consonants: j->i/w->u

# I used Phon[1-3] for each languages' specific realizations
ipa2realization = {
    ### VOWELS AND GLIDES
    'e':['e','ə','ɪ̈',''],# transitional vocoid
    # shi: ''  from wiki "a vowel may be heard"; meaning not always
    # tzm: ɪ̈,ə

    'a':['a','æ','ɐ','ʕ','ɑˤ'],
    # shi: a,æ,ɐ,ʕ

    'i':['i','ɪ','j','ɨ','e','ɪj','ɪˤ','ʝ','ɪʝ'],
    # shi: i,ɪ,j

    'u':['u','ɤ','w','ʊ','o','ʊw','wʊ''ʊˤ'],
    # shi: u,ɤ,w,ʊ
    
    'j':['j','i','ɪ','ʝ'],
    #[j] is [i] between consonants
    
    'w':['w','u','ʊ'],
    #[w] is [u] between consonants


    # Bilabials
    'b':['b','β'], # f from  https://en.wikipedia.org/wiki/Tarifit#Assimilations
    'bʷ':['bʷ','βʷ'],
    'p':['p','pʷ'],# rif only, may be pharyngealized
    'm':'m',
    'mʷ':'mʷ',

    # Labiodental
    'f':'f',

    # Alveolar
    'n':['n','nˤ','n̪','n̪ˤ','ŋ'],# rif has ŋ as assimilation

    's':['s','s̪'],
    'sˤ':['sˤ','s̪ˤ'],
    'z':['z','z̪'],
    'zˤ':['zˤ','z̪ˤ'],

    't':['t','θ','tʰ','t̪'],
    'tˤ':['tˤ','θˤ','t̪ˤ','θˤ'],
    #'t͡s':'t͡s',
    'ts':'ts',
    'd':['d','ð','d̪'],
    'dˤ':['dˤ','ðˤ','d̪ˤ'],
    #'d͡z':'d͡z',
    'dz':'dz',

    'l':['l','ɫ'],
    'r':['r','r̪','ɾ','r̪','rˤ'],
    'rˤ':['ṛ','r̪'],
    'ɺ':'ɺ',# Rif Berber only

    # by FRANÇOIS DELL, MOHAMED ELMEDLAOUI
    # r should be IPA ɾ or r depending on context

    # Post Alveolar
    'ʃ':['ʃ','ç'],
    'ʃˤ':'ʃˤ',#rif only
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

    # Uvular
    #https://en.wikipedia.org/wiki/Central_Atlas_Tamazight#Phonology 
    # "/χʷ/ and /ʁʷ/ rare—native speakers can freely substitute /χ ʁ/"
    'χ':['χ','x'],
    'χʷ':['χʷ','xʷ','χ'],
    'ʁ':['ʁ','ɣ'],# or 'ʁ'
    'ʁʷ':['ʁʷ','ɣʷ','ʁ'],
    'q':'q',# or 'qʷ' or 'ɢ'
    'qʷ':'qʷ',
 
    # Pharyngeal
    'ħ':['ʜ','ħ'],# CONSENSUS 1,2,3
    'ʕ':['ʢ','ʕ'],

    # Glottal 
    'h':['h','ɦ'],
    
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
    data = utils.load_datasets_zgh()
    cur =  concatenate_datasets([data['common_voice_22_0'],data['moroccan_amazigh_asr']])

    dicts = {}
    dicts['all2ipa'] = {}
    # We want to collect the totality of words that exist, 
    # and do so to create an IPA->IPA pronunciation dictionary
    vocab = {}

    for row in cur:
        row['text']= row['text'].replace('[]-','')
        words = re.sub(r"[?.,!\":;\'\t\*\n]",'', row['text']).lower().split(' ')
        for w in words:
            #if bool(re.search(r'(\d+|%|p|o|_|v|\(|\)|σ|\[|\])',w)) or len(w)== 0 or w=='-':
            if bool(re.search(r'(\d+|%|o|_|v|\(|\)|σ|\[|\])',w)) or len(w)== 0 or w=='-':
                continue
                #assert(False)
            if 'common_voice_22_0' == row['origin']:
                trans = transliterate(w,tifinagh2ipa)
            else:
                trans = transliterate(w,latin2ipa)
            for t in trans:
                w_trans = ''.join(t)
                vocab[w_trans] = ' '.join(t)
                dicts['all2ipa'][w]= ' '.join(w_trans)


    for w in vocab:
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

