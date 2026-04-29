#The Montreal Forced Aligner by default goes through four primary stages of training. 
# 1 of alignment uses monophone models, where each phone is modelled the same regardless of phonological context. 
# 2 triphone models, where context on either side of a phone is taken into account for acoustic models.
# 3 LDA+MLLT to learn a transform of the features that makes each phone's features maximally different. 
# 4 enhances the triphone model by taking into account speaker differences, and calculates a transformation of the mel frequency cepstrum coefficients (MFCC) features for each speaker. 


# Phonological rules: https://montreal-forced-aligner.readthedocs.io/en/v3.3.6/_modules/montreal_forced_aligner/dictionary/multispeaker.html#MultispeakerDictionaryMixin.apply_phonological_rules
# Phone_groups https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/dictionary/mixins.py
# Phonological rules are applied during dictionary setup, whereas phone groups are done in the triphone step (much later)




# Multiple allophones, adding every option from Phon[1-3] so the aligner can choose the best match.
# Ignoring gemminates as of now

# RIF vocalized r replacements: [Phon3] -> IGNORED
# iɾ -> ɛa
# uɾ -> ɔa
# aɾ -> a/æ
# iɾˤ-> ɪˤɑ
# uɾˤ-> ʊˤa
# aɾˤ-> ɑˤ

# DENTALIZATION (e.g. n -> n̪) 
# shi dentalizes 
# tzm/rif dont 

# LABIALIZATION: contrastive (according to Phon2)
# rif: kʷ,gʷ
# tzm: xʷ,ɣʷ,qʷ,χʷ,ʁʷ
# shi: kʷ,gʷ,qʷ,χʷ,ʁʷ

# PHARINGEALIZATION: contrastive (according to Phon2)
# rif: dˤ,zˤ,rˤ,ʃˤ
# tzm: tˤ,dˤ,sˤ,zˤ,lˤ,nˤ,rˤ
# shi: tˤ,dˤ,sˤ,zˤ,lˤ,rˤ

# SPIRANTIZATION: existing but non-contrastive
# rif:  explicitly mentioned https://books.google.de/books?id=ZDxrzQEACAAJ&redir_esc=y
#       both b and β have the same orthography

# Generally a huge amount of vowel realizations due to Berber's limited vowel set
# shi does not allow hiats, therefore : i->j/u->w/a->ʕ
# conversely, between consonants: j->i/w->u
# reference for this is: Syllables in Tashlhiyt Berber and in Moroccan Arabic
# by FRANÇOIS DELL, MOHAMED ELMEDLAOUI

# I used Phon[1-3] for each languages' specific realizations
ipa2realization = {
    ### VOWELS AND GLIDES
    'e':['','ə','ɪ̈'],# transitional vocoid
    # shi: ''  from wiki "a vowel may be heard"; meaning not always
    # tzm: ɪ̈,ə
    # rif: '' 

    'a':['a','æ','ɐ','ʕ','ɑ','ɑˤ'],
    # shi: a,æ,ɐ,ʕ 
    # tzm: æ,ɐ,ɑ
    # rif: a,æ,ɑˤ (ɑˤ in vicinity of pharyngealized cons.)


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



