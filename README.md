# montreal_forced_aligner_tachelhit
Master Thesis Project for Potsdam university: MFA is a method to generate timestamped transcriptions for a language, given only the audio (and using many pre-transcribed text and audio as training). 


# Run to setup the environment 

```

conda create -n aligner -c conda-forge python=3.11 montreal-forced-aligner pip -y # as advised by MFA docu
# just enter until its completely created

conda activate aligner


# All these are run within the new environment
export THREADS=14 # change for whatever you like
conda config --set default_threads $THREADS 
conda env config vars set OMP_NUM_THREADS$THREADS
conda env config vars set OPENBLAS_NUM_THREADS=$THREADS
conda env config vars set MKL_NUM_THREADS=$THREADS

conda deactivate && conda activate aligner

pip install datasets==3.6.0
pip install soundfile==0.13.1
pip install torch==2.10.0 
pip install torchaudio==2.10
pip install torchcodec==0.10

```


# Prepare data for the aligner

```
conda activate aligner # Activate your environment

python prepare_paths.py # cleanup previous files; generate needed paths

# We need to train individual languages in the order
# kab > zgh > tzm/shi

# Generate unified vocabulary and all pronunciation dictionaries
python kab_build_dicts.py
python zgh_build_dicts.py # we will consider shi/tzm as one
python merge_dicts.py # generate all.dict 

python kab_gen_corpus_acoustic_model.py # CAREFUL; THIS IS COMPUTATIONALLY HEAVY
python zgh_gen_corpus_acoustic_model.py 
python shi_gen_corpus_acoustic_model.py 
python tzm_gen_corpus_acoustic_model.py 


```

# Train aligners

```

alias mfa_train='mfa train  --clean --single_speaker -j 12 --overwrite'
alias mfa_adapt='mfa adapt --clean --single_speaker -j 12 --overwrite'
alias mfa_align='mfa align --clean --single_speaker -j 12 --overwrite'

# Heavy computational work. Beware.
# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH 
# 1 job = 1 core, I am using 14 here
# --single_speaker is required to parallelize
# and just splits by utterance
# From the MFA documentation
# "Single speaker mode creates multiprocessing splits based on utterances rather than speakers. This mode also disables speaker adaptation equivalent to --uses_speaker_adaptation false."

# train kabyle model
# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH
mfa_train --phone_groups_path ./phone_groups/kab_ortho.yaml --rules_path ./rules/zgh.yaml ./corpus/kab ./dicts/kab_all.dict ./output/kab_model.zip --output_directory ./output/kab

# train zgh model based on kabyl model
# mfa adapt [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH  OUTPUT_MODEL_PATH
mfa_adapt ./corpus/zgh ./dicts/zgh_all.dict ./output/kab_model.zip ./output/zgh_model.zip --output_directory ./output/zgh

# attempt align on shi corpus using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/shi ./dicts/zgh_all.dict ./output/zgh_model.zip ./output/shi

# attempt align on tzm using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/tzm ./dicts/zgh_all.dict ./output/zgh_model.zip ./output/tzm


```


# FULL run for dissertation/reproduceability (training with/without kabyl)

```

conda create -n aligner -c conda-forge python=3.11 montreal-forced-aligner pip -y # as advised by MFA docu
# just enter until its completely created

conda activate aligner


# All these are run within the new environment
export THREADS=14 # change for whatever you like
conda config --set default_threads $THREADS 
conda env config vars set OMP_NUM_THREADS=$THREADS
conda env config vars set OPENBLAS_NUM_THREADS=$THREADS
conda env config vars set MKL_NUM_THREADS=$THREADS

conda deactivate && conda activate aligner

pip install datasets==3.6.0
pip install soundfile==0.13.1
pip install torch==2.10.0 
pip install torchaudio==2.10
pip install torchcodec==0.10
pip install tgt==1.5
pip install chardet==7.4.3


python prepare_paths.py # cleanup previous files; generate needed paths

# We need to train individual languages in the order
# kab > zgh > tzm/shi

# Generate unified vocabulary and all pronunciation dictionaries
python kab_build_dicts.py  
python zgh_build_dicts.py 
python merge_dicts.py 

python kab_gen_corpus_acoustic_model.py # CAREFUL; THIS IS COMPUTATIONALLY HEAVY
python zgh_gen_corpus_acoustic_model.py 
python shi_gen_corpus_acoustic_model.py 
python tzm_gen_corpus_acoustic_model.py 

# Useful alias
alias mfa_train='mfa train  --clean --single_speaker -j 12 --overwrite'
alias mfa_adapt='mfa adapt --clean --single_speaker -j 12 --overwrite'
alias mfa_align='mfa align --clean --single_speaker -j 12 --overwrite'

#####################################################
########### TRAINING WITH KAB AS BASE ###############
#####################################################

# CAREFUL; FIRST STEP IS COMPUTATIONALLY HEAVY

######### TRAINING #########
# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH
mfa_train --phone_groups_path ./phone_groups/kab_ortho.yaml --rules_path ./rules/zgh.yaml ./corpus/kab ./dicts/kab_all.dict ./output_kab/kab_model.zip --output_directory ./output_kab/kab

######### ADJUSTING #########
# train zgh model based on kabyl model
# mfa adapt [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH  OUTPUT_MODEL_PATH
mfa_adapt ./corpus/zgh ./dicts/zgh_all.dict ./output_kab/kab_model.zip ./output_kab/zgh_model.zip --output_directory ./output_kab/zgh

######### ALIGNING #########
# attempt align on shi corpus using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/shi ./dicts/zgh_all.dict ./output_kab/zgh_model.zip ./output_kab/shi

# attempt align on tzm using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/tzm ./dicts/zgh_all.dict ./output_kab/zgh_model.zip ./output_kab/tzm


#####################################################
############TRAINING WITHOUT KAB AS BASE ############
#####################################################


######### TRAINING #########
# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH
mfa_train --phone_groups_path ./phone_groups/kab_ortho.yaml --rules_path ./rules/zgh.yaml ./corpus/zgh ./dicts/zgh_vocab.dict ./output_nokab/zgh_model.zip --output_directory ./output_nokab/zgh

######### ALIGNING #########
# attempt align on shi corpus using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/shi ./dicts/zgh_vocab.dict ./output_nokab/zgh_model.zip ./output_nokab/shi

# attempt align on tzm using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/tzm ./dicts/zgh_vocab.dict ./output_nokab/zgh_model.zip ./output_nokab/tzm


#####################################################
####################### PLOTTING ####################
#####################################################

python evaluate_alignments.py output_kab
python evaluate_alignments.py output_nokab





```


