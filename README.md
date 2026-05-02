# montreal_forced_aligner_tachelhit
Master Thesis Project for Potsdam university: MFA is a method to generate timestamped transcriptions for a language, given only the audio (and using many pre-transcribed text and audio as training). 


# Run to setup the environment 

```
conda create -n aligner -c conda-forge python=3.11 montreal-forced-aligner

conda activate aligner


# All these are run withing the new environment

conda config --set default_threads 14 # change for whatever you like
conda env config vars set OMP_NUM_THREADS=14
conda env config vars set OPENBLAS_NUM_THREADS=14
conda env config vars set MKL_NUM_THREADS=14


pip install datasets==3.6.0
pip install soundfile==0.13.1
pip install torch==2.10.0 
pip install torchaudio==2.10
pip install torchcodec==0.10
```


# Prepare data for the aligner

```
conda activate aligner # Activate your environment

# We need to train individual languages in the order
# kab > zgh > tzm/shi

# Generate unified vocabulary and all pronunciation dictionaries
python kab_build_dicts.py
python zgh_build_dicts.py # we will consider shi/tzm as one
python merge_dicts.py # generate all.dict 

python kab_gen_corpus_acoustic_model.py 
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
mfa_train --phone_groups_path ./phone_groups/kab_ortho.yaml ./corpus/kab ./dicts/kab_all.dict ./output/kab_model.zip --output_directory ./output/kab

# train zgh model based on kabyl model
# mfa adapt [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH  OUTPUT_MODEL_PATH
mfa_adapt ./corpus/zgh ./dicts/zgh_all.dict ./output/kab_model.zip ./output/zgh_model.zip --output_directory ./output/zgh

# attempt align on shi corpus using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/shi ./dicts/zgh_all.dict ./output/zgh_model.zip ./output/shi_model.zip --output_directory ./output/shi

# attempt align on tzm using zgh model
# mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY    
mfa_align ./corpus/tzm ./dicts/zgh_all.dict ./output/zgh_model.zip ./output/tzm_model.zip --output_directory ./output/tzm


```



