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


# Run to run the aligner

```
conda activate aligner

# Generate unified vocabulary and all pronunciation dictionaries
python gen_corpus_acoustic_model.py 

# Generate TextGrid/wav files 
python gen_corpus_acoustic_model.py 

# mfa validate DICTIONARY_PATH CORPUS_DIRECTORY 
mfa validate ./corpus ./dicts/vocab.dict

# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH 
# 1 job = 1 core, I am using 14 here
# --single_speaker is required to parallelize
# and just splits by utterance
# From the MFA documentation
# "Single speaker mode creates multiprocessing splits based on utterances rather than speakers. This mode also disables speaker adaptation equivalent to --uses_speaker_adaptation false."

mfa train --clean --single_speaker  -j 12 ./corpus ./dicts/vocab.dict ./output/model.zip --output_directory ./output/corpus_aligned
# --phone_groups_path phone_groups/IPA_only.yaml
# --phone_groups_path

# --rules_path

```

