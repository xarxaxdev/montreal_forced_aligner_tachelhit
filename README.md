# montreal_forced_aligner_tachelhit
Master Thesis Project for Potsdam university: MFA is a method to generate timestamped transcriptions for a language, given only the audio (and using many pre-transcribed text and audio as training). 


# Run to setup the environment 

```
conda create -n aligner -c conda-forge python=3.11 montreal-forced-aligner

conda activate aligner

pip install datasets==3.6.0
pip install soundfile==0.13.1

```


# Run to run the aligner

```
conda activate aligner
# Generate TextGrid/wav files 
python gen_corpus_acoustic_model.py 
# mfa validate DICTIONARY_PATH CORPUS_DIRECTORY 
mfa validate ./corpus ./dicts/arabic_ipa.dict
# mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH 
mfa train --clean ./corpus ./dicts/arabic_ipa.dict ./output/model.zip ./output/corpus_aligned  --use_mllt false
```
