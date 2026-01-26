# Check dataset has proper MFA format

# Create(if needed) and activate environment
conda activate aligner

### MFA STEPS

python gen_corpus_acoustic_model.py 
# mfa validate DICTIONARY_PATH CORPUS_DIRECTORY 
mfa validate ./corpus ./dicts/arabig_ipa.dict

#mfa train CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH
