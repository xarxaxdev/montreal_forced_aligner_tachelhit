# https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html#dictionaries-with-pronunciation-probability
# Using probability to keep zgh words non-used during training on kab, and to not use kab words when adjusting the model to zgh

def add_prob(line, x):
    return line.replace('\t',f'\t{x}\t',1)#replace only 1st


with open("./dicts/kab_vocab.dict") as f1, open("./dicts/zgh_vocab.dict") as f2:
    kab = f1.readlines()[1:] 
    zgh = f2.readlines()[1:]
     
    combined = f1.read() + "".join(f2.readlines()[1:])#No need replicating line 1

with open("./dicts/kab_all.dict", "w") as out:
    combined = [add_prob(l,1.0) for l in kab] + [add_prob(l,0.01) for l in zgh]
    out.write(''.join(combined))

with open("./dicts/zgh_all.dict", "w") as out:
    combined = [add_prob(l,0.01) for l in kab] + [add_prob(l,1.0) for l in zgh]
    out.write(''.join(combined))
