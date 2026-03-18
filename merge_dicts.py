with open("./dicts/kab_vocab.dict") as f1, open("./dicts/zgh_vocab.dict") as f2:
    combined = f1.read() + "".join(f2.readlines()[1:])#No need replicating line 1

with open("./dicts/all.dict", "w") as out:
    out.write(combined)
