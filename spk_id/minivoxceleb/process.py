import os
import numpy as np

spkid = {}
utt2spk = {}
count = 0

for file in os.listdir("train"):
    spk = file[:4]
    if spk in spkid.keys():
        utt2spk[file] = spkid[spk]
    else:
        spkid[spk] = count
        utt2spk[file] = count
        count += 1

for file in os.listdir("test"):
    spk = file[:4]
    if spk in spkid.keys():
        utt2spk[file] = spkid[spk]
    else:
        spkid[spk] = count
        utt2spk[file] = count
        count += 1

print(spkid)
np.save("utt2spk.npy", utt2spk)

