import glob
import os
import sys
import numpy as np
import struct

def bytes_to_int(bytes: list) -> int:
        result = 0
        for byte in bytes:
            result = (result << 8) + byte
        return result

#Borrowed from: https://gist.github.com/lukasklein/8c474782ed66c7115e10904fecbed86a
def get_flac_duration(filename: str) -> float:
    with open(filename, 'rb') as f:
        if f.read(4) != b'fLaC':
            print(filename)
            raise ValueError('File is not a flac file')
        header = f.read(4)
        while len(header):
            meta = struct.unpack('4B', header)  # 4 unsigned chars
            block_type = meta[0] & 0x7f  # 0111 1111
            size = bytes_to_int(header[1:4])

            if block_type == 0:  # Metadata Streaminfo
                streaminfo_header = f.read(size)
                unpacked = struct.unpack('2H3p3p8B16p', streaminfo_header)
                samplerate = bytes_to_int(unpacked[4:7]) >> 4
                sample_bytes = [(unpacked[7] & 0x0F)] + list(unpacked[8:12])
                total_samples = bytes_to_int(sample_bytes)
                duration = float(total_samples) / samplerate

                return duration
            header = f.read(4)


def collect_data(datadir, traindir, testdir, th=30.0):
  if not os.path.exists(os.path.join(datadir, "train")):
    os.makedirs(os.path.join(datadir, "train"))
  for spk in os.listdir(traindir):
    spkdir = os.path.join(traindir, spk)
    spklength = 0.0
    spkover = False
    for path, _, files in os.walk(spkdir):
      for file in files:
        if file[-3:] == "txt":
          continue
        flacf = os.path.join(path, file)
        os.system("ln -sf " + flacf + " " + os.path.join(datadir, "train"))
        spklength += get_flac_duration(flacf)
        if spklength > th:
          spkover = True
          break
      if spkover:
        break
  
  if not os.path.exists(os.path.join(datadir, "test")):
    os.makedirs(os.path.join(datadir, "test"))
  for spk in os.listdir(testdir):
    spkdir = os.path.join(testdir, spk)
    spklength = 0.0
    spkover = False
    for path, _, files in os.walk(spkdir):
      for file in files:
        if file[-3:] == "txt":
          continue
        flacf = os.path.join(path, file)
        os.system("ln -sf " + flacf + " " + os.path.join(datadir, "test"))
        spklength += get_flac_duration(flacf)
        if spklength > th:
          spkover = True
          break
      if spkover:
        break


def process(datadir):
  trainf = open(os.path.join(datadir,"librispeech_tr.scp"),"w")
  if not os.path.exists(os.path.join(datadir, "wavs")):
    os.makedirs(os.path.join(datadir, "wavs"))
  spk = {}
  spk_count = 0
  wav_to_spk = {}
  for file in os.listdir(os.path.join(datadir, "train")):
    if file.endswith(".flac"):
      wav_file = file[:-4] + "wav"
      os.system("ffmpeg -y -i " + os.path.join(datadir,"train",file) + " " + os.path.join(datadir, "wavs", wav_file))
      curr_spk = file.split("-")[0]
      if curr_spk not in spk:
        spk[curr_spk] = str(spk_count)
        spk_count += 1
      wav_to_spk[wav_file] = spk[curr_spk]
      trainf.write(file[:-4] + "wav" + "\n")
  trainf.close()
  
  testf = open(os.path.join(datadir,"librispeech_te.scp"),"w")
  for file in os.listdir(os.path.join(datadir, "test")):
    if file.endswith(".flac"):
      wav_file = file[:-4] + "wav"
      os.system("ffmpeg -y -i " + os.path.join(datadir,"test",file) + " " + os.path.join(datadir, "wavs", wav_file))
      curr_spk = file.split("-")[0]
      if curr_spk not in spk:
        spk[curr_spk] = str(spk_count)
        spk_count += 1
      wav_to_spk[wav_file] = spk[curr_spk]
      testf.write(file[:-4] + "wav" + "\n")
  testf.close()

  np.save(os.path.join(datadir, "librispeech_dict.npy"), wav_to_spk)



if __name__ == "__main__":
  assert len(sys.argv) == 4
  datadir = sys.argv[1]
  traindir = sys.argv[2]
  testdir = sys.argv[3]
  collect_data(datadir, traindir, testdir)
  process(datadir)
