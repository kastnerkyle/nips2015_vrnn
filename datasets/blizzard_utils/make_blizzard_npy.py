from scipy.io import wavfile
import cPickle
import os
data_dir = '/Tmp/kastner/unseg/wavn'
list_len = 200
i = 0
l = []
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
for n, f in enumerate(files):
    sr, d = wavfile.read(f)
    l.append(d)
    if len(l) >= list_len:
        print("Dumping at file %i of %i" % (n, len(files)))
        cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
        i += 1
        l = []
#dump last chunk
cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
