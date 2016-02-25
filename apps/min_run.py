import sys
import glob
import errno
path = 'run*.txt'
files = glob.glob(path)
times = []
for file in files:
    with open(file, 'r') as f:
        for line in f:
            pass
        words = [word.strip() for word in line.split()]
        times.append((file, float(words[-1])))
times.sort(key = lambda x: x[1])
for t in times:
    print t
