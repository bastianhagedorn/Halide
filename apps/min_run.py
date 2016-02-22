import sys
import glob
import errno
path = 'run*.txt'
files = glob.glob(path)
min_time = float("inf")
min_config = ''
for file in files:
    with open(file, 'r') as f:
        for line in f:
            pass
        words = [word.strip() for word in line.split()]
        if float(words[-1]) < min_time:
            min_time = float(words[-1])
            min_config = file
print min_config, min_time
