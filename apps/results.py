from collections import OrderedDict
import ConfigParser

from sqlalchemy import create_engine
from pandas import DataFrame

import inspect
import os.path
import math
import sys

srcdir = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))

apps      = open(os.path.join(srcdir, 'apps.txt')).read().split()
apps.extend(open(os.path.join(srcdir, 'raceapps.txt')).read().split())
tests     = open(os.path.join(srcdir, 'tests.txt')).read().split()
conv      = open(os.path.join(srcdir, 'conv.txt')).read().split()

def ingest(app, path=''):
    c = ConfigParser.ConfigParser()
    for f in 'ref_perf.txt', 'auto_perf.txt', 'naive_perf.txt', 'sweep_perf.txt':
        c.read(os.path.join(path, app, f))
    df = DataFrame([dict(c.items(s)) for s in c.sections()])

    # coerce types
    for col in df.columns:
        try:
            ints = df[col].astype(int)
            df[col] = ints
        except:
            try:
                floats = df[col].astype(float)
                df[col] = floats
            except:
                pass # keep as string

    # coerce old data names if present
    df = df.rename(columns={'nthreads':'threads'})

    app_name = app.replace('_', ' ').title()

    df.insert(0, 'app', app)
    df.insert(0, 'app_name_pretty', app_name)
    assert(len(df))
    df['throughput'] = 1000.0 / df.runtime # runs/sec
    df['speedup'] = 0.0

    # this is a little bullshit, but DataFrame slice indexing gets confusing
    ref = df[df.version == 'ref']#.set_index('threads')
    def compute_speedup(row):
        r = ref[ref.threads == row.threads].runtime.iloc[0] #FFFfffuuu
        row.speedup = r / row.runtime
        return row
    df = df.apply(compute_speedup, axis=1)

    df['runtime_norm'] = df.runtime / max(df.runtime)
    df['throughput_norm'] = df.throughput / max(df.throughput)

    return df

#
# TODO: work in progress on ingesting ini files with repeated config data
class MultiOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(MultiOrderedDict, self).__setitem__(key, value)

if __name__ == '__main__':
    c = ConfigParser.ConfigParser(dict_type=MultiOrderedDict)
    c.read('/tmp/run.txt')
    #for s in c.sections():
    #    for i in c.options(s):
    #
    #        print i,
    print [dict([(k,v.split('\n')[0]) for k,v in c.items(s)]) for s in c.sections()]
