#!/usr/bin/env python
from pygg import *
from pandas import DataFrame

from ConfigParser import ConfigParser

import os.path
import math
import argparse, sys

parser = argparse.ArgumentParser(description="Graph the benchmarks")
parser.add_argument('-a', '--apps', const=True, default=False, help='Graph the apps (default if nothing else specified)', action='store_const')
parser.add_argument('-c', '--conv', const=True, default=False, help='Conv layer experiment', action='store_const')
parser.add_argument('-t', '--tests', const=True, default=False, help='Graph the tests', action='store_const')
parser.add_argument('-e', '--extra', type=str, default=[], help='Include specific benchmarks', nargs='+')
parser.add_argument('-x', '--exclude', type=str, default=[], help='Exclude specific benchmarks', nargs='+')
parser.add_argument('-n', '--dont_normalize', const=True, default=False, help='Disable normalization of runtimes', action='store_const')

args = parser.parse_args()
print args

disabled = args.exclude

apps = open('apps.txt').read().split()
tests = open('tests.txt').read().split()
conv = open('conv.txt').read().split()

benches = []
if not (args.apps or args.tests or args.extra or args.conv):
    benches = apps
else:
    if args.apps:
        benches.extend(apps)
    if args.tests:
        benches.extend(tests)
    if args.conv:
        benches.extend(conv)
    if args.extra:
        benches.extend(args.extra)

benches = filter(lambda a: a not in disabled, benches)

print 'Testing:\n  ' + '\n  '.join(benches)

res = DataFrame()#columns=['app', 'ver', 'threads', 'perf', 'speedup'])

for app in benches:
    try:
        times_ref = []
        times_auto = []
        speed_up = []
        num_samples = -1
        
        c = ConfigParser()
        for f in 'ref_perf.txt', 'auto_perf.txt', 'naive_perf.txt':
            c.read(os.path.join(app, f))
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
        
        df.insert(0, 'app', app_name)
        df['throughput'] = 1000.0 / df.runtime # runs/sec
        df['speedup'] = 0.0
        
        # this is a little bullshit, but DataFrame slice indexing gets confusing
        ref = df[df.version == 'ref']#.set_index('threads')
        def compute_speedup(row):
            r = ref[ref.threads == row.threads].runtime.iloc[0] #FFFfffuuu
            row.speedup = r / row.runtime
            return row
        df = df.apply(compute_speedup, axis=1)
        
        if not args.dont_normalize:
            df.runtime = df.runtime / max(df.runtime)
            df.throughput = df.throughput / max(df.throughput)
            
        res = res.append(df)

    except IOError,e:
        print 'Skipping missing: '+app

log2_threads = scale_x_continuous(trans='log2_trans()')
log_vertical = scale_y_continuous(trans='log_trans()')

pl = {}
for p in ('speedup','throughput','runtime'):
    pl[p] = ggplot('data', aes(x='threads', y=p))

bars = geom_bar(aes(fill='version'), stat="'identity'", position="'dodge'")
lines = geom_line(aes(colour='version')) #+ log_vertical


def save(name, geom):
    wrap = facet_wrap('~ app')
    ggsave('benchmarks-{0}.png'.format(name),
            pl[name] + geom + log2_threads + wrap,
            data=res,
            prefix='library(scales)')

save('speedup', bars)
save('runtime', lines)
save('throughput', lines)
