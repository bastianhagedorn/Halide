#!/usr/bin/env python
from pygg import *
from pandas import DataFrame

import os.path
import math
import argparse, sys

res = DataFrame(columns=['app', 'ver', 'par', 'vec', 'balance', 'memsize',
                         'threads', 'runtime', 'speedup'])

for app in os.listdir('apps'):
    runs = [r for r in os.listdir(os.path.join('apps', app)) \
            if r.startswith('run.')]
    if not runs:
        continue
    for run in runs:
        params = run.split('.')[1:-1]
        par,vec,balance,memsize,threads = params
        runtime = open(os.path.join('apps',app,run)).readlines()[-1]
        runtime = runtime.split(':')[-1].strip()
        res = res.append({
                'app':      app,
                'ver':      'auto',
                'par':      int(par),
                'vec':      int(vec),
                'balance':  int(balance),
                'memsize':  int(memsize),
                'threads':  int(threads),
                'runtime':  float(runtime),
                'speedup':  0},
            ignore_index=True)

for app in res.app.unique():
    runs = res[res.app == app].copy()
    runs.runtime /= max(runs.runtime) # normalize runtime
    res.loc[res.app == app] = runs
    
pl = ggplot('data') + facet_wrap('~ app')
#pl = pl + geom_line(aes(x='seq_along(runtime)', y='runtime'))
hist = geom_histogram(aes('runtime'), bins=20)

ggsave('schmoo.png'.format(app), pl+hist, data=res)

sys.exit()

log2_threads = scale_x_continuous(trans='log2_trans()')
log_vertical = scale_y_continuous(trans='log_trans()')

pl = {}
for p in ('speedup','throughput','runtime'):
    pl[p] = ggplot('data', aes(x='threads', y=p))

bars = geom_bar(aes(fill='ver'), stat="'identity'", position="'dodge'")
lines = geom_line(aes(colour='ver')) #+ log_vertical


def save(name, geom):
    wrap = facet_wrap('~ app')
    ggsave('benchmarks-{0}.png'.format(name),
            pl[name] + geom + log2_threads + wrap,
            data=res,
            prefix='library(scales)')

save('speedup', bars)
save('runtime', lines)
save('throughput', lines)
