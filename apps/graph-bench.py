from pygg import *
from pandas import DataFrame

import os.path
import math

disabled_apps = ['local_laplacian']
apps = open('apps.txt').read().split()
apps+= open('tests.txt').read().split()
apps = filter(lambda a: a not in disabled_apps, apps)

times_ref = []
times_auto = []
speed_up = []
res = DataFrame(columns=['app', 'ver', 'threads', 'perf', 'speedup'])

for app in apps:
    
    num_samples = -1
    with open(os.path.join(app, "ref_perf.txt")) as f:
        times_ref = [ float(l) for l in f ]
        num_samples = len(times_ref)
    
    with open(os.path.join(app, "auto_perf.txt")) as f:
        times_auto = [ float(l) for l in f ]
        assert(num_samples == len(times_auto))
    
    speed_up = [ ref/auto for (ref, auto) \
                           in zip( times_ref, times_auto ) ]
    
    app_name = app.replace('_', ' ').title()
    res = res.append(DataFrame({
        'app': [app_name]*num_samples,
        'ver': 'auto',
        'threads': [2**i for i in range(num_samples)],
        'runtime': times_auto, # msecs
        'throughput': [1000.0/t for t in times_auto], # runs/sec
        'speedup': speed_up
    }))
    
    res = res.append(DataFrame({
        'app': [app_name]*num_samples,
        'ver': 'ref',
        'threads': [2**i for i in range(num_samples)],
        'runtime': times_ref, # msecs
        'throughput': [1000.0/t for t in times_ref], # runs/sec
        'speedup': [1.0]*num_samples
    }))

log2_threads = scale_x_continuous(trans='log2_trans()')

pl = {}
for p in ('speedup','throughput','runtime'):
    pl[p] = ggplot('data', aes(x='threads', y=p))

bars = geom_bar(aes(fill='ver'), stat="'identity'", position="'dodge'")
lines = geom_line(aes(colour='ver'))

def save(name, geom):
    wrap = facet_wrap('~ app')
    ggsave('benchmarks-{0}.png'.format(name),
            pl[name] + geom + log2_threads + wrap,
            data=res,
            prefix='library(scales)')

save('speedup', bars)
save('runtime', lines)
save('throughput', lines)
