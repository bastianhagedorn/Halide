from pygg import *
from pandas import DataFrame

import os.path
import math
import argparse, sys

parser = argparse.ArgumentParser(description="Graph the benchmarks")
parser.add_argument('-a', '--apps', const=True, default=False, help='Graph the apps (default if nothing else specified)', action='store_const')
parser.add_argument('-t', '--tests', const=True, default=False, help='Graph the tests', action='store_const')
parser.add_argument('-e', '--extra', type=str, default=[], help='Include specific benchmarks', nargs='+')
parser.add_argument('-x', '--exclude', type=str, default=[], help='Exclude specific benchmarks', nargs='+')
parser.add_argument('-n', '--dont_normalize', const=True, default=False, help='Disable normalization of runtimes', action='store_const')

args = parser.parse_args()
print args

disabled = args.exclude

apps = open('apps.txt').read().split()
tests= open('tests.txt').read().split()

benches = []
if not (args.apps or args.tests or args.extra):
    benches = apps
else:
    if args.apps:
        benches.extend(apps)
    if args.tests:
        benches.extend(tests)
    if args.extra:
        benches.extend(args.extra)

benches = filter(lambda a: a not in disabled, benches)

print 'Testing:\n  ' + '\n  '.join(benches)

res = DataFrame(columns=['app', 'ver', 'threads', 'perf', 'speedup'])

for app in benches:
    try:
        times_ref = []
        times_auto = []
        speed_up = []
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
        auto = DataFrame({
            'app': [app_name]*num_samples,
            'ver': 'auto',
            'threads': [2**i for i in range(num_samples)],
            'runtime': times_auto, # msecs
            'throughput': [1000.0/t for t in times_auto], # runs/sec
            'speedup': speed_up
        })
    
        ref = DataFrame({
            'app': [app_name]*num_samples,
            'ver': 'ref',
            'threads': [2**i for i in range(num_samples)],
            'runtime': times_ref, # msecs
            'throughput': [1000.0/t for t in times_ref], # runs/sec
            'speedup': [1.0]*num_samples
        })
        
        results = auto.append(ref)
        
        if not args.dont_normalize:
            max_runtime = max(auto.append(ref).runtime)
            results.runtime = results.runtime / max_runtime
            max_throughput = max(auto.append(ref).throughput)
            results.throughput = results.throughput / max_throughput
        
        res = res.append(results)

    except IOError,e:
        print 'Skipping missing: '+app

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
