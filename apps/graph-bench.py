from pygg import *
from pandas import DataFrame

import os.path
import math

apps = ["blur", "hist", "unsharp", "harris", "local_laplacian",         \
        "interpolate", "bilateral_grid", "camera_pipe", "conv_layer",   \
        "mat_mul", "cost_function_test", "overlap_test", "split_test",  \
        "tile_vs_inline_test", "data_dependent_test", "parallel_test",  \
        "vgg", "large_window_test" ]

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
        'threads': range(1, num_samples+1),
        'perf': times_auto,
        'speedup': speed_up
    }))
    
    res = res.append(DataFrame({
        'app': [app_name]*num_samples,
        'ver': 'ref',
        'threads': range(1, num_samples+1),
        'perf': times_ref,
        'speedup': [1.0]*num_samples
    }))

pl = ggplot('data', aes(x='threads', y='speedup'))
geom = geom_bar(aes(fill='ver'), stat="'identity'", position="'dodge'")
geom = geom + facet_wrap('~ app')
ggsave('benchmarks.png', pl + geom, data=res)
