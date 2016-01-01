# from ggplot import *
# import pandas as pd

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os.path
import math

times_ref = {}
times_auto = {}
speed_up = {}
p = 1

for app in ["blur", "hist", "unsharp", "harris", "local_laplacian", "interpolate", "bilateral_grid", \
            "camera_pipe", "conv_layer", "mat_mul", "vgg", "cost_function_test", "overlap_test", \
            "split_test", "tile_vs_inline_test", "data_dependent_test",\
            "parallel_test", "large_window_test"] :

        with open(os.path.join(app, "ref_perf.txt")) as f:
            times_ref[app] = [ float(l) for l in f ]

        with open(os.path.join(app, "auto_perf.txt")) as f:
            times_auto[app] = [ float(l) for l in f ]

        speed_up[app] = [ref/auto for (ref,auto) in zip(times_ref[app], times_auto[app])]

        ax = plt.subplot(5, 4, p)
        ax.bar([1, 2, 3, 4], speed_up[app], alpha=0.4, align='center')
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speed Up")
        ax.set_ylim([0, math.ceil(max(speed_up[app]))])
        ax.set_xticks([1, 2, 3 , 4])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(app.replace('_',' ').title())
        p = p+1
        print speed_up[app]

fig = plt.gcf()
fig.set_size_inches(20,20)
plt.tight_layout(pad=3.0)

# plt.show()
plt.savefig('benchmarks.png', dpi=150)
