#!/usr/bin/env python
from pandas import DataFrame

from ConfigParser import ConfigParser
from sqlalchemy import create_engine

import os.path
import math
import argparse, sys

def parse_runtime(run_str):
    return run_str.split()[1]

def get_runtimes(fname):
    lines = []
    runtimes = {}
    for line in fname:
        lines.append(line)
    assert(len(lines)%2==0)
    for i in xrange(0, len(lines), 2):
        runtimes[lines[i].strip('\n').strip(':')] = \
                float(parse_runtime(lines[i+1]))
    return runtimes

results = []
res_naive = open('fig8_naive.txt', 'r')
results.append((get_runtimes(res_naive), 'cpu', 'naive'))
res_naive.close()

res_auto = open('fig8_auto.txt', 'r')
results.append((get_runtimes(res_auto), 'cpu', 'auto'))
res_auto.close()

res_ref = open('fig8_ref.txt', 'r')
results.append((get_runtimes(res_ref), 'cpu', 'ref'))
res_ref.close()

res_naive_gpu = open('fig8_naive_gpu.txt', 'r')
results.append((get_runtimes(res_naive_gpu), 'gpu', 'naive'))
res_naive_gpu.close()

res_auto_gpu = open('fig8_auto_gpu.txt', 'r')
results.append((get_runtimes(res_auto_gpu), 'gpu', 'auto'))
res_auto_gpu.close()

res_ref_gpu = open('fig8_ref_gpu.txt', 'r')
results.append((get_runtimes(res_ref_gpu), 'gpu', 'ref'))
res_ref_gpu.close()

f = open('fig8.csv', 'w')
f.write('app,runtime,runtime_norm,threads,throughput,throughput_norm,version\n')

best_cpu_time = {}
best_gpu_time = {}

for r in results:
    for app, time in r[0].iteritems():
        if r[1] == 'gpu':
            if app in best_gpu_time:
                if time < best_gpu_time[app]:
                    best_gpu_time[app] = time
            else:
                best_gpu_time[app] = time
        else:
            if app in best_cpu_time:
                if time < best_cpu_time[app]:
                    best_cpu_time[app] = time
            else:
                best_cpu_time[app] = time

#print best_cpu_time
#print best_gpu_time

for r in results:
    for app, time in r[0].iteritems():
        entry = ""
        if r[1] == 'gpu':
            norm_time = time/best_gpu_time[app]
            norm_thrpt = best_gpu_time[app]/time
        else:
            norm_time = time/best_cpu_time[app]
            norm_thrpt = best_cpu_time[app]/time
        entry += app + "," \
                + str(time) + "," \
                + str(norm_time) + "," \
                + r[1] + "," \
                + str(1000/time) + "," \
                + str(norm_thrpt) + "," \
                + r[2] + "\n"
        f.write(entry)
f.close()
