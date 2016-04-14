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
res_naive_6 = open('fig7_naive_6.txt', 'r')
results.append((get_runtimes(res_naive_6), '6', 'naive'))
res_naive_6.close()

res_auto_6 = open('fig7_auto_6.txt', 'r')
results.append((get_runtimes(res_auto_6), '6', 'auto'))
res_auto_6.close()

res_ref_6 = open('fig7_ref_6.txt', 'r')
results.append((get_runtimes(res_ref_6), '6', 'ref'))
res_ref_6.close()

res_6config_6 = open('fig7_6config_6.txt', 'r')
results.append((get_runtimes(res_6config_6), '6', '6config'))
res_6config_6.close()

res_at_6 = open('fig7_at_6.txt', 'r')
results.append((get_runtimes(res_at_6), '6', 'at'))
res_at_6.close()

res_polymage_6 = open('fig7_polymage_6.txt', 'r')
results.append((get_runtimes(res_polymage_6), '6', 'polymage'))
res_polymage_6.close()

f = open('fig7.csv', 'w')
f.write('app,runtime,runtime_norm,threads,throughput,throughput_norm,version\n')

best_cpu_time = {}

for r in results:
    for app, time in r[0].iteritems():
        if app in best_cpu_time:
            if time < best_cpu_time[app]:
                best_cpu_time[app] = time
        else:
            best_cpu_time[app] = time

#print best_cpu_time

for r in results:
    for app, time in r[0].iteritems():
        entry = ""
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
