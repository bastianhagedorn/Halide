#!/usr/bin/env python
from pandas import DataFrame

from ConfigParser import ConfigParser
from sqlalchemy import create_engine

import os.path
import math
import argparse, sys

from results import *

parser = argparse.ArgumentParser(description="Graph the benchmarks")
parser.add_argument('-a', '--apps', const=True, default=False, help='Graph the apps (default if nothing else specified)', action='store_const')
parser.add_argument('-c', '--conv', const=True, default=False, help='Conv layer experiment', action='store_const')
parser.add_argument('-t', '--tests', const=True, default=False, help='Graph the tests', action='store_const')
parser.add_argument('-e', '--extra', type=str, default=[], help='Include specific benchmarks', nargs='+')
parser.add_argument('-x', '--exclude', type=str, default=[], help='Exclude specific benchmarks', nargs='+')
# TODO: arg to set outfile

args = parser.parse_args()
print args

disabled = args.exclude

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

print 'Loading:\n  ' + '\n  '.join(benches)

res = DataFrame()

for app in benches:
    try:
        res = res.append(ingest(app))
    except: # IOError,e:
        print 'Skipping missing or malformed: '+app
    #except:
    #    print '\n\nFailed on',app,'\n'
    #    raise

db = create_engine('sqlite:///benchmarks.db')
res.to_sql('benchmarks', db)
res.to_csv('benchmarks.csv')
