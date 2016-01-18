#!/usr/bin/env python
from pygg import *
import pandas
from sqlalchemy import create_engine

db = create_engine('sqlite:///benchmarks.db')
res = pandas.read_sql('benchmarks', db)

log2_threads = scale_x_continuous(trans='log2_trans()')
log_vertical = scale_y_continuous(trans='log_trans()')

pl = {}
for p in ('speedup','throughput', 'throughput_norm','runtime'):
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
# save('runtime', lines)
save('throughput_norm', lines)
