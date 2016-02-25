#!/usr/bin/env python
from pygg import *
import pandas
from sqlalchemy import create_engine
from tempfile import mkstemp

import sys, os

srcdir=os.path.expanduser('~/proj/ravi-halide-autoschedule/latedays/bench-numactl-2016-01-16')
if len(sys.argv) > 1:
    srcdir=sys.argv[1]

dbfname='benchmarks.db'
if len(sys.argv) > 2:
    dbfname=sys.argv[2]

dbfile = 'sqlite:///'+os.path.join(srcdir, dbfname)
db = create_engine(dbfile)
res = pandas.read_sql('benchmarks', db)

"""
t = theme(axis.line=element_blank(),
           axis.text.x=element_blank(),
           axis.text.y=element_blank(),
           axis.ticks=element_blank(),
           axis.title.x=element_blank(),
           axis.title.y=element_blank(),
           legend.position="none",
           panel.background=element_blank(),
           panel.border=element_blank(),
           panel.grid.major=element_blank(),
           panel.grid.minor=element_blank(),
           plot.background=element_blank())
"""

prolog = """
library(ggplot2)
require(grid)

data = read.csv('{csvfile}',sep=',')
data$version <- factor(data$version, levels=c('naive','ref','auto', 'sweep'))
data$threads <- factor(data$threads)
data$app <- factor(data$app)

t = theme(
          axis.line=element_blank(),
          axis.text.x=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks=element_blank(),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),

          panel.background=element_blank(),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.border=element_blank(),

          panel.margin=unit(0,'pt'),
          plot.margin=unit(c(0,0,-2,-2),'pt'),

          plot.title = element_text(size=9),

          plot.background=element_blank(),

          legend.position="none"
          )
"""
#          panel.background=element_rect(fill='grey97'),
#          panel.grid.major=element_line(size=0.25),
#          panel.border=element_rect(color='grey90', fill=NA, size=0.5),

printable_name = {
    'blur': 'BLUR',
    'unsharp': 'UNSHARP',
    'harris': 'HARRIS',
    'camera_pipe': 'CAMERA',
    'non_local_means': 'NLMEANS',
    'max_filter': 'MAXFILTER',
    'interpolate': 'MSCALE_INTERP',
    'local_laplacian': 'LOCAL_LAPLACIAN',
    'lens_blur': 'LENS_BLUR',
    'bilateral_grid': 'BILATERAL',
    'hist': 'HIST_EQ',
    'conv_layer': 'CONVLAYER',
    'vgg': 'VGG',
    'mat_mul': 'MATMUL'
}

def plot(app):
    pl = ggplot("subset(data, (data$app == '{0}') & (data$threads == 1 | data$threads == 4))".format(app),
                aes(x='threads', y='throughput_norm')) + ylim(0,1) + labs(x='NULL',y='NULL')+ guides(fill='FALSE')
    pl+= geom_bar(aes(fill='version'), width='0.5', stat="'identity'", position="position_dodge(width=0.55)")
    pl+= scale_fill_manual('values=c("#F2BB57","#456B92","#E54C00", "#00E54C")')
    #pl+= ggtitle("'{0}'".format(printable_name[app]))

    return str(pl)
    # app_name_norm = app.replace(' ', '_').lower()
    # fname = 'fig1-{0}.png'.format(app_name_norm)

    # ggsave('fig1-{0}.png'.format(app_name_norm),
    #         pl,
    #         #data=res[(res.app == app) & ((res.threads == 1) | (res.threads == 4))],
    #         prefix="""
    #             data = subset(read.csv('benchmarks.csv',sep=','),  (threads == 1 | threads == 4))
    #             data$version <- factor(data$version, levels=c('naive','auto','ref'))
    #             data$threads <- factor(data$threads)
    #         """.format(app))
    sys.exit()


apps = res.app.unique()
for app in apps:
    print '\n\n\n===== {0} ====='.format(app)
    app_name_norm = app.replace(' ', '_').lower()
    fname = 'fig1-{0}.pdf'.format(app_name_norm)

    # select
    reldata = res[((res.threads == 4) | (res.threads == 1)) & (res.app == app)]

    #re-normalize
    reldata.throughput_norm = reldata.throughput_norm / max(reldata.throughput_norm)

    assert(max(reldata.throughput_norm) == 1.0)

    (csvfp,csvfile) = mkstemp(suffix='.csv')
    reldata.to_csv(csvfile)

    prog = prolog.format(csvfile=csvfile) + '\n'
    print prog

    prog += "ggsave('{0}', {1} + t, width=9.883, height=7.392, units = 'in')".format(fname, plot(app))
    execute_r(prog, True)
