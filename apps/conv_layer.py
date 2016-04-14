#/usr/bin/env python
from pygg import *
import pandas
from sqlalchemy import create_engine
from tempfile import mkstemp

import sys, os

resfname='conv_layer.csv'
res = pandas.read_csv(resfname)

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
require(gridExtra)

data = read.csv('{csvfile}',sep=',')
data$version <- factor(data$version, levels=c('auto'))
data$threads <- factor(data$threads)
data$app <- factor(data$app)

t = theme(
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          axis.line = element_line(colour = "grey20", size = 0.15),
          axis.text.x = element_text(colour="grey20",size=2, face="plain"),
          axis.text.y = element_text(colour="grey20",size=2, face="plain"),

          axis.ticks=element_blank(),
          panel.grid.major=element_blank(),
          panel.background=element_blank(),
          panel.grid.minor=element_blank(),
          panel.border=element_blank(),

          axis.ticks.margin = unit(1,'pt'),
          axis.ticks.length = unit(0,'pt'),
          panel.margin=unit(0,'pt'),
          plot.title = element_text(size=2.5),
          plot.margin= unit(c(0, 0, 0, 0), "lines"),

          plot.background=element_blank(),

          legend.position="none"
          )
"""
#          axis.line=element_blank(),
#          axis.text.x=element_blank(),
#          axis.text.y=element_blank(),

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
    pl = ggplot("subset(data, (data$app == '{0}') & (data$threads == 1 | data$threads == 2 | data$threads == 3 | data$threads == 4 | data$threads == 5 | data$threads == 6 | data$threads == 7 | data$threads == 8 | data$threads == 9 ))".format(app),
                aes(x='threads', y='runtime_norm')) + ylim(0,2) # + labs(x='NULL',y='NULL') + guides(fill='FALSE')
    pl+= geom_bar(aes(fill='version'), width='0.25', stat="'identity'", position="position_dodge(width=0.25), binwidth=0")
    pl+= scale_fill_manual('values=c("#F95738")')
    pl+= ggtitle("'{0}'".format(printable_name[app]))
    pl+= scale_x_discrete('expand=c(0, 0.5), labels=c("layer1", "layer2","layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9")')
    pl+= scale_y_continuous('limits=c(0, 2), expand=c(0, 0), breaks=c(0, 1), labels = c("0", "1")')
    pl+= coord_fixed(ratio = 1)
    #pl+= geom_hline(yintercept=1)
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
prog = "plots <- list()" + '\n'
plot_num = 0
arrange_str = ""
for app in apps:
    print '\n\n\n===== {0} ====='.format(app)
    plot_num = plot_num + 1
    app_name_norm = app.replace(' ', '_').lower()
    fname = 'fig1-{0}.pdf'.format(app_name_norm)

    # select
    reldata = res[((res.threads == 1) | (res.threads == 2) | (res.threads == 3)
                  | (res.threads == 4) | (res.threads == 5) | (res.threads == 6) |
                  (res.threads == 7) | (res.threads == 8) | (res.threads == 9)) & (res.app == app)]

    (csvfp,csvfile) = mkstemp(suffix='.csv')
    reldata.to_csv(csvfile)

    prog += prolog.format(csvfile=csvfile) + '\n'

    arrange_str += "p{0},".format(plot_num)
    prog += "p{0} <- {1} + t".format(plot_num, plot(app)) + '\n'
prog += "pdf('conv_layer.pdf', width = 5, height = 1)" + '\n'
prog += "grid.arrange(" + arrange_str + "ncol = 1, clip=TRUE)" + '\n'
prog += "dev.off()" + '\n'
print prog
execute_r(prog, True)

