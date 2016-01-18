#!/bin/bash

label="conv_configs"

qsub -N $label ${HOME}/ravi/apps/qsub/bench.pbs -F "-l $label -p6 -e conv_layer_config_1 -e conv_layer_config_2 -e conv_layer_config_3 -e conv_layer_config_4 -e conv_layer_config_5 -e conv_layer_config_6 -e conv_layer_config_7 -e conv_layer_config_8 -e conv_layer_config_9"