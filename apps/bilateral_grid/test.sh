#!/usr/bin/env bash
PROCESS="./filter_$1"
${PROCESS} ../images/gray.png out.png 0.1 10
