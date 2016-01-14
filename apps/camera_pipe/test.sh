#!/usr/bin/env bash
PROCESS="./process_$1"
${PROCESS} ../images/bayer_raw.png 3700 2.0 50 5 out.png
