qsub -N camera_pipe_vec16 -v HL_AUTO_VEC_WIDTH=16 /home/jrk-temp/ravi/apps/qsub/bench.pbs -F "-e camera_pipe -l batchtest -p 6"
