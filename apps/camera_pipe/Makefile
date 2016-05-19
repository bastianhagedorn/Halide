TESTS=process_ref process_auto process_naive process_sweep process_rand # specify before include!
include ../support/Makefile.inc

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

auto: clean_auto process_auto

camera_pipe: ../../ camera_pipe.cpp
	$(CXX) $(CXXFLAGS) camera_pipe.cpp $(LIB_HALIDE) -o camera_pipe -ldl \
		-lpthread -lz $(LDFLAGS) -ltinfo

curved_ref.o: camera_pipe
	./camera_pipe 8 0 # 8-bit output,

curved_auto.o: camera_pipe
	./camera_pipe 8 -1 # 8-bit output,

curved_naive.o: camera_pipe
	HL_AUTO_NAIVE=1 ./camera_pipe 8 -1 # 8-bit output,

curved_naive_gpu.o: camera_pipe
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./camera_pipe 8 -2 # 8-bit output,

curved_auto_gpu.o: camera_pipe
	HL_AUTO_GPU=1 ./camera_pipe 8 -2 # 8-bit output,

curved_sweep.o: camera_pipe
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=32768 ./camera_pipe 8 -1 # 8-bit output,

curved_rand.o: camera_pipe
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=18 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=15 HL_AUTO_FAST_MEM_SIZE=262144 ./camera_pipe 8 -1 # 8-bit output,
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=16384 ./camera_pipe 8 -1 # 8-bit output,

fcam/Demosaic.o: fcam/Demosaic.cpp fcam/Demosaic.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

process_%: process.cpp curved_%.o fcam/Demosaic.o
	$(CXX) $(CXXFLAGS) $^ \
		-o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)
out.png: process_ref
	./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

../../bin/HalideTraceViz:
	$(MAKE) -C ../../util/HalideTraceViz

camera_pipe.avi: camera_pipe.cpp viz.sh $(HALIDE_TRACE_VIZ) ../../util/HalideTraceViz
	bash viz.sh

clean: clean_bench clean_variants
	rm -f out.png process_ref process_auto process_sweep process_rand curved*.o \
		camera_pipe fcam/*.o curved.s curved.h process_naive_gpu process_auto_gpu
