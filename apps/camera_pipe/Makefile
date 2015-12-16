include ../support/Makefile.inc

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

CXX=icpc -march=native -ansi-alias -O3 -Wall -openmp -vec-report
#CXX=g++ -march=native -O3 -Wall -fopenmp

all: process

camera_pipe: ../../ camera_pipe.cpp
	#$(MAKE) -C $(LIB_HALIDE)
	$(CXX) $(CXXFLAGS) camera_pipe.cpp -g $(LIB_HALIDE) -o camera_pipe -ldl -lpthread -lz $(LDFLAGS) -ltinfo

curved.o: camera_pipe
	./camera_pipe 8 0 # 8-bit output,

fcam/Demosaic.o: fcam/Demosaic.cpp fcam/Demosaic.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

fcam/Demosaic_ARM.o: fcam/Demosaic_ARM.cpp fcam/Demosaic_ARM.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

process: process.cpp cam_c.cpp curved.o fcam/Demosaic.o fcam/Demosaic_ARM.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)

out.png: process
	./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

../../bin/HalideTraceViz:
	$(MAKE) -C ../../ bin/HalideTraceViz

camera_pipe.avi: camera_pipe.cpp viz.sh $(HALIDE_TRACE_VIZ) ../../bin/HalideTraceViz
	bash viz.sh

bench: process
	OMP_NUM_THREADS=1 HL_NUM_THREADS=1  ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	OMP_NUM_THREADS=2 HL_NUM_THREADS=2  ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	OMP_NUM_THREADS=4 HL_NUM_THREADS=4  ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	OMP_NUM_THREADS=8 HL_NUM_THREADS=8  ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=1 numactl --physcpubind=0 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	#HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

clean:
	rm -f out.png process curved.o camera_pipe fcam/*.o
