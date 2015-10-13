include ../support/Makefile.inc

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

all: process

camera_pipe: ../../ camera_pipe.cpp
	#$(MAKE) -C $(LIB_HALIDE)
	$(CXX) $(CXXFLAGS) camera_pipe.cpp -g $(LIB_HALIDE) -o camera_pipe -ldl -lpthread -lz $(LDFLAGS) -ltinfo

curved.o: camera_pipe
	./camera_pipe 8 1 # 8-bit output,

fcam/Demosaic.o: fcam/Demosaic.cpp fcam/Demosaic.h
	$(CXX) $(CXXFLAGS) -c -Wall -fopenmp -O3 $< -o $@

fcam/Demosaic_ARM.o: fcam/Demosaic_ARM.cpp fcam/Demosaic_ARM.h
	$(CXX) $(CXXFLAGS) -c -Wall -fopenmp -O3 $< -o $@

process: process.cpp curved.o fcam/Demosaic.o fcam/Demosaic_ARM.o
	$(CXX) $(CXXFLAGS) -Wall -O3 $^ -o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)

out.png: process
	./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

../../bin/HalideTraceViz:
	$(MAKE) -C ../../ bin/HalideTraceViz

camera_pipe.avi: camera_pipe.cpp viz.sh $(HALIDE_TRACE_VIZ) ../../bin/HalideTraceViz
	bash viz.sh

bench: process
	HL_NUM_THREADS=1 numactl --physcpubind=0 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png
	HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

clean:
	rm -f out.png process curved.o camera_pipe fcam/*.o
