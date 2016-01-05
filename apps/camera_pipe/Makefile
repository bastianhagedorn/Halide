include ../support/Makefile.inc

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

camera_pipe: ../../ camera_pipe.cpp
	#$(MAKE) -C $(LIB_HALIDE)
	$(CXX) $(CXXFLAGS) camera_pipe.cpp $(LIB_HALIDE) -o camera_pipe -ldl \
		-lpthread -lz $(LDFLAGS) -ltinfo

curved_ref: camera_pipe
	./camera_pipe 8 0 # 8-bit output,

curved_auto: camera_pipe
	./camera_pipe 8 -1 # 8-bit output,

fcam/Demosaic.o: fcam/Demosaic.cpp fcam/Demosaic.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

fcam/Demosaic_ARM.o: fcam/Demosaic_ARM.cpp fcam/Demosaic_ARM.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

process_ref: process.cpp curved_ref fcam/Demosaic.o 
	$(CXX) $(CXXFLAGS) process.cpp fcam/Demosaic.o curved.o \
		-o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)

process_auto: process.cpp curved_auto fcam/Demosaic.o 
	$(CXX) $(CXXFLAGS) process.cpp fcam/Demosaic.o curved.o \
		-o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)

out.png: process_ref
	./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

../../bin/HalideTraceViz:
	$(MAKE) -C ../../util/HalideTraceViz

camera_pipe.avi: camera_pipe.cpp viz.sh $(HALIDE_TRACE_VIZ) ../../util/HalideTraceViz
	bash viz.sh

bench_ref: process_ref
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./process_ref ../images/bayer_raw.png 3700 2.0 50 5 out.png >> ref_perf.txt;\
    done

bench_auto: process_auto
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./process_auto ../images/bayer_raw.png 3700 2.0 50 5 out.png >> auto_perf.txt;\
    done

clean:
	rm -f out.png process_ref process_auto curved.o camera_pipe fcam/*.o \
		ref_perf.txt auto_perf.txt curved.s curved.h
