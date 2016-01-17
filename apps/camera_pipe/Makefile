TESTS=process_ref process_auto process_naive # specify before include!
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

clean: clean_bench
	rm -f out.png process_ref process_auto curved*.o \
		camera_pipe fcam/*.o curved.s curved.h
