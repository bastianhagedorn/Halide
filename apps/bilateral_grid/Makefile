TESTS=filter_ref filter_auto filter_naive filter_sweep filter_rand# specify before include!
include ../support/Makefile.inc

auto: clean_auto filter_auto

bilateral_grid: bilateral_grid.cpp
	$(CXX) $(CXXFLAGS) bilateral_grid.cpp $(LIB_HALIDE) -o \
		bilateral_grid -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bilateral_grid_ref.o: bilateral_grid
	./bilateral_grid 8 0

bilateral_grid_naive.o: bilateral_grid
	# naive
	HL_AUTO_NAIVE=1 ./bilateral_grid 8 -1

bilateral_grid_naive_gpu.o: bilateral_grid
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./bilateral_grid 8 -2

bilateral_grid_auto_gpu.o: bilateral_grid
	HL_AUTO_GPU=1 ./bilateral_grid 8 -2

bilateral_grid_auto.o: bilateral_grid
	./bilateral_grid 8 -1

bilateral_grid_sweep.o: bilateral_grid
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=262144 ./bilateral_grid 8 -1

bilateral_grid_rand.o: bilateral_grid
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=30 HL_AUTO_FAST_MEM_SIZE=33554432 ./bilateral_grid 8 -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=16 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=40 HL_AUTO_FAST_MEM_SIZE=524288 ./bilateral_grid 8 -1

filter_%: bilateral_grid_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

bilateral_grid.mp4: bilateral_grid.cpp viz.sh filter_ref
	bash viz.sh

out.png: filter_ref
	./filter_ref ../images/gray.png out.png 0.1 10

clean: clean_bench clean_variants
	rm -f bilateral_grid bilateral_grid.mp4 bilateral_grid.o bilateral_grid.h bilateral_grid_*.o\
		filter_ref filter_auto filter_naive filter_sweep filter_rand filter_naive_gpu filter_auto_gpu out.png
