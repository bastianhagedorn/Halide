TESTS=filter_ref filter_auto filter_naive # specify before include!
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

bilateral_grid_auto.o: bilateral_grid
	./bilateral_grid 8 -1

filter_%: bilateral_grid_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

bilateral_grid.mp4: bilateral_grid.cpp viz.sh filter_ref
	bash viz.sh

out.png: filter_ref
	./filter_ref ../images/gray.png out.png 0.1 10

clean: clean_bench clean_variants
	rm -f bilateral_grid bilateral_grid.mp4 bilateral_grid.o bilateral_grid.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
