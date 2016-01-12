include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g -Werror -ffast-math

bilateral_grid: bilateral_grid.cpp
	$(CXX) $(CXXFLAGS) bilateral_grid.cpp $(LIB_HALIDE) -o \
		bilateral_grid -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bilateral_grid_ref: bilateral_grid
	./bilateral_grid 8 0

bilateral_grid_auto: bilateral_grid
	./bilateral_grid 8 -1

filter_auto: bilateral_grid_auto filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp bilateral_grid.o -lpthread -ldl -o filter_auto \
		$(PNGFLAGS)

filter_ref: bilateral_grid_ref filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp bilateral_grid.o -lpthread -ldl -o filter_ref \
		$(PNGFLAGS)

bilateral_grid.mp4: bilateral_grid.cpp viz.sh filter_ref
	bash viz.sh

out.png: filter_ref
	./filter_ref ../images/gray.png out.png 0.1 10

bench_ref: filter_ref
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_ref ../images/gray.png \
												out.png 0.1 10 >> ref_perf.txt;\
    done

bench_auto: filter_auto
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_auto ../images/gray.png \
												out.png 0.1 10 >> auto_perf.txt;\
    done

clean:
	rm -f bilateral_grid bilateral_grid.mp4 bilateral_grid.o bilateral_grid.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
