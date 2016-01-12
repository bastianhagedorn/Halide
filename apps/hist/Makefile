include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g -Werror -ffast-math

hist: hist.cpp
	$(CXX) $(CXXFLAGS) hist.cpp $(LIB_HALIDE) -o \
		hist -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

hist_ref: hist
	./hist 0

hist_auto: hist
	./hist -1

filter_auto: hist_auto filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp hist.o -lpthread -ldl -o filter_auto \
		$(PNGFLAGS)

filter_ref: hist_ref filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp hist.o -lpthread -ldl -o filter_ref \
		$(PNGFLAGS)

bench_ref: filter_ref
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_ref ../images/rgb.png \
												out.png >> ref_perf.txt;\
    done

bench_auto: filter_auto
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_auto ../images/rgb.png \
												out.png >> auto_perf.txt;\
    done

clean:
	rm -f hist hist.o hist.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
