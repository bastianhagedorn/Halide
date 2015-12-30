include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g -Werror -ffast-math

unsharp: unsharp.cpp
	$(CXX) $(CXXFLAGS) unsharp.cpp $(LIB_HALIDE) -o \
		unsharp -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

unsharp_ref: unsharp
	./unsharp 0

unsharp_auto: unsharp
	./unsharp -1

filter_auto: unsharp_auto filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp unsharp.o -lpthread -ldl -o filter_auto \
		$(PNGFLAGS)

filter_ref: unsharp_ref filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp unsharp.o -lpthread -ldl -o filter_ref \
		$(PNGFLAGS)

bench_ref: filter_ref
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_ref ../images/rgb.png \
												out.png >> ref_perf.txt;\
    done

bench_auto: filter_auto
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./filter_auto ../images/rgb.png \
												out.png >> auto_perf.txt;\
    done

clean:
	rm -f unsharp unsharp.o unsharp.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
