include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

out.png: interpolate
	./interpolate ../images/rgba.png out.png

bench_ref: interpolate
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./interpolate ../images/rgba.png out.png 2>> ref_perf.txt;\
    done

bench_auto: interpolate
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./interpolate ../images/rgba.png out.png -1 2>> auto_perf.txt;\
    done

clean:
	rm -f interpolate interpolate.h out.png ref_perf.txt auto_perf.txt
