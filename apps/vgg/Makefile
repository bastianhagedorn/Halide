include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g

.PHONY: clean

vgg: vgg.cpp layers.h
	$(CXX) $(CXXFLAGS) vgg.cpp $(LIB_HALIDE) -o vgg -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bench_ref: vgg
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./vgg 1 2>> ref_perf.txt;\
    done

bench_auto: vgg
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./vgg -1 2>> auto_perf.txt;\
    done

clean:
	rm -f vgg vgg.h ref_perf.txt auto_perf.txt
