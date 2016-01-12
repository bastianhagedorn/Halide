include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g

.PHONY: clean

parallel: parallel.cpp
	$(CXX) $(CXXFLAGS) parallel.cpp $(LIB_HALIDE) -o parallel -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bench_ref: parallel
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./parallel 0 ../images/rgb.png 2>> ref_perf.txt;\
    done

bench_auto: parallel
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./parallel -1 ../images/rgb.png 2>> auto_perf.txt;\
    done

clean:
	rm -f parallel parallel.h out.png ref_perf.txt auto_perf.txt
