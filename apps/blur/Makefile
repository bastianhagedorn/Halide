include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

halide_blur: halide_blur.cpp
	$(CXX) $(CXXFLAGS) halide_blur.cpp $(LIB_HALIDE) -o halide_blur -ldl -lpthread -lz $(LDFLAGS) -ltinfo

halide_blur_ref: halide_blur
	./halide_blur 0

halide_blur_auto: halide_blur
	./halide_blur -1

test_ref: test.cpp halide_blur_ref
	$(CXX) $(CXXFLAGS) test.cpp halide_blur.o -o test_ref -lpthread -ldl $(PNGFLAGS)

test_auto: test.cpp halide_blur_auto
	$(CXX) $(CXXFLAGS) test.cpp halide_blur.o -o test_auto -lpthread -ldl $(PNGFLAGS)

bench_ref: test_ref
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_ref >> ref_perf.txt;\
    done

bench_auto: test_auto
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_auto >> auto_perf.txt;\
    done

clean:
	rm -f test halide_blur halide_blur.h halide_blur.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
