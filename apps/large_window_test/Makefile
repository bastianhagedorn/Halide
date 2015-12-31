include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

large_window_test: large_window_test.cpp
	$(CXX) $(CXXFLAGS) large_window_test.cpp $(LIB_HALIDE) -o large_window_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

large_window_test_ref: large_window_test
	./large_window_test 0

large_window_test_auto: large_window_test
	./large_window_test -1

test_ref: test.cpp large_window_test_ref
	$(CXX) $(CXXFLAGS) test.cpp large_win.o -o test_ref -lpthread -ldl $(PNGFLAGS)

test_auto: test.cpp large_window_test_auto
	$(CXX) $(CXXFLAGS) test.cpp large_win.o -o test_auto -lpthread -ldl $(PNGFLAGS)

bench_ref: test_ref
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_ref >> ref_perf.txt;\
    done

bench_auto: test_auto
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_auto >> auto_perf.txt;\
    done

clean:
	rm -f test large_window_test large_win.h large_win.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
