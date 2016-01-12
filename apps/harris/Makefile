include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g -Werror -ffast-math

harris: harris.cpp
	$(CXX) $(CXXFLAGS) harris.cpp $(LIB_HALIDE) -o \
		harris -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

harris_ref: harris
	./harris 0

harris_auto: harris
	./harris -1

filter_auto: harris_auto filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp harris.o -lpthread -ldl -o filter_auto \
		$(PNGFLAGS)

filter_ref: harris_ref filter.cpp
	$(CXX) $(CXXFLAGS) filter.cpp harris.o -lpthread -ldl -o filter_ref \
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
	rm -f harris harris.o harris.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
