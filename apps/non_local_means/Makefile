TESTS=filter_andrew filter_dillon filter_auto filter_naive
include ../support/Makefile.inc

all: filter

non_local_means: non_local_means.cpp
	$(CXX) $(CXXFLAGS) $^ -g $(LIB_HALIDE) -o $@ -lpthread -ldl -lz -ltinfo $(LDFLAGS) \
	$(LLVM_SHARED_LIBS)

non_local_means_naive.o: non_local_means
	HL_AUTO_NAIVE=1 ./$(<) 7 7 -1

non_local_means_auto.o: non_local_means
	./$(<) 7 7 -1

non_local_means_base.o: non_local_means
	./$(<) 7 7 0
	mv non_local_means_ref.o $@

non_local_means_andrew.o: non_local_means
	./$(<) 7 7 1
	mv non_local_means_ref.o $@

non_local_means_dillon.o: non_local_means
	./$(<) 7 7 2
	mv non_local_means_ref.o $@

filter_%: non_local_means_%.o filter.cpp
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -Wall -Werror $^ -lpthread -ldl -o $@ $(PNGFLAGS)

out.png: filter_base
	HL_NUM_THREADS=4 ./$(<) ../images/rgb_small_noisy.png out.png 0.12

clean:
	rm -rf non_local_means non_local_means_dillon non_local_means_andrew non_local_means*.o non_local_means.h filter_* *.dSYM out.png
