TESTS=filter_andrew filter_dillon filter_auto filter_naive filter_sweep filter_rand
include ../support/Makefile.inc

all: filter

auto: clean_auto filter_auto

non_local_means: non_local_means.cpp
	$(CXX) $(CXXFLAGS) $^ -g $(LIB_HALIDE) -o $@ -lpthread -ldl -lz -ltinfo $(LDFLAGS) \
	$(LLVM_SHARED_LIBS)

non_local_means_naive.o: non_local_means
	HL_AUTO_NAIVE=1 ./$(<) 7 7 -1

non_local_means_naive_gpu.o: non_local_means
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./$(<) 7 7 -2

non_local_means_auto_gpu.o: non_local_means
	HL_AUTO_GPU=1 ./$(<) 7 7 -2

non_local_means_auto.o: non_local_means
	./$(<) 7 7 -1

non_local_means_sweep.o: non_local_means
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=262144 ./$(<) 7 7 -1

non_local_means_rand.o: non_local_means
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=1 HL_AUTO_FAST_MEM_SIZE=131072 ./$(<) 7 7 -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=131072 ./$(<) 7 7 -1

non_local_means_ref.o: non_local_means
	./$(<) 7 7 0

#non_local_means_andrew.o: non_local_means
#	./$(<) 7 7 1
#	mv non_local_means_ref.o $@

#non_local_means_dillon.o: non_local_means
#	./$(<) 7 7 2
#	mv non_local_means_ref.o $@

filter_%: non_local_means_%.o filter.cpp
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -Wall -Werror $^ -lpthread -ldl -o $@ $(PNGFLAGS)

out.png: filter_ref
	HL_NUM_THREADS=4 ./$(<) ../images/rgb_small_noisy.png out.png 0.12

clean: clean_variants
	rm -rf non_local_means non_local_means_dillon non_local_means_andrew non_local_means*.o non_local_means.h filter_* *.dSYM out.png
