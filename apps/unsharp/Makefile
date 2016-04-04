TESTS=filter_auto filter_ref filter_naive filter_sweep filter_rand
include ../support/Makefile.inc

auto: clean_auto filter_auto

unsharp: unsharp.cpp
	$(CXX) $(CXXFLAGS) unsharp.cpp $(LIB_HALIDE) -o \
		unsharp -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

unsharp_ref.o: unsharp
	./unsharp 0

unsharp_auto.o: unsharp
	./unsharp -1

unsharp_naive.o: unsharp
	HL_AUTO_NAIVE=1 ./unsharp -1

unsharp_naive_gpu.o: unsharp
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./unsharp -2

unsharp_auto_gpu.o: unsharp
	HL_AUTO_GPU=1 ./unsharp -2

unsharp_sweep.o: unsharp
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=131072 ./unsharp -1

unsharp_rand.o: unsharp
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=1048576 ./unsharp -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=16384 ./unsharp -1

filter_%: unsharp_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_variants
	rm -f unsharp unsharp.o unsharp.h\
		filter_ref filter_auto filter_sweep filter_rand out.png
