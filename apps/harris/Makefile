TESTS=filter_ref filter_auto filter_naive filter_sweep filter_rand # specify before include!
include ../support/Makefile.inc

auto: clean_auto filter_auto

harris: harris.cpp
	$(CXX) $(CXXFLAGS) harris.cpp $(LIB_HALIDE) -o \
		harris -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

harris_ref.o: harris
	./harris 0

harris_auto.o: harris
	./harris -1

harris_naive.o: harris
	HL_AUTO_NAIVE=1 ./harris -1

harris_naive_gpu.o: harris
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./harris -2

harris_auto_gpu.o: harris
	HL_AUTO_GPU=1 ./harris -2

harris_sweep.o: harris
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=32768 ./harris -1

harris_rand.o: harris
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=32768 ./harris -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=16 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=16384 ./harris -1

filter_%: harris_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f harris harris*.o harris.h\
		filter_ref filter_auto filter_naive filter_sweep out.png filter_auto_gpu filter_naive_gpu
