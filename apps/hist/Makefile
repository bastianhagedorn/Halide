TESTS=filter_ref filter_auto filter_naive filter_sweep filter_rand # specify before include!
include ../support/Makefile.inc

auto: clean_auto filter_auto

hist: hist.cpp
	$(CXX) $(CXXFLAGS) hist.cpp $(LIB_HALIDE) -o \
		hist -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

hist_ref.o: hist
	./hist 0

hist_auto.o: hist
	./hist -1

hist_naive.o: hist
	HL_AUTO_NAIVE=1 ./hist -1

hist_naive_gpu.o: hist
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./hist -2

hist_auto_gpu.o: hist
	HL_AUTO_GPU=1 ./hist -2

hist_sweep.o: hist
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=32768 ./hist -1

hist_rand.o: hist
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=5 HL_AUTO_FAST_MEM_SIZE=524288 ./hist -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=4 HL_AUTO_BALANCE=40 HL_AUTO_FAST_MEM_SIZE=262144 ./hist -1

filter_%: hist_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f hist hist*.o hist.h\
		filter_ref filter_auto filter_sweep filter_rand out.png filter_naive_gpu filter_auto_gpu
