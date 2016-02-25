TESTS=filter_ref filter_auto filter_naive filter_sweep # specify before include!
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

harris_sweep.o: harris
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=131072 ./harris -1

filter_%: harris_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f harris harris*.o harris.h\
		filter_ref filter_auto bench_*.txt ref_perf.txt auto_perf.txt out.png
