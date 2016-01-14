TESTS=conv_bench # specify before include!
include ../support/Makefile.inc

.PHONY: clean

conv_bench: conv_bench.cpp
	$(CXX) $(CXXFLAGS) conv_bench.cpp $(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean:
	rm -f conv_bench conv_bench.h ref_perf.txt auto_perf.txt
