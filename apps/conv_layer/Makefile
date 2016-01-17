TESTS=conv_bench # specify before include!
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto conv_bench

conv_bench: conv_bench.cpp
	$(CXX) $(CXXFLAGS) conv_bench.cpp $(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench clean_variants
	rm -f conv_bench
