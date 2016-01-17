TESTS=parallel
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto parallel

parallel: parallel.cpp
	$(CXX) $(CXXFLAGS) parallel.cpp $(LIB_HALIDE) -o parallel -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

cleangen: .dummy

clean: clean_bench clean_variants
	rm -f parallel parallel.h out.png ref_perf.txt auto_perf.txt
