TESTS=parallel
include ../support/Makefile.inc

.PHONY: clean

parallel: parallel.cpp
	$(CXX) $(CXXFLAGS) parallel.cpp $(LIB_HALIDE) -o parallel -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f parallel parallel.h out.png ref_perf.txt auto_perf.txt
