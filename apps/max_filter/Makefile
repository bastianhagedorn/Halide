TESTS=max_filter
include ../support/Makefile.inc

CXXFLAGS += -g -Wall

.PHONY: clean

max_filter: max_filter.cpp
	$(CXX) $(CXXFLAGS) max_filter.cpp $(LIB_HALIDE) -o max_filter -lpthread -ldl -lz -ltinfo \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

out.png: max_filter
	HL_NUM_THREADS=4 ./max_filter ../images/rgb.png out.png 0

clean:
	rm -f max_filter out.png
