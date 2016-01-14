TESTS=interpolate
include ../support/Makefile.inc

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

out.png: interpolate
	./interpolate ../images/rgba.png out.png

clean: clean_bench
	rm -f interpolate out.png
