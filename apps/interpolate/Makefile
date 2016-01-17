TESTS=interpolate
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto interpolate

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

out.png: interpolate
	./interpolate ../images/rgba.png out.png

clean: clean_bench clean_variants
	rm -f interpolate out.png
