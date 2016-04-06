TESTS=lens_blur
include ../support/Makefile.inc

CXXFLAGS += -g -Wall

.PHONY: clean

auto: clean_auto lens_blur

lens_blur: lens_blur.cpp
	$(CXX) $(CXXFLAGS) lens_blur.cpp $(LIB_HALIDE) -o lens_blur -lpthread -ldl -lz -ltinfo \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

out.png: lens_blur
	./lens_blur left.png right.png out.png 0

clean: clean_variants
	rm -f lens_blur out.png
