TESTS=vgg
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto vgg

vgg: vgg.cpp layers.h
	$(CXX) $(CXXFLAGS) vgg.cpp $(LIB_HALIDE) -o vgg -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench clean_variants
	rm -f vgg
