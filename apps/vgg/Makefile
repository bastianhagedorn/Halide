TESTS=vgg
include ../support/Makefile.inc

.PHONY: clean

vgg: vgg.cpp layers.h
	$(CXX) $(CXXFLAGS) vgg.cpp $(LIB_HALIDE) -o vgg -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f vgg
