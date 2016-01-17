TESTS=mat_mul
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto mat_mul

mat_mul: mat_mul.cpp
	$(CXX) $(CXXFLAGS) mat_mul.cpp $(LIB_HALIDE) -o mat_mul -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench clean_variants
	rm -f mat_mul
