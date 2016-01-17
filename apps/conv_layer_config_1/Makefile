TESTS=conv_bench_1 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto conv_bench_1

#p_N=16 p_d_w=224 p_d_h=224 p_ch=3 p_n_f=64 p_f_w=3 p_f_h=3
conv_bench_1: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=16 -Dp_d_w=224 -Dp_d_h=224 -Dch=3 -Dp_n_f=64 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
