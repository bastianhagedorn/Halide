TESTS=conv_bench_2 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto conv_bench_2

#p_N=16 p_d_w=112 p_d_h=112 p_ch=64 p_n_f=128 p_f_w=3 p_f_h=3
conv_bench_2: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=16 -Dp_d_w=112 -Dp_d_h=112 -Dp_ch=64 -Dp_n_f=128 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
