TESTS=conv_bench_3 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

auto: clean_auto conv_bench

#p_N=16 p_d_w=56 p_d_h=56 p_ch=128 p_n_f=256 p_f_w=3 p_f_h=3
conv_bench_3: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=16 -Dp_d_w=56 -Dp_d_h=56 -Dch=128 -Dp_n_f=256 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
