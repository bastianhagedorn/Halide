TESTS=conv_bench_8 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

#p_N=32 p_d_w=14 p_d_h=14 p_ch=256 p_n_f=64 p_f_w=3 p_f_h=3
conv_bench_8: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=32 -Dp_d_w=14 -Dp_d_h=14 -Dp_ch=256 -Dp_n_f=64 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
