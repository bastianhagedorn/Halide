TESTS=conv_bench_6 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

#p_N=16 p_d_w=28 p_d_h=28 p_ch=512 p_n_f=512 p_f_w=3 p_f_h=3
conv_bench_6: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=16 -Dp_d_w=28 -Dp_d_h=28 -Dp_ch=512 -Dp_n_f=512 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
