TESTS=conv_bench_4 # specify before include!
include ../support/Makefile.inc

.PHONY: clean

#p_N=4 p_d_w=56 p_d_h=56 p_ch=256 p_n_f=256 p_f_w=3 p_f_h=3
conv_bench_4: ../conv_layer/conv_bench.cpp
	$(CXX) $(CXXFLAGS) ../conv_layer/conv_bench.cpp -Dp_N=4 -Dp_d_w=56 -Dp_d_h=56 -Dp_ch=256 -Dp_n_f=256 -Dp_f_w=3 -Dp_f_h=3\
		$(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean: clean_bench
	rm -f conv_bench
