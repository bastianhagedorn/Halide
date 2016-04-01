include ../support/Makefile.inc

l1_vs_shared: gpu_l1_vs_shared.cpp
	$(CXX) $(CXXFLAGS) gpu_l1_vs_shared.cpp $(LIB_HALIDE) -o l1_vs_shared -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

clean:
	rm -f l1_vs_shared
