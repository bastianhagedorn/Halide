TESTS=filter_auto filter_ref filter_naive
include ../support/Makefile.inc

unsharp: unsharp.cpp
	$(CXX) $(CXXFLAGS) unsharp.cpp $(LIB_HALIDE) -o \
		unsharp -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

unsharp_ref.o: unsharp
	./unsharp 0

unsharp_auto.o: unsharp
	./unsharp -1

unsharp_naive.o: unsharp
	HL_AUTO_NAIVE=1 ./unsharp -1

filter_%: unsharp_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean:
	rm -f unsharp unsharp.o unsharp.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
