TESTS=filter_ref filter_auto # specify before include!
include ../support/Makefile.inc

harris: harris.cpp
	$(CXX) $(CXXFLAGS) harris.cpp $(LIB_HALIDE) -o \
		harris -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

harris_ref.o: harris
	./harris 0

harris_auto.o: harris
	./harris -1

filter_%: harris_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean:
	rm -f harris harris_*.o harris.h\
		filter_ref filter_auto bench_*.txt ref_perf.txt auto_perf.txt out.png
