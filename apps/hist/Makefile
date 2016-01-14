TESTS=filter_ref filter_auto # specify before include!
include ../support/Makefile.inc

hist: hist.cpp
	$(CXX) $(CXXFLAGS) hist.cpp $(LIB_HALIDE) -o \
		hist -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

hist_ref.o: hist
	./hist 0

hist_auto.o: hist
	./hist -1

filter_%: hist_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_bench
	rm -f hist hist*.o hist.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
