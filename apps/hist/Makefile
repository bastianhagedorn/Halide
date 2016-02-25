TESTS=filter_ref filter_auto filter_naive filter_sweep # specify before include!
include ../support/Makefile.inc

auto: clean_auto filter_auto

hist: hist.cpp
	$(CXX) $(CXXFLAGS) hist.cpp $(LIB_HALIDE) -o \
		hist -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

hist_ref.o: hist
	./hist 0

hist_auto.o: hist
	./hist -1

hist_naive.o: hist
	HL_AUTO_NAIVE=1 ./hist -1

hist_sweep.o: hist
	HL_AUTO_SWEEP=1 ./hist -1

filter_%: hist_%.o filter.cpp
	$(CXX) $(CXXFLAGS) $^ -lpthread -ldl -o $@ \
		$(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f hist hist*.o hist.h\
		filter_ref filter_auto ref_perf.txt auto_perf.txt out.png
