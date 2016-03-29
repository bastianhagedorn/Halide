include ../support/Makefile.inc

auto: test_auto clean_auto

overlap_test: overlap_test.cpp
	$(CXX) $(CXXFLAGS) overlap_test.cpp $(LIB_HALIDE) -o overlap_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

overlap_ref.o: overlap_test
	./overlap_test 0

overlap_auto.o: overlap_test
	./overlap_test -1

overlap_naive.o: overlap_test
	HL_AUTO_NAIVE=1 ./overlap_test -1

overlap_sweep.o: overlap_test
	HL_AUTO_SWEEP=1 ./overlap_test -1

test_%: test.cpp overlap_%.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f test overlap_test overlap.h overlap*.o test_ref test_auto
