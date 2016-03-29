include ../support/Makefile.inc

auto: clean_auto test_auto

cost_function_test: cost_function_test.cpp
	$(CXX) $(CXXFLAGS) cost_function_test.cpp $(LIB_HALIDE) -o cost_function_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

cost_ref.o: cost_function_test
	./cost_function_test 0

cost_auto.o: cost_function_test
	./cost_function_test -1

cost_naive.o: cost_function_test
	HL_AUTO_NAIVE=1 ./cost_function_test -1

cost_sweep.o: cost_function_test
	HL_AUTO_SWEEP=1 ./cost_function_test -1

test_%: test.cpp cost_%.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f test cost_function_test cost.h cost*.o test_ref \
    test_auto
