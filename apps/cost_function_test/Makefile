include ../support/Makefile.inc

cost_function_test: cost_function_test.cpp
	$(CXX) $(CXXFLAGS) cost_function_test.cpp $(LIB_HALIDE) -o cost_function_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

cost_function_test_ref.o: cost_function_test
	./cost_function_test 0

cost_function_test_auto.o: cost_function_test
	./cost_function_test -1

test_%: test.cpp cost_function_test_%
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean:
	rm -f test cost_function_test cost.h cost.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
