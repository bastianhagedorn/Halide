include ../support/Makefile.inc

auto: clean_auto test_auto

data_dependent_test: data_dependent_test.cpp
	$(CXX) $(CXXFLAGS) data_dependent_test.cpp $(LIB_HALIDE) -o data_dependent_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

data_dep_ref.o: data_dependent_test
	./data_dependent_test 0

data_dep_auto.o: data_dependent_test
	./data_dependent_test -1

data_dep_naive.o: data_dependent_test
	HL_AUTO_NAIVE=1 ./data_dependent_test -1

data_dep_sweep.o: data_dependent_test
	HL_AUTO_SWEEP=1 ./data_dependent_test -1

test_%: test.cpp data_dep_%.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean: clean_bench clean_variants
	rm -f test data_dependent_test data_dep.h data_dep*.o test_ref \
    test_auto
