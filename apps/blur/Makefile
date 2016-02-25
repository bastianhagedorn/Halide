include ../support/Makefile.inc

auto: clean_auto test_auto

halide_blur: halide_blur.cpp
	$(CXX) $(CXXFLAGS) halide_blur.cpp $(LIB_HALIDE) -o halide_blur $(LLVM_LDFLAGS)

halide_blur_ref.o: halide_blur
	./halide_blur 0

halide_blur_auto.o: halide_blur
	./halide_blur -1

halide_blur_naive.o: halide_blur
	HL_AUTO_NAIVE=1 ./halide_blur -1

halide_blur_sweep.o: halide_blur
	HL_AUTO_SWEEP=1 ./halide_blur -1

test_%: test.cpp halide_blur_%.o
	$(CXX) $(CXXFLAGS) $^ -fopenmp -o $@ -march=native -lpthread -ldl $(PNGFLAGS)

clean: clean_bench
	rm -f test halide_blur halide_blur.h halide_blur*.o test_ref \
    test_auto
