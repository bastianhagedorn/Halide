include ../support/Makefile.inc

halide_blur: halide_blur.cpp
	$(CXX) $(CXXFLAGS) halide_blur.cpp $(LIB_HALIDE) -o halide_blur -ldl -lpthread -lz $(LDFLAGS) -ltinfo

halide_blur_ref.o: halide_blur
	./halide_blur 0

halide_blur_auto.o: halide_blur
	./halide_blur -1

test_%: test.cpp halide_blur_%.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread -ldl $(PNGFLAGS)

clean: clean_bench
	rm -f test halide_blur halide_blur.h halide_blur*.o test_ref \
    test_auto
