include ../support/Makefile.inc

CXXFLAGS += -g -Wall

.PHONY: clean

lens_blur: lens_blur.cpp
	$(CXX) $(CXXFLAGS) lens_blur.cpp $(LIB_HALIDE) -o lens_blur -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

out.png: lens_blur
	./lens_blur left.png right.png out.png

clean:
	rm -f lens_blur out.png
