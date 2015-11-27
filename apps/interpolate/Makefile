include ../support/Makefile.inc

CXXFLAGS += -g -Wall

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

out.png: interpolate
	./interpolate ../images/rgba.png out.png

bench: interpolate
	HL_NUM_THREADS=1  ./interpolate ../images/rgba.png out.png
	HL_NUM_THREADS=2  ./interpolate ../images/rgba.png out.png
	HL_NUM_THREADS=4  ./interpolate ../images/rgba.png out.png
	HL_NUM_THREADS=8  ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=1 numactl --physcpubind=0 ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./interpolate ../images/rgba.png out.png
	#HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./interpolate ../images/rgba.png out.png

clean:
	rm -f interpolate interpolate.h out.png
