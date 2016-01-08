include ../support/Makefile.inc

all: filter

non_local_means: non_local_means.cpp
	$(CXX) $(CXXFLAGS) non_local_means.cpp -g $(LIB_HALIDE) -o non_local_means -lpthread -ldl -lz $(LDFLAGS) \
	$(LLVM_SHARED_LIBS)

non_local_means.o: non_local_means
	./non_local_means 5 5

filter: non_local_means.o filter.cpp
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -Wall -Werror filter.cpp non_local_means.o -lpthread -ldl -o filter  $(PNGFLAGS)

out.png: filter
	./filter ../images/rgb.png out.png 0.03

clean:
	rm -f non_local_means non_local_means.o non_local_means.h filter

