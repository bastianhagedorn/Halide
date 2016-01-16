#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "benchmark.h"
#include "halide_image.h"
#include "large_win.h"

using namespace Halide::Tools;

//#define cimg_display 0
//#include "CImg.h"
//using namespace cimg_library;

// typedef CImg<uint16_t> Image;

double t;

void test(Image<uint16_t> in) {
    Image<uint16_t> out(in.width(), in.height());
    // Call it once to initialize the halide runtime stuff

    t = benchmark(10, 1, [&]() {
        large_win(in, out);
    });
}

int main(int argc, char **argv) {

    Image<uint16_t> input(6408, 4802);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

    test(input);
    double halide_time = t;

    printf("runtime: %g\n", halide_time * 1000);

    return 0;
}
