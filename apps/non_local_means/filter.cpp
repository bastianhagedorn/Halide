#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "non_local_means.h"

#include "benchmark.h"
#include "halide_image.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("Usage: ./filter input.png output.png sigma\n"
               "e.g. ./filter input.png output.png 0.1\n");
        return 0;
    }

    Image<float> input = load_image(argv[1]);
    Image<float> output(input.width(), input.height(), input.channels());

    non_local_means(atof(argv[3]), input, output);

    // Timing code. Timing doesn't include copying the input data to
    // the gpu or copying the output back.
    double min_t = benchmark(1, 1, [&]() {
        non_local_means(atof(argv[3]), input, output);
    });
    printf("%g ms\n", min_t * 1e3);

    save_image(output, argv[2]);

    return 0;
}
