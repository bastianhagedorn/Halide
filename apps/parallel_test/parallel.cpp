#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>

#include "benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

using std::vector;

int main(int argc, char **argv) {

    ImageParam input(Float(32), 3);

    Var x("x"), y("y"), z("z"), c("c");
    Func f("f");
    f(x, y, z, c) = input(x, y, z) + input(x, y, c);

    int sched = atoi(argv[1]);
    Target target = get_target_from_environment();
    if (sched == 0) {
        f.parallel(y).vectorize(x, 8);
    }

    f.bound(x, 0, 1536).bound(y, 0, 2560).bound(c, 0, 3);
    // JIT compile the pipeline eagerly, so we don't interfere with timing
    if (sched == -1)
        f.compile_jit(target, true);
    else
        f.compile_jit(target, false);

    Image<float> in_png = load_image(argv[2]);
    input.set(in_png);
    Image<float> out(1536, 2560, 3, 3);

    double best = benchmark(5, 1, [&]() { f.realize(out); });
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
