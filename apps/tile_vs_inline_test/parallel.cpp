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

    Var x("x"), y("y"), z("z"), c("c"), fused("fused");
    Var yi("yi"), xi("xi");
    Func f("f"), g("g");
    f(x, y, z, c) = (input(x, y, c)/input(x, z, c));
    g(x, y, c) =  f(x, y, (x+y)%10, c)
                  + f(x, y+1, (x*y)%10, c)
                  + f(x, y+2, (x-y)%10, c)
                  + f(x+1, y, (x)%10, c)
                  + f(x+2, y, (y)%10, c);

    int sched = atoi(argv[1]);
    Target target = get_target_from_environment();
    if (sched == 1) {
        // Testing when there is no inlining
        g.split(y, y, yi, 64).reorder(yi, c, y).parallel(y).vectorize(x, 8);
        f.compute_at(g, c).vectorize(x, 8);
    } else if (sched == 0) {
        // Testing when there is inlining
        g.reorder(c, y).parallel(y).vectorize(x, 8);
    }

    g.bound(x, 0, 1534).bound(y, 0, 2558).bound(c, 0, 3);
    // JIT compile the pipeline eagerly, so we don't interfere with timing
    if (sched == -1)
        g.compile_jit(target, true);
    else
        g.compile_jit(target, false);

    Image<float> in_png = load_image(argv[2]);
    input.set(in_png);
    Image<float> out(1534, 2558, 3);

    double best = benchmark(5, 1, [&]() { g.realize(out); });
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
