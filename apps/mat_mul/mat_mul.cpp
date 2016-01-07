#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>
#include "benchmark.h"

int main(int argc, char **argv) {
    int size = 2048;
    Image<float> A(size, size);
    Image<float> B(size, size);
    Image<float> C(size, size);

    for (int y = 0; y < A.height(); y++) {
        for (int x = 0; x < A.width(); x++) {
            A(x, y) = rand() & 0xfff;
        }
    }

    for (int y = 0; y < B.height(); y++) {
        for (int x = 0; x < B.width(); x++) {
            B(x, y) = rand() & 0xfff;
        }
    }

    Var x, y;

    Func prod("prod");
    RDom r(0, size);

    prod(x, y) = 0.0f;
    prod(x, y) += A(x, r.x) * B(r.x, y);

    prod.bound(x, 0, size).bound(y, 0, size);

    int sched = atoi(argv[1]);

    if (sched == 0) {
        Var xi, yi;
        RVar ri;
        prod.compute_root().parallel(y).vectorize(x, 8);
        prod.update().parallel(y).reorder(x, r.x).vectorize(x, 8);
        //prod.update().split(x, x, xi, 64).split(y, y, yi, 64).
        //              split(r.x, r.x, ri, 64).parallel(y).reorder(xi,
        //                      ri, yi, r.x, x, y).vectorize(xi);
        //prod.print_loop_nest();
    }

    Target target = get_target_from_environment();
    if (sched == -1)
        prod.compile_jit(target, true);
    else
        prod.compile_jit(target, false);

    std::vector<Func> outs;
    outs.push_back(prod);
    double best = benchmark(3, 1, [&]() { prod.realize(C); });
    std::cerr << best * 1e3 << std::endl;
}
