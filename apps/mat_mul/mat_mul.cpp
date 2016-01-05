#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>
#include "benchmark.h"

int main(int argc, char **argv) {

    Image<float> A(2048, 1024);
    Image<float> B(1024, 2048);
    Image<float> C(2048, 2048);

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
    RDom r(0, 1024);

    prod(x, y) = 0.0f;
    prod(x, y) += A(x, r.x) * B(r.x, y);

    prod.bound(x, 0, 2048).bound(y, 0, 2048);

    int sched = atoi(argv[1]);

    if (sched == 0) {
        prod.compute_root().parallel(y).vectorize(x);
        prod.update().parallel(y).reorder(x, r.x).vectorize(x);
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
