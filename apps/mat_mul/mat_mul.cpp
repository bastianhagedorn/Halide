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

    Func out;
    out(x, y) = prod(x, y);

    out.bound(x, 0, size).bound(y, 0, size);

    int sched = atoi(argv[1]);

    if (sched == 0) {
        Var xi, yi, xii, yii;
        // Tile the output domain
        prod.compute_at(out, x).vectorize(x);
        prod.update().reorder(x, y, r).vectorize(x).unroll(y);
        out.tile(x, y, xi, yi, 16, 4).vectorize(xi).unroll(yi).parallel(y);
    }

    Target target = get_target_from_environment();
    if (sched == -1)
        out.compile_jit(target, true);
    else
        out.compile_jit(target, false);

    target.set_features({Target::NoAsserts, Target::NoRuntime, Target::NoBoundsQuery});
    // out.compile_to_assembly("/dev/stderr", {A, B}, target);

    std::vector<Func> outs;
    outs.push_back(out);
    double best = benchmark(3, 1, [&]() { out.realize(C); });
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
