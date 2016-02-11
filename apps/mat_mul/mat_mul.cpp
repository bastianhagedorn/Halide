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
    Func AT("AT");
    RDom r(0, size);

    AT(x, y) = A(y, x);

    prod(x, y) = 0.0f;
    //prod(x, y) += A(x, r.x) * B(r.x, y);
    prod(x, y) += AT(r.x, x) * B(r.x, y);

    Func out;
    out(x, y) = prod(x, y);

    out.bound(x, 0, size).bound(y, 0, size);

    int sched = atoi(argv[1]);

    if (sched == 0) {
        Var xi, yi, xii, yii;
        // Tile the output domain
        AT.compute_root().parallel(y);
        prod.compute_at(out, x).vectorize(x);
        prod.update().reorder(x, y, r).vectorize(x);//.unroll(y);
        out.tile(x, y, xi, yi, 16, 16).vectorize(xi).unroll(yi).parallel(y);
        out.print_loop_nest();
    } else if (sched == 1) {
        Var xi, yi;
        RVar ri;
        constexpr int tile_size = 128;
        // Tile the output domain
        AT.compute_root();//.parallel(y);
        prod.update().split(r, r, ri, tile_size).reorder(ri, x, y, r).vectorize(x, 8);
        out.tile(x, y, xi, yi, 8, 8).vectorize(xi);//.parallel(y);
        prod.compute_at(out, x);
        out.print_loop_nest();
    }

    Target target = get_target_from_environment();
    if (sched == -1)
        out.compile_jit(target, true);
    else
        out.compile_jit(target, false);

    //target.set_features({Target::NoAsserts, Target::NoRuntime, Target::NoBoundsQuery});
    //out.compile_to_assembly("/dev/stderr", {A, B}, target);

    std::vector<Func> outs;
    outs.push_back(out);
    double best = benchmark(3, 1, [&]() { out.realize(C); });
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
