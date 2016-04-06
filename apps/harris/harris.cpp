#include "Halide.h"
#include <stdio.h>
using namespace Halide;
#include "../support/auto_build.h"

Expr sum3x3(Func f, Var x, Var y) {
    return f(x-1, y-1) + f(x-1, y) + f(x-1, y+1) +
           f(x, y-1)   + f(x, y)   + f(x, y+1) +
           f(x+1, y-1) + f(x+1, y) + f(x+1, y+1);
}

int main(int argc, char **argv) {

    ImageParam in(Float(32), 3);

    Var x, y, c;

    Func gray("gray");
    gray(x, y) = 0.299f * in(x, y, 0) + 0.587f * in(x, y, 1)
                 + 0.114f * in(x, y, 2);


    Func Iy("Iy");
    Iy(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x-1, y+1)*(1.0f/12) +
               gray(x, y-1)*(-2.0f/12) + gray(x, y+1)*(2.0f/12) +
               gray(x+1, y-1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);


    Func Ix("Ix");
    Ix(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x+1, y-1)*(1.0f/12) +
               gray(x-1, y)*(-2.0f/12) + gray(x+1, y)*(2.0f/12) +
               gray(x-1, y+1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);


    Func Ixx("Ixx");
    Ixx(x, y) = Ix(x, y) * Ix(x, y);

    Func Iyy("Iyy");
    Iyy(x, y) = Iy(x, y) * Iy(x, y);

    Func Ixy("Ixy");
    Ixy(x, y) = Ix(x, y) * Iy(x, y);

    Func Sxx("Sxx");

    Sxx(x, y) = sum3x3(Ixx, x, y);

    Func Syy("Syy");
    Syy(x, y) = sum3x3(Iyy, x, y);


    Func Sxy("Sxy");
    Sxy(x, y) = sum3x3(Ixy, x, y);

    Func det("det");
    det(x, y) = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);

    Func trace("trace");
    trace(x, y) = Sxx(x, y) + Syy(x, y);

    Func harris("harris");
    harris(x, y) = det(x, y) - 0.04f * trace(x, y) * trace(x, y);

    Func shifted("shifted");
    shifted(x, y) = harris(x + 2, y + 2);

    //shifted.bound(x, 0, 1530).bound(y, 0, 2554);
    shifted.estimate(x, 0, 1530).estimate(y, 0, 2554);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    Target target = get_target_from_environment();
    if (schedule == 0) {
        Var yi, xi;
        if (target.has_gpu_feature()) {
            shifted.gpu_tile(x, y, 14, 14);
            Ix.compute_at(shifted, Var::gpu_blocks()).gpu_threads(x, y);
            Iy.compute_at(shifted, Var::gpu_blocks()).gpu_threads(x, y);
            shifted.print_loop_nest();
        } else {
            shifted.tile(x, y, xi, yi, 128, 128)
                .vectorize(xi, 8).parallel(y);
            Ix.compute_at(shifted, x).vectorize(x, 8);
            Iy.compute_at(shifted, x).vectorize(x, 8);
            shifted.print_loop_nest();
        }
    }


    if (schedule == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }
    auto_build(shifted, "harris", {in}, target, (schedule == -1 || schedule == -2));

    return 0;
}
