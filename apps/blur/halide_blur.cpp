#include "Halide.h"
#include "../support/auto_build.h"
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(16), 2);
    Func blur_x("blur_x"), blur_y("blur_y");
    Var x("x"), y("y"), xi("xi"), yi("yi");

    // The algorithm
    blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;

    // Adding bounds
    //blur_y.bound(x, 0, 6400).bound(y, 0, 4800);
    blur_y.estimate(x, 0, 6400).estimate(y, 0, 4800);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    if (schedule == 0) {
        // Repository schedule
        blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
        blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
    }

    Target target = get_target_from_environment();

    if (schedule == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        target.set_feature(Halide::Target::CUDA);
        //target.set_feature(Halide::Target::Debug);
    }

    auto_build(blur_y, "halide_blur", {input}, target,
                                        (schedule == -1 || schedule == -2));

    /*
    blur_y.split(y, y, yi, 4).split(x, x, xi, 64).reorder(xi, yi, x, y).
                                                parallel(y).vectorize(xi);
    blur_x.compute_at(blur_y, x).vectorize(x);

    blur_y.compile_to_lowered_stmt("halide_blur_lower", {input});
    blur_y.compile_to_c("halide_gen_blur.cpp",  {input});
    blur_y.compile_to_header("halide_gen_blur.h",  {input});
    */

    return 0;
}
