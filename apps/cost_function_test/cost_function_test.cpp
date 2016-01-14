#include "Halide.h"
using namespace Halide;
#include "../support/auto_build.h"

int main(int argc, char **argv) {

    ImageParam input(UInt(16), 2);
    Var x("x"), y("y"), xi("xi"), yi("yi");

    // Simple test scenario
    int num_stencils = 15;

    // TODO
    // 1) Create a very long pipeline and check the effectiveness of the
    //    cost function
    //    - The auto-scheduler should split the grouping at right points in
    //      the pipeline
    //    - Fusion should happen in the right order
    //    - When inlining is enabled the combination of both inling and tiling
    //      should yield better performance

    std::vector<Func> stencils;
    for(int i = 0; i < num_stencils; i ++) {
        Func s("stencil_" + std::to_string(i));
        stencils.push_back(s);
    }

    // The algorithm
    stencils[0](x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    for (int i = 1; i < num_stencils; i++) {
        stencils[i](x, y) = (stencils[i-1](x, y) + stencils[i-1](x, y+1) +
                             stencils[i-1](x, y+2))/3;
    }

    // Adding bounds
    stencils[num_stencils - 1].bound(x, 0, 6200).bound(y, 0, 4600);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    if (schedule == 0) {
        stencils[num_stencils - 1].split(y, y, yi, 64).split(x, x, xi, 64).
            reorder(xi, yi, x, y).parallel(y).vectorize(xi, 8);
        for (int i = 0; i < num_stencils - 1; i+=1)
            stencils[i].compute_at(stencils[num_stencils - 1], x).vectorize(x, 8);
        stencils[num_stencils - 1].print_loop_nest();
    }

    Target target = get_target_from_environment();
    auto_build(stencils[num_stencils - 1], "cost", {input}, target, (schedule == -1));

    return 0;
}
