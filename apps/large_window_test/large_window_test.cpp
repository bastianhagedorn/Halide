#include "Halide.h"
using namespace Halide;
#include "../support/auto_build.h"

int main(int argc, char **argv) {

    ImageParam in(UInt(16), 2);
    Var x("x"), y("y"), c("c");

    Func in_b("in_b");
    in_b = BoundaryConditions::repeat_edge(in);

    int win_size = 15;
    RDom w(-win_size, win_size, -win_size, win_size);
    Func f("f");
    f(x, y) = sum(in_b(x + w.x, y + w.y), "sum1")/1024;

    Func g("g");
    g(x, y) = sum(f(x + w.x, y + w.y), "sum2")/1024;

    // Adding bounds
    g.bound(x, 0, 6408).bound(y, 0, 4802);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    if (schedule == 0) {
        Var xi, yi;
        in_b.compute_root();
        f.compute_root().parallel(y).vectorize(x, 8);
        g.compute_root().parallel(y).vectorize(x, 8);
        g.print_loop_nest();
    }

    std::vector<Func> outs;
    outs.push_back(g);
    Pipeline test(outs);

    Target target = get_target_from_environment();
    auto_build(test, "large_win", {in}, target, (schedule == -1));

    return 0;
}
