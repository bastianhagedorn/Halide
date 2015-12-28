#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(16), 2);
    Var x("x"), y("y"), c("c");


    Func f("f");
    f(x, y) = input(x, y) * input(x, y);

    Func g("g");
    g(x, y) = (f(x, y) + f(x + 1, y))/2;

    Func h("h");
    h(x, y) = (f(x, y) + f(x, y+1))/2;

    // Adding bounds
    g.bound(x, 0, 6200).bound(y, 0, 4600);
    h.bound(x, 0, 6200).bound(y, 0, 4600);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    if (schedule == 0) {
        g.parallel(y).vectorize(x, 8);
        h.parallel(y).vectorize(x, 8);
    }

    std::vector<Func> outs;
    outs.push_back(h);
    outs.push_back(g);
    Pipeline test(outs);

    Target target = get_target_from_environment();
    if (schedule == -1)
        test.compile_to_file("data_dep", {input}, target, true);
    else
        test.compile_to_file("data_dep", {input});

    return 0;
}
