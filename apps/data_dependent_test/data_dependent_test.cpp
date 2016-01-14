#include "Halide.h"
using namespace Halide;
#include "../support/auto_build.h"

int main(int argc, char **argv) {

    ImageParam input(UInt(16), 2);
    Var x("x"), y("y"), c("c");


    Func f("f");
    f(x, y, c) = input(x, y) * input(c, c);

    Func g("g");
    g(x, y) = (f(x, y, input(x, y)%10) + f(x + 1, y, (input(x, y) - 1)%10))/2;

    // Adding bounds
    g.bound(x, 0, 6200).bound(y, 0, 4600);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    if (schedule == 0) {
        g.parallel(y).vectorize(x, 8);
    }

    Target target = get_target_from_environment();
    auto_build(g, "data_dep", {input}, target, (schedule == -1));

    return 0;
}
