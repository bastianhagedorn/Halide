#include "Halide.h"
using namespace Halide;

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
    if (schedule == -1)
        g.compile_to_file("data_dep", {input}, target, true);
    else
        g.compile_to_file("data_dep", {input});

    return 0;
}
