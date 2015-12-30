#include "Halide.h"
#include <stdio.h>
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam in(UInt(8), 3);

    Var x, y, c;

    Func gray("gray");
    gray(x, y) = 0.299f * in(x, y, 0) + 0.587f * in(x, y, 1)
                 + 0.114f * in(x, y, 2);

    Func hist("hist");
    hist(x) = 0;
    RDom r(0, in.width(), 0, in.height(), 0, 2);
    Expr bin = cast<int>(clamp(gray(r.x, r.y), 0, 255));
    hist(bin) += 1;

    hist.bound(x, 0, 256);
    // Pick a schedule
    int schedule = atoi(argv[1]);

    Target target = get_target_from_environment();
    if (schedule == 0) {
        hist.compute_root();
        hist.print_loop_nest();
    }

    if (schedule == -1)
        hist.compile_to_file("hist", {in}, target, true);
    else
        hist.compile_to_file("hist", {in}, target);

    return 0;
}
