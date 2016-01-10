#include <stdio.h>
#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {

    Func f("f"), g("g"), h("h"); Var x("x"), y("y");

    h(x, y) = x + y;
    g(x, y) = (h(x-1, y-1) + h(x+1, y+1))/2;
    f(x, y) = (g(x-1, y-1) + g(x+1, y+1))/2;

    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
    f.tile(x, y, xo, yo, xi, yi, 8, 8);
    f.compute_at(0);
    g.compute_at(2);
    h.compute_at(99); // should become innermost

    f.print_loop_nest();

    printf("Success!\n");
    return 0;
}
