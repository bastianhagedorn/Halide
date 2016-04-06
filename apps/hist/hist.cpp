#include "Halide.h"
#include <stdio.h>
using namespace Halide;
#include "../support/auto_build.h"

int main(int argc, char **argv) {

    ImageParam in(UInt(8), 3);

    Var x, y, c;

    Func Y("Y");
    Y(x, y) = 0.299f * in(x, y, 0) + 0.587f * in(x, y, 1)
              + 0.114f * in(x, y, 2);

    Func Cr("Cr");
    Expr R = in(x, y, 0);
    Cr(x, y) = (R - Y(x, y)) * 0.713f + 128;

    Func Cb("Cb");
    Expr B = in(x, y, 2);
    Cb(x, y) = (B - Y(x, y)) * 0.564f + 128;

    Func hist("hist");
    hist(x) = 0;
    RDom r(0, 1536, 0, 2560);
    Expr bin = cast<uint8_t>(clamp(Y(r.x, r.y), 0, 255));
    hist(bin) += 1;

    Func cdf("cdf");
    cdf(x) = hist(0);
    RDom b(1, 255);
    cdf(b.x) = cdf(b.x - 1) + hist(b.x);

    cdf.bound(x, 0, 256);

    Func eq("equalize");

    Expr cdf_bin = cast<uint8_t>(clamp(Y(x, y), 0 , 255));
    eq(x, y) = clamp(cdf(cdf_bin) * (255.0f/(in.height() * in.width())), 0 , 255);

    Func color("color");
    Expr red = cast<uint8_t>(clamp(eq(x, y) + (Cr(x, y) - 128) * 1.4f, 0, 255));
    Expr green = cast<uint8_t> (clamp(eq(x, y) - 0.343f * (Cb(x, y) - 128) - 0.711f * (Cr(x, y) -128), 0, 255));
    Expr blue = cast<uint8_t> (clamp(eq(x, y) + 1.765f * (Cb(x, y) - 128), 0, 255));
    color(x, y, c) = select(c == 0, red, select(c == 1, green , blue));

    //color.bound(x, 0, 1536).bound(y, 0, 2560).bound(c, 0, 3);
    color.estimate(x, 0, 1536).estimate(y, 0, 2560).estimate(c, 0, 3);

    // Pick a schedule
    int schedule = atoi(argv[1]);

    Target target = get_target_from_environment();
    if (schedule == 0) {
        hist.compute_root();
        cdf.compute_root();
        eq.compute_root().parallel(y);
        color.parallel(y).vectorize(x, 8);
    }

    if (schedule == -2) {
        //target.set_feature(Halide::Target::CUDA);
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }

    auto_build(color, "hist", {in}, target, (schedule == -1 || schedule == -2));

    return 0;
}
