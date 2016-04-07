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

    Func hist_rows("hist_rows");
    hist_rows(x, y) = 0;
    RDom rx(0, 1536);
    Expr bin = cast<uint8_t>(clamp(Y(rx, y), 0, 255));
    hist_rows(bin, y) += 1;

    Func hist("hist");
    hist(x) = 0;
    RDom ry(0, 2560);
    hist(x) += hist_rows(x, ry);

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
        if (target.has_gpu_feature()) {
            Y.compute_root().gpu_tile(x, y, 16, 16);
            hist_rows.compute_root().gpu_tile(y, 16).update().gpu_tile(y, 16);
            hist.compute_root().gpu_tile(x, 16).update().gpu_tile(x, 16);
            cdf.compute_root().gpu_single_thread();
            Cr.compute_at(color, Var::gpu_threads());
            Cb.compute_at(color, Var::gpu_threads());
            eq.compute_at(color, Var::gpu_threads());
            color.compute_root()
                .reorder(c, x, y).bound(c, 0, 3).unroll(c)
                .gpu_tile(x, y, 16, 16);
        } else {
            Y.compute_root().parallel(y, 8).vectorize(x, 8);

            hist_rows.compute_root()
                .vectorize(x, 8)
                .parallel(y, 8)
                .update()
                .parallel(y, 8);
            hist.compute_root()
                .vectorize(x, 8)
                .update()
                .reorder(x, ry)
                .vectorize(x, 8)
                .unroll(x, 4)
                .parallel(x)
                .reorder(ry, x);

            cdf.compute_root();
            eq.compute_at(color, x).unroll(x);
            Cb.compute_at(color, x).vectorize(x);
            Cr.compute_at(color, x).vectorize(x);
            color.reorder(c, x, y)
                .bound(c, 0, 3)
                .unroll(c)
                .parallel(y, 8)
                .vectorize(x, 8);
        }
        color.print_loop_nest();
    }

    if (schedule == -2) {
        //target.set_feature(Halide::Target::CUDA);
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }

    auto_build(color, "hist", {in}, target, (schedule == -1 || schedule == -2));

    return 0;
}
