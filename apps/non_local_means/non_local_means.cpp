#include "Halide.h"
#include <stdio.h>
#include "../support/auto_build.h"

using namespace Halide;

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: non_local_means <patch_size> <search_area>\n");
        return 0;
    }

    // This implements the basic description of non-local means found at https://en.wikipedia.org/wiki/Non-local_means.

    int patch_size = atoi(argv[1]);
    int search_area = atoi(argv[2]);
    int schedule = 999;
    if (argc > 3) { schedule = atoi(argv[3]); }

    ImageParam input(Float(32), 3);
    Param<float> sigma;
    Var x("x"), y("y"), z("z"), c("c");

    Expr inv_sigma_sq = -1.0f/(sigma*sigma*patch_size*patch_size);

    // Add a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Define the difference images.
    Var dx("dx"), dy("dy");
    Func dc("d");
    dc(x, y, dx, dy, c) = pow(clamped(x, y, c) - clamped(x + dx, y + dy, c), 2);

    // Sum across color channels
    RDom channels(0, 3);
    Func d("d");
    d(x, y, dx, dy) = sum(dc(x, y, dx, dy, channels));

    // Find the patch differences by blurring the difference images.
    RDom patch_dom(-patch_size/2, patch_size);
    Func blur_d_y("blur_d_y");
    blur_d_y(x, y, dx, dy) = sum(d(x, y + patch_dom, dx, dy));

    Func blur_d("blur_d");
    blur_d(x, y, dx, dy) = sum(blur_d_y(x + patch_dom, y, dx, dy));

    // Compute the weights from the patch differences.
    Func w("w");
    w(x, y, dx, dy) = fast_exp(blur_d(x, y, dx, dy)*inv_sigma_sq);

    // Add an alpha channel
    Func clamped_with_alpha;
    clamped_with_alpha(x, y, c) = select(c == 0, clamped(x, y, 0),
                                         c == 1, clamped(x, y, 1),
                                         c == 2, clamped(x, y, 2),
                                         1.0f);

    // Define a reduction domain for the search area.
    RDom s_dom(-search_area/2, search_area, -search_area/2, search_area);

    // Compute the sum of the pixels in the search area.
    Func non_local_means_sum("non_local_means_sum");
    non_local_means_sum(x, y, c) += w(x, y, s_dom.x, s_dom.y) * clamped_with_alpha(x + s_dom.x, y + s_dom.y, c);

    Func non_local_means("non_local_means");
    non_local_means(x, y, c) =
        clamp(non_local_means_sum(x, y, c) / non_local_means_sum(x, y, 3), 0.0f, 1.0f);

    // Schedule.
    Target target = get_target_from_environment();

    if (schedule == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }

    // Require 3 channels for input and output.
    input.set_min(2, 0).set_extent(2, 3);
    non_local_means.output_buffer().set_min(2, 0).set_extent(2, 3);
    // non_local_means.bound(x, 0, 192).bound(y, 0, 320).bound(c, 0, 3);
    non_local_means.estimate(x, 0, 192).estimate(y, 0, 320).estimate(c, 0, 3);

    Var tx("tx"), ty("ty"), xi, yi;

    if (schedule == 0) {
        if (target.has_gpu_feature()) {
            non_local_means.compute_root()
                .reorder(c, x, y).unroll(c)
                .gpu_tile(x, y, 16, 8);

            d.compute_at(non_local_means_sum, s_dom.x)
                .tile(x, y, xi, yi, 2, 2)
                .unroll(xi)
                .unroll(yi)
                .gpu_threads(x, y);

            blur_d_y.compute_at(non_local_means_sum, s_dom.x)
                .unroll(x, 2).gpu_threads(x, y);

            blur_d.compute_at(non_local_means_sum, s_dom.x)
                .gpu_threads(x, y);

            non_local_means_sum.compute_at(non_local_means, Var::gpu_blocks())
                .gpu_threads(x, y)
                .update()
                .reorder(x, y, c, s_dom.x, s_dom.y)
                .gpu_threads(x, y);

        } else {
            non_local_means.compute_root()
                .reorder(c, x, y)
                .tile(x, y, tx, ty, x, y, 16, 8)
                .parallel(ty)
                .vectorize(x, 8);

            blur_d_y.compute_at(non_local_means, tx)
                .reorder(y, x)
                .vectorize(x, 8);
            d.compute_at(blur_d_y, x)
                .vectorize(x, 8);
            non_local_means_sum.compute_at(non_local_means, x)
                .reorder(c, x, y)
                .bound(c, 0, 4).unroll(c)
                .vectorize(x, 8);
            non_local_means_sum.update(0)
                .reorder(c, x, y, s_dom.x, s_dom.y)
                .unroll(c)
                .vectorize(x, 8);
            blur_d.compute_at(non_local_means_sum, x)
                .vectorize(x, 8);
        }
    }

    /*
    switch (schedule) {
    case 0:  {
        d.compute_root();
        blur_d_y.compute_root();
        blur_d.compute_root();
        non_local_means.compute_root();
        break;
    }
    case 1: { // ANDREW
        // blur_d_y and blur_d should share a single loop over s_dom.x
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 4).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, x, y, s_dom.y).unroll(c).parallel(y, 4).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(blur_d, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.y).parallel(y, 8).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:49 am. 6.1296 ms. I think I've converged.
        break;
    }
    case 2: { // DILLON
        // Back to tiling. Let's play with the tile size?
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 16, 8)
            .parallel(ty)
            .vectorize(x, 8);

        blur_d_y.compute_at(non_local_means, tx)
            .reorder(y, x)
            .vectorize(x, 8);
        d.compute_at(blur_d_y, x)
            .vectorize(x, 8);
        non_local_means_sum.compute_at(non_local_means, x)
            .reorder(c, x, y)
            .bound(c, 0, 4).unroll(c)
            .vectorize(x, 8);
        non_local_means_sum.update(0)
            .reorder(c, x, y, s_dom.x, s_dom.y)
            .unroll(c)
            .vectorize(x, 8);
        blur_d.compute_at(non_local_means_sum, x)
            .vectorize(x, 8);

        // Time: 43 min
        // Runtime: 4.9 ms (Wow, 16x8 tiles is shockingly small, there is a ton of redundant work! That suggests this schedule might be badly designed from the outset. Perhaps many global passes over the image would be better. However, I'm having a hard time thinking of how to do this without excessive memory usage.)
        break;
    }
    default: {
        break;
    }
    }
    */

    auto_build(non_local_means, "non_local_means", {sigma, input}, target, schedule == -1 || schedule == -2);

    // non_local_means.print_loop_nest();

    return 0;
}
