#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: non_local_means <patch_size> <search_area>\n");
        return 0;
    }

    // This implements the basic description of non-local means found at https://en.wikipedia.org/wiki/Non-local_means.

    int patch_size = atoi(argv[1]);
    int search_area = atoi(argv[2]);
    int schedule = atoi(argv[3]);

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

    // Require 3 channels for input and output.
    input.set_min(2, 0).set_extent(2, 3);
    non_local_means.output_buffer().set_min(2, 0).set_extent(2, 3);
    non_local_means.bound(x, 0, 192).bound(y, 0, 320).bound(c, 0, 3);

    switch (schedule) {
    case 0:  {
        d.compute_root();
        blur_d_y.compute_root();
        blur_d.compute_root();
        non_local_means.compute_root();
        // Start at 10:10 am. Baseline = 284.841 ms
        break;
    }
    case 1: {
        // Move the loop over the RDom outermost, and compute the
        // blurred errors inside it.
        non_local_means_sum.compute_root().bound(c, 0, 4);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c);
        w.compute_at(non_local_means_sum, x);
        blur_d.compute_at(non_local_means_sum, y);
        blur_d_y.compute_at(non_local_means_sum, y);
        d.compute_at(non_local_means_sum, s_dom.x);
        // 10:14 am. 53.11 ms
        break;
    }
    case 2: {
        // Study the pseudocode for a while. The only problem I see is
        // the loop order of the normalization step.
        non_local_means_sum.compute_root().bound(c, 0, 4);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c);
        w.compute_at(non_local_means_sum, x);
        blur_d.compute_at(non_local_means_sum, y);
        blur_d_y.compute_at(non_local_means_sum, y);
        d.compute_at(non_local_means_sum, s_dom.x);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        // 10:24 am. 52.857 ms
        break;
    }
    case 3: {
        // Parallelize and vectorize. We don't parallelize the outer
        // loop because it's the reduction.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 8);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c).parallel(y, 8).vectorize(x, 8);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 8);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 8);
        d.compute_at(non_local_means_sum, s_dom.x).parallel(y, 8).vectorize(x, 8);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 8);
        // 10:27 am. 7.849 ms
        break;
    }
    case 4: {
        // It's a pretty small image. Change parallelism granularity and see what happens.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 4).vectorize(x, 8);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c).parallel(y, 4).vectorize(x, 8);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 8);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 8);
        d.compute_at(non_local_means_sum, s_dom.x).parallel(y, 4).vectorize(x, 8);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 4).vectorize(x, 8);
        // 10:32 am. 8.222 ms
        break;
    }
    case 5: {
        // Change it in the other direction...
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 16).vectorize(x, 8);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c).parallel(y, 16).vectorize(x, 8);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 8);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 8);
        d.compute_at(non_local_means_sum, s_dom.x).parallel(y, 16).vectorize(x, 8);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 16).vectorize(x, 8);
        // 10:33 am. 7.9624 ms
        break;
    }
    case 6: {
        // 8 was fine. Now try changing the vector width
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c).parallel(y, 8).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.x).parallel(y, 8).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:34 am. 7.5598 ms
        break;
    }
    case 7: {
        // Push it further...
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 32);
        non_local_means_sum.update().reorder(c, x, y, s_dom.x, s_dom.y).unroll(c).parallel(y, 8).vectorize(x, 32);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 32);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 32);
        d.compute_at(non_local_means_sum, s_dom.x).parallel(y, 8).vectorize(x, 32);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 32);
        // 10:36 am. 8.014 ms
        break;
    }
    case 8: {
        // We assumed that the loop over the accumulation RDom should be outermost. Maybe that's not the case. Let's just schedule the whole thing as a stencil to sanity check.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, s_dom.y, x, y).unroll(c).parallel(y, 8).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:40 am. 10.18ms
        break;
    }
    case 9: {
        // Actually maybe we want the RDom loop over y outside, but the RDom loop over x inside, to minimize the vertical footprint of the stencil. Also I think we want d outside the loop over y, so it isn't redundantly recomputed.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, x, y, s_dom.y).unroll(c).parallel(y, 8).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.y).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:41am. 14.8ms. That doesn't seem right...
        break;
    }
    case 10: {
        // I dropped the parallelism on d.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 8).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, x, y, s_dom.y).unroll(c).parallel(y, 8).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.y).parallel(y, 8).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:43 am. 6.6 ms! Progress.
        break;
    }
    case 11: {
        // Now that there's more work inside the loop over y, maybe I should lower the granularity.
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y, 4).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, x, y, s_dom.y).unroll(c).parallel(y, 4).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.y).parallel(y, 8).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:45 am. 6.2588 ms! Progress.
        break;
    }
    case 12: {
        // Drop it down to 1
        non_local_means_sum.compute_root().bound(c, 0, 4).parallel(y).vectorize(x, 16);
        non_local_means_sum.update().reorder(c, s_dom.x, x, y, s_dom.y).unroll(c).parallel(y).vectorize(x, 16);
        w.compute_at(non_local_means_sum, x).vectorize(x);
        blur_d.compute_at(non_local_means_sum, y).vectorize(x, 16);
        blur_d_y.compute_at(non_local_means_sum, y).vectorize(x, 16);
        d.compute_at(non_local_means_sum, s_dom.y).parallel(y, 8).vectorize(x, 16);
        non_local_means.reorder(c, x, y).bound(c, 0, 3).unroll(c).parallel(y, 8).vectorize(x, 16);
        // 10:46 am. 6.48 ms
        break;
    }
    case 13: {
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
    default:
        break;
    }





    non_local_means.print_loop_nest();

    non_local_means.compile_to_file("non_local_means", {sigma, input}, target, schedule == -1);

    return 0;
}
