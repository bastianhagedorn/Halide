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

    Expr inv_sigma_sq = 1.0f/(sigma*sigma*patch_size*patch_size);

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
    non_local_means_sum(x, y, c) += w(x, y, s_dom.x, s_dom.y) * clamped_with_alpha(x, y, c);

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
        break;
    }
    case 1: {
        // Put more schedules here.
    }
    default:
        break;
    }





    non_local_means.print_loop_nest();

    non_local_means.compile_to_file("non_local_means", {sigma, input}, target, schedule == -1);

    return 0;
}
