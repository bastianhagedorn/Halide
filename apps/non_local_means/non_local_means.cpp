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

    ImageParam input(Float(32), 3);
    Param<float> sigma;
    Var x("x"), y("y"), z("z"), c("c");

    Expr inv_sigma_sq = 1.0f/(sigma*sigma);

    // Add a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Define the difference images.
    Var dx("dx"), dy("dy");
    Func d("d");
    d(x, y, dx, dy, c) = pow(clamped(x, y, c) - clamped(x + dx, y + dy, c), 2);

    // Find the patch differences by blurring the difference images.
    RDom patch_dom(-patch_size/2, patch_size);
    Func blur_d_y("blur_d_y");
    blur_d_y(x, y, dx, dy, c) = sum(d(x, y + patch_dom, dx, dy, c));

    Func blur_d("blur_d");
    blur_d(x, y, dx, dy, c) = sum(blur_d_y(x + patch_dom, y, dx, dy, c))/(search_area*search_area);

    // Compute the weights from the patch differences.
    Func w("w");
    //RDom channels(input.min(2), input.extent(2));
    RDom channels(0, 3);
    w(x, y, dx, dy) = fast_exp(-sum(blur_d(x, y, dx, dy, channels))*inv_sigma_sq);

    // Define a reduction domain for the search area.
    RDom s_dom(-search_area/2, search_area, -search_area/2, search_area);

    // Compute the reciprocal of the sum of the weights, so we can normalize the result below.
    Func normalize("normalize");
    normalize(x, y) = fast_inverse(sum(w(x, y, s_dom.x, s_dom.y)) + 1e-6f);

    // Compute the weighted sum of the pixels in the search area.
    Func non_local_means("non_local_means");
    non_local_means(x, y, c) =
        clamp(normalize(x, y)*sum(clamped(x + s_dom.x, y + s_dom.y, c)*w(x, y, s_dom.x, s_dom.y)), 0.0f, 1.0f);

    // Schedule.
    Target target = get_target_from_environment();

    // Require 3 channels for input and output.
    input.set_min(2, 0).set_extent(2, 3);
    non_local_means.output_buffer().set_min(2, 0).set_extent(2, 3);
    non_local_means.bound(x, 0, 192).bound(y, 0, 320).bound(c, 0, 3);

    /*
    d.compute_root();
    blur_d_y.compute_root();
    blur_d.compute_root();
    non_local_means.compute_root();
    */

    non_local_means.compile_to_file("non_local_means", {sigma, input}, target, true);

    return 0;
}
