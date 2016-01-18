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

    Var tx("tx"), ty("ty");

    switch (schedule) {
    case 0:  {
        d.compute_root();
        blur_d_y.compute_root();
        blur_d.compute_root();
        non_local_means.compute_root();
        non_local_means_sum.compute_root();
        // Time: 0 min
        // Runtime: 135 ms
        break;
    }
    case 1: {
        // Tile output, compute all other funcs at tiles.
        // Reorder c innermost.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64);

        d.compute_at(non_local_means, tx);
        blur_d_y.compute_at(non_local_means, tx);
        blur_d.compute_at(non_local_means, tx);
        non_local_means_sum.compute_at(non_local_means, tx);
        // Time: 5 min
        // Runtime: 147 ms (a bit slower, but it's a stepping stone towards more optimizations.
        break;
    }
    case 2: {
        // The summation doesn't need to be computed in tiles, but it
        // does need to be computed once per pixel.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64);

        d.compute_at(non_local_means, tx);
        blur_d_y.compute_at(non_local_means, tx);
        blur_d.compute_at(non_local_means, tx);
        non_local_means_sum.compute_at(non_local_means, x);
        // Time: 9 min
        // Runtime: 131 ms
        break;
    }
    case 3: {
        // Let's parallelize over tiles.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty);

        d.compute_at(non_local_means, tx);
        blur_d_y.compute_at(non_local_means, tx);
        blur_d.compute_at(non_local_means, tx);
        non_local_means_sum.compute_at(non_local_means, x);
        // Time: 10 min
        // Runtime: 55.7 ms (not as much improvement as I expected, redundant compute is excessive?)
        break;
    }
    case 4: {
        // Reorder c innermost for non_local_means_sum, and unroll it to get simplification with the 1.0 weight.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty);

        d.compute_at(non_local_means, tx);
        blur_d_y.compute_at(non_local_means, tx);
        blur_d.compute_at(non_local_means, tx);
        non_local_means_sum.compute_at(non_local_means, x)
            .reorder(c, x, y)
            .bound(c, 0, 4).unroll(c);
        non_local_means_sum.update(0)
            .reorder(c, s_dom.x, s_dom.y, x, y)
            .unroll(c);
        // Time: 15 min
        // Runtime: 31 ms
        break;
    }
    case 5: {
        // Let's try reordering the reduction so we can compute blur_d inside it.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty);

        d.compute_at(non_local_means, tx);
        blur_d_y.compute_at(non_local_means, tx);
        non_local_means_sum.compute_at(non_local_means, x)
            .reorder(c, x, y)
            .bound(c, 0, 4).unroll(c);
        non_local_means_sum.update(0)
            .reorder(c, x, y, s_dom.x, s_dom.y)
            .unroll(c);
        blur_d.compute_at(non_local_means_sum, x);

        // Time: 15 min
        // Runtime: 27 ms
        break;
    }
    case 6: {
        // Vectorize!
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty)
            .vectorize(x, 8);

        d.compute_at(non_local_means, tx)
            .vectorize(x, 8);
        blur_d_y.compute_at(non_local_means, tx)
            .vectorize(x, 8);
        non_local_means_sum.compute_at(non_local_means, x)
            .reorder(c, x, y)
            .bound(c, 0, 4).unroll(c)
            .vectorize(x, 8);
        non_local_means_sum.update(0)
            .reorder(c, x, y, s_dom.x, s_dom.y)
            .unroll(c)
            .vectorize(x, 8);
        blur_d.compute_at(non_local_means_sum, x);

        // Time: 23 min
        // Runtime: 13 ms
        break;
    }
    case 7: {
        // There are too many passes over the reduction domain to compute blur_d_y. Let's try to minimize the distance between producer (d) and consumer (blur_d_y).
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty)
            .vectorize(x, 8);

        // It turns out that to do this, we need to compute d in vertical strips consumed by blur_d_y.
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
        blur_d.compute_at(non_local_means_sum, x);

        // Time: 35 min
        // Runtime: 12.5 ms (Barely made any difference :( )
        break;
    }
    case 8: {
        // I just noticed that blur_d is not vectorized (looking at the HTML stmt). Fix that.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
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

        // Time: 37 min
        // Runtime: 8.8 ms
        break;
    }
    case 9: {
        // We didn't get a linear speedup earlier when parallelizing. Maybe it is a bad idea to tile, and we should only split into horizontal strips.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .split(y, ty, y, 64)
            .parallel(ty)
            .vectorize(x, 8);

        blur_d_y.compute_at(non_local_means, ty)
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

        // Time: 39 min
        // Runtime: 9 ms (probably noise, didn't help much.
        break;
    }
    case 10: {
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
    case 11: {
        // Let's try computing some things globally instead, to minimize redundant work.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 32, 16)
            .parallel(ty)
            .vectorize(x, 8);

        blur_d_y.compute_root()
            .reorder(y, x)
            .vectorize(x, 8)
            .parallel(dy);
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

        // Time: 44 min
        // Runtime: 8.7 ms
        break;
    }
    case 12: {
        // That was slower. Let's try two levels of tiling.
        non_local_means.compute_root()
            .reorder(c, x, y)
            .tile(x, y, tx, ty, x, y, 64, 64)
            .parallel(ty)
            .vectorize(x, 8);

        blur_d_y.compute_at(non_local_means, tx)
            .tile(x, y, tx, ty, x, y, 16, 8)
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

        // Time: 47 min
        // Runtime: 14.15 ms
        // Things are going downhill. Schedule 10 is the winner.
        break;
    }
    default:
        break;
    }





    non_local_means.print_loop_nest();
    non_local_means.compile_to_lowered_stmt("non_local_means.html", {sigma, input}, HTML);

    non_local_means.compile_to_file("non_local_means", {sigma, input}, target, schedule == -1);

    return 0;
}
