// Circular-support max filter. Does some trickery to get O(r) per pixel for radius r, not O(r^2).

#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>

#include "benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

using std::vector;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n\t./lens_blur in.png out.png schedule\n";
        return 1;
    }
    int schedule = atoi(argv[3]);

    ImageParam input_im(Float(32), 3);
    Param<int> radius;

    Func input = BoundaryConditions::repeat_edge(input_im);

    Var x, y, c, t;

    Expr slices = cast<int>(ceil(log(radius) / logf(2))) + 1;

    // A sequence of vertically-max-filtered versions of the input,
    // each filtered twice as tall as the previous slice. All filters
    // are downward-looking.
    Func vert_log;
    vert_log(x, y, c, t) = undef<float>();
    vert_log(x, y, c, 0) = input(x, y, c);
    RDom r(-radius, input_im.height() + radius, 1, slices-1);
    vert_log(x, r.x, c, r.y) = max(vert_log(x, r.x, c, r.y - 1), vert_log(x, r.x + clamp((1<<(r.y-1)), 0, radius*2), c, r.y - 1));

    // We're going to take a max filter of arbitrary diameter
    // by maxing two samples from its floor log 2 (e.g. maxing two
    // 8-high overlapping samples). This next Func tells us which
    // slice to draw from for a given radius:
    Func slice_for_radius;
    slice_for_radius(t) = cast<int>(floor(log(2*t+1) / logf(2)));

    // Produce every possible vertically-max-filtered version of the image:
    Func vert;
    // t is the blur radius
    Expr slice = clamp(slice_for_radius(t), 0, slices);
    Expr first_sample = vert_log(x, y - t, c, slice);
    Expr second_sample = vert_log(x, y + t + 1 - clamp(1 << slice, 0, 2*radius), c, slice);
    vert(x, y, c, t) = max(first_sample, second_sample);

    Func filter_height;
    RDom dy(0, radius+1);
    filter_height(x) = sum(select(x*x + dy*dy < (radius+0.25f)*(radius+0.25f), 1, 0));

    // Now take an appropriate horizontal max of them at each output pixel
    Func final;
    RDom dx(-radius, 2*radius+1);
    final(x, y, c) = maximum(vert(x + dx, y, c, clamp(filter_height(dx), 0, radius+1)));

    Var tx, xi;
    switch (schedule) {
    case 0:
        vert_log.compute_root();
        vert.compute_root();
        slice_for_radius.compute_root();
        filter_height.compute_root();
        final.compute_root();
        break;
        // 7pm. 3347ms
    case 1:
        // Probably want to inline vert, and just let it sample twice
        // from vert log whenever it needs it instead.
        vert_log.compute_root();
        slice_for_radius.compute_root();
        filter_height.compute_root();
        final.compute_root();
        break;
        // 7:01pm. 4546.13ms. Nope.
    case 2:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // There's a common loop over c.
        vert_log.compute_at(final, c);
        vert.compute_at(final, c);
        final.compute_root();
        break;
        // 7:03pm. 2817ms
    case 3:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        vert_log.compute_at(final, c);

        // Oh, right, vert gets resampled very often by adjacent
        // pixels per scanline of the output.
        vert.compute_at(final, y);
        final.compute_root();
        break;
        // 7:05pm. 1956ms
    case 4:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Vectorize some things...
        vert_log.compute_at(final, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8);;
        break;
        // 7:09pm. 834ms
    case 5:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Move vert closer in to final
        vert_log.compute_at(final, c).vectorize(x, 8);
        final.split(x, x, xi, 32).vectorize(xi, 8);
        vert.compute_at(final, x).vectorize(x, 8);

        break;
        // 7:13pm. 1036ms. Nope. Too much redundant work.
    case 6:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Forgot to vectorize the updates...
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8);;
        break;
        // 7:16pm. 571ms
    case 7:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Better loop ordering?
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8);;
        break;
        // 7:18pm. 410ms
    case 8:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Get some unrolling in the inner loop of final by widening the vectors.
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 16);
        break;
        // 7:21pm. 422ms
    case 9:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // I think if I swap t and x in vert I'll get better reuse of values of vert_log.
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8).reorder(t, x);
        final.compute_root().vectorize(x, 8);
        break;
        // 7:23pm. 1069ms. Oops. I guess this jumps across planes of vert_log all the time.
    case 10:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Add parallelism
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8).parallel(y);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8).parallel(y).parallel(c);
        break;
        // 7:26pm. 155ms
    case 11:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Change granularity
        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8).parallel(y, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8).parallel(y, 8).parallel(c);
        break;
        // 7:26pm. 169ms

    case 12:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8).parallel(y);

        // More parallelism here by slicing up x
        vert_log.update(1).split(x, tx, x, 64).reorder(x, r.x, r.y, tx, c).vectorize(x, 8).parallel(tx);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8).parallel(y).parallel(c);
        break;
        // 7:30pm. 175ms
   case 13:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        vert_log.compute_at(final, c);
        vert_log.update(0).vectorize(x, 8).parallel(y);

        // Change granularity...
        vert_log.update(1).split(x, tx, x, 256).reorder(x, r.x, r.y, tx, c).vectorize(x, 8).parallel(tx);
        vert.compute_at(final, y).vectorize(x, 8);
        final.compute_root().vectorize(x, 8).parallel(y).parallel(c);
        break;
        // 7:31pm. 174ms
   case 14:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        vert_log.compute_root();
        vert_log.update(0).vectorize(x, 8).parallel(y);
        // Maybe move c in and avoid the nested parallelism
        vert_log.update(1).split(x, tx, x, 64).reorder(x, r.x, r.y, tx, c).vectorize(x, 8).parallel(tx);
        vert.compute_at(final, c).vectorize(x, 8);
        final.compute_root().vectorize(x, 8).reorder(x, c, y).parallel(y);
        break;
        // 7:34pm. 177ms

    case 15:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // vert_log.update(1) doesn't have enough parallelism, but I
        // can't figure out how to give it more... Split whole image
        // into slices.

        final.compute_root().split(x, tx, x, 256).reorder(x, y, c, tx).fuse(c, tx, t).parallel(t).vectorize(x, 8);
        vert_log.compute_at(final, t);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);

        break;
        // 7:42pm. 110ms !!!

    case 16:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Try different granularity...

        final.compute_root().split(x, tx, x, 64).reorder(x, y, c, tx).fuse(c, tx, t).parallel(t).vectorize(x, 8);
        vert_log.compute_at(final, t);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);

        break;
        // 7:43pm. 133ms
    case 17:
        // These don't matter, just LUTs
        slice_for_radius.compute_root();
        filter_height.compute_root();

        // Try different granularity...

        final.compute_root().split(x, tx, x, 512).reorder(x, y, c, tx).fuse(c, tx, t).parallel(t).vectorize(x, 8);
        vert_log.compute_at(final, t);
        vert_log.update(0).vectorize(x, 8);
        vert_log.update(1).reorder(x, r.x, r.y, c).vectorize(x, 8);
        vert.compute_at(final, y).vectorize(x, 8);

        break;
        // 7:44pm. 130ms. Looks like I had it right with 256.

        // 7:50pm. Can't think of anything else to do. Calling it here. Case 15 was best.

    default:
        break;
    }


    // Run it

    Image<float> in = load_image(argv[1]);
    input_im.set(in);
    radius.set(26);
    Image<float> out(in.width(), in.height(), in.channels());
    Target target = get_target_from_environment();
    if (schedule == -1) {
        final.compile_jit(target, true);
    } else {
        final.compile_jit(target);
    }



    std::cout << "Running... " << std::endl;
    double best = benchmark(3, 3, [&]() { final.realize(out); });
    std::cout << " took " << best * 1e3 << " msec." << std::endl;

    save_image(out, argv[2]);

    return 0;
}
