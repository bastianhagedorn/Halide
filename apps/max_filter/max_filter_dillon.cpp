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

    Image<float> in = load_image(argv[1]);

    ImageParam input_im(Float(32), 3);
    const int radius = 26;

    Func input = BoundaryConditions::repeat_edge(input_im);

    Var x, y, c, t;

    const int slices = (int)(ceilf(logf(radius) / logf(2))) + 1;

    // A sequence of vertically-max-filtered versions of the input,
    // each filtered twice as tall as the previous slice. All filters
    // are downward-looking.
    Func vert_log;
    vert_log(x, y, c, t) = input(x, y, c);
    RDom r(-radius, in.height() + radius, 1, slices-1);
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

    final.bound(x, 0, in.width()).bound(y, 0, in.height()).bound(c, 0, in.channels());

    Var tx, xi;
    Var ty;
    switch (schedule) {
    case 0:
        vert_log.compute_root();
        vert.compute_root();
        slice_for_radius.compute_root();
        filter_height.compute_root();
        final.compute_root();
        // Time: 0 min
        // Runtime: 4500 ms
        break;
    case 1:
        // First let's just parallelize/vectorize. It makes it easier
        // to see the relative improvement of other optimizations.
        slice_for_radius.compute_root();
        filter_height.compute_root();

        vert_log.compute_root();
        vert_log.update(0)
            .reorder(r.x, c, x)
            .vectorize(x, 8)
            .parallel(x, 4);

        vert.compute_root()
            .reorder(t, x, c, y)
            .parallel(y, 8)
            .vectorize(x, 8);
        final.compute_root()
            .reorder(c, x, y)
            .vectorize(x, 8)
            .parallel(y, 8);
        // Time: 5 min
        // Runtime: 1079 ms
        break;
    case 2:
        // Let's try inlining these LUT-ish Funcs.
        slice_for_radius.compute_root();

        vert_log.compute_root();
        vert_log.update(0)
            .reorder(r.x, c, x)
            .vectorize(x, 8)
            .parallel(x, 4);

        vert.compute_root()
            .reorder(t, x, c, y)
            .parallel(y, 8)
            .vectorize(x, 8);
        final.compute_root()
            .reorder(c, x, y)
            .vectorize(x, 8)
            .parallel(y, 8);
        // Time: 6 min
        // Runtime: 3142 ms (Wow, bad idea!)
        break;
    case 3:
        // How about the other LUT?
        filter_height.compute_root();

        vert_log.compute_root();
        vert_log.update(0)
            .reorder(r.x, c, x)
            .vectorize(x, 8)
            .parallel(x, 4);

        vert.compute_root()
            .reorder(t, x, c, y)
            .parallel(y, 8)
            .vectorize(x, 8);
        final.compute_root()
            .reorder(c, x, y)
            .vectorize(x, 8)
            .parallel(y, 8);
        // Time: 6 min
        // Runtime: 1274 ms (Still slower, keep them at root).
        break;
    case 4:
        // It might be a bad idea to reorder c innermost in the
        // output. We could compute each channel entirely one at a
        // time.
        slice_for_radius.compute_root();
        filter_height.compute_root();

        final.compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);

        vert_log.compute_at(final, c);
        vert_log.update(0)
            .reorder(r.x, x)
            .vectorize(x, 8)
            .parallel(x, 4);

        vert.compute_at(final, c)
            .reorder(t, x, y)
            .parallel(y, 8)
            .vectorize(x, 8);
        // Time: 8 min
        // Runtime: 580 ms (Wow! I was not expecting that. Must be a weird cache near the size of one channel of the imgae?
        break;
    case 5:
        // Tile the output?
        slice_for_radius.compute_root();
        filter_height.compute_root();

        final.compute_root()
            .tile(x, y, tx, ty, x, y, 64, 64)
            .vectorize(x, 8)
            .parallel(ty);

        vert_log.compute_at(final, tx);
        vert_log.update(0)
            .reorder(r.x, x)
            .vectorize(x, 8);

        vert.compute_at(final, tx)
            .reorder(t, x, y)
            .vectorize(x, 8);
        // Time: 10 min
        // Runtime: 1939 ms
        break;
    case 6:
        // I can't find a way to tile this with reasonable efficiency. Back to schedule 4.
        // I just noticed we should be computing the vertical blur at each scanline.
        // I got stalled here for a while tracking down a boneheaded
        // mistake in my schedule making things slower than they
        // should be (forgot to remove a parallel directive from the
        // prior schedule that was nested and now very small
        // granularity).
        slice_for_radius.compute_root();
        filter_height.compute_root();

        final.compute_root()
            .vectorize(x, 8)
            .parallel(y);

        vert_log.compute_at(final, c);
        vert_log.update(0)
            .reorder(r.x, x)
            .vectorize(x, 8)
            .parallel(x, 4);

        vert.compute_at(final, y)
            .reorder(t, x, y)
            .vectorize(x, 8);
        // Time: 25 min
        // Runtime: 275 ms
        break;
    case 7:
        // Let's try splitting the output in x.
        slice_for_radius.compute_root();
        filter_height.compute_root();

        final.compute_root()
            .split(x, tx, x, 64)
            .reorder(x, y, c, tx)
            .vectorize(x, 8)
            .parallel(tx);

        vert_log.compute_at(final, tx);
        vert_log.update(0)
            .reorder(r.x, x)
            .vectorize(x, 8);

        vert.compute_at(final, y)
            .reorder(t, x, y)
            .vectorize(x, 8);
        // Time: 30 min
        // Runtime: 284 ms
        break;
    case 8:
        // That was a little slower, but a bit easier to work with, and avoids large buffers at root (memory consumption would be high).
        // Playing with the tile size yielded a bit of an improvement.
        slice_for_radius.compute_root();
        filter_height.compute_root();

        final.compute_root()
            .split(x, tx, x, 128)
            .reorder(x, y, c, tx)
            .vectorize(x, 8)
            .parallel(tx);

        vert_log.compute_at(final, tx);
        vert_log.update(0)
            .reorder(r.x, x)
            .vectorize(x, 8);

        vert.compute_at(final, y)
            .reorder(t, x, y)
            .vectorize(x, 8);
        // Time: 35 min
        // Runtime: 218 ms
        break;
    default:
        break;
    }

    final.compile_to_lowered_stmt("final.html", final.infer_arguments(), HTML, get_jit_target_from_environment());

    // Run it

    input_im.set(in);
    //radius.set(26);
    Image<float> out(in.width(), in.height(), in.channels());
    Target target = get_target_from_environment();
    if (schedule == -1) {
        final.compile_jit(target, true);
    } else {
        final.compile_jit(target);
    }

    // std::cout << "Running... " << std::endl;
    double best = benchmark(3, 3, [&]() { final.realize(out); });
    std::cout << "runtime: " << best * 1e3 << std::endl;

    // save_image(out, argv[2]);

    return 0;
}
