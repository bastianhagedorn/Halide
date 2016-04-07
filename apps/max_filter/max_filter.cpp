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

    //final.bound(x, 0, in.width()).bound(y, 0, in.height()).bound(c, 0, in.channels());
    final.estimate(x, 0, in.width()).estimate(y, 0, in.height()).estimate(c, 0, in.channels());

    Target target = get_target_from_environment();

    Var tx, xi;
    if (schedule == 0) {
        if (target.has_gpu_feature()) {
            slice_for_radius.compute_root();
            filter_height.compute_root();
            Var xo;

            final
                .split(x, xo, xi, 128)
                .reorder(xi, xo, y, c)
                .gpu_blocks(xo, y, c).gpu_threads(xi);

            vert_log.compute_root()
                .reorder(c, t, x, y)
                .gpu_tile(x, y, 16, 16)
                .update()
                .split(x, xo, xi, 128)
                .reorder(r.x, r.y, xi, xo, c)
                .gpu_blocks(xo, c).gpu_threads(xi);

        } else {
            // These don't matter, just LUTs
            slice_for_radius.compute_root();
            filter_height.compute_root();

            // vert_log.update(1) doesn't have enough parallelism, but I
            // can't figure out how to give it more... Split whole image
            // into slices.

            final.compute_root().split(x, tx, x, 256).reorder(x, y, c, tx).fuse(c, tx, t).parallel(t).vectorize(x, 8);
            vert_log.compute_at(final, t);
            vert_log.vectorize(x, 8);
            vert_log.update().reorder(x, r.x, r.y, c).vectorize(x, 8);
            vert.compute_at(final, y).vectorize(x, 8);
            // 7:42pm. 110ms !!!
        }
    } else {
        vert_log.compute_root();
        vert.compute_root();
        slice_for_radius.compute_root();
        filter_height.compute_root();
        final.compute_root();
    }


    // Run it

    input_im.set(in);
    //radius.set(26);
    Image<float> out(in.width(), in.height(), in.channels());

    if (schedule == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
        //target.set_feature(Halide::Target::CUDA);
    }

    if (schedule == -1 || schedule == -2) {
        final.compile_jit(target, true);
    } else {
        final.compile_jit(target);
    }

    // std::cout << "Running... " << std::endl;
    Buffer buf(out);
    double best = benchmark(5, 5, [&]() { final.realize(buf);}, [&]() { buf.device_sync(); });
    std::cout << "runtime: " << best * 1e3 << std::endl;

    // save_image(out, argv[2]);

    return 0;
}
