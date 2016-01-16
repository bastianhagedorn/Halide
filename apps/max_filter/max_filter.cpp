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
