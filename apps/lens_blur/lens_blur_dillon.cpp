#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>

#include "benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

using std::vector;

Var x, y, z, c;

// Downsample with a 1 3 3 1 filter
Func downsample(Func f) {
    Func downx, downy;

    downx(x, y, _) = (f(2*x-1, y, _) + 3.0f * (f(2*x, y, _) + f(2*x+1, y, _)) + f(2*x+2, y, _)) / 8.0f;
    downy(x, y, _) = (downx(x, 2*y-1, _) + 3.0f * (downx(x, 2*y, _) + downx(x, 2*y+1, _)) + downx(x, 2*y+2, _)) / 8.0f;

    return downy;
}

// Upsample using bilinear interpolation
Func upsample(Func f) {
    Func upx, upy;

    upx(x, y, _) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, _) + 0.75f * f(x/2, y, _);
    upy(x, y, _) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), _) + 0.75f * upx(x, y/2, _);

    return upy;

}


int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n\t./lens_blur left.png right.png out.png\n";
        return 1;
    }

    // The number of displacements to consider
    const int slices = 32;

    // The depth to focus on
    const int focus_depth = 13;

    // The increase in blur radius with misfocus depth
    const float blur_radius_scale = 0.5;

    // The number of samples of the aperture to use
    const int aperture_samples = 32;

    const int maximum_blur_radius = std::max(slices - focus_depth, focus_depth) * blur_radius_scale;

    ImageParam left_im(UInt(8), 3), right_im(UInt(8), 3);

    Func left = BoundaryConditions::repeat_edge(left_im);
    Func right = BoundaryConditions::repeat_edge(right_im);

    Func diff;
    diff(x, y, z, c) = min(absd(left(x, y, c), right(x + 2*z, y, c)),
                           absd(left(x, y, c), right(x + 2*z + 1, y, c)));

    Func cost;
    cost(x, y, z) = (pow(cast<float>(diff(x, y, z, 0)), 2) +
                     pow(cast<float>(diff(x, y, z, 1)), 2) +
                     pow(cast<float>(diff(x, y, z, 2)), 2));

    // Compute confidence of cost estimate at each pixel by taking the
    // variance across the stack.
    Func cost_confidence;
    {
        RDom r(0, slices);
        Expr a = sum(pow(cost(x, y, r), 2)) / slices;
        Expr b = pow(sum(cost(x, y, r) / slices), 2);
        cost_confidence(x, y) = a - b;
    }

    // Do a push-pull thing to blur the cost volume with an
    // exponential-decay type thing to inpaint over regions with low
    // confidence.
    Func cost_pyramid_push[8];
    cost_pyramid_push[0](x, y, z, c) =
        select(c == 0, cost(x, y, z) * cost_confidence(x, y), cost_confidence(x, y));
    //Expr w = left_im.width(), h = left_im.height();
    int w = 992;
    int h = 1024;
    for (int i = 1; i < 8; i++) {
        cost_pyramid_push[i](x, y, z, c) = downsample(cost_pyramid_push[i-1])(x, y, z, c);
        w /= 2;
        h /= 2;
        cost_pyramid_push[i] = BoundaryConditions::repeat_edge(cost_pyramid_push[i], {{0, w}, {0, h}});
    }

    Func cost_pyramid_pull[8];
    cost_pyramid_pull[7](x, y, z, c) = cost_pyramid_push[7](x, y, z, c);
    for (int i = 6; i >= 0; i--) {
        cost_pyramid_pull[i](x, y, z, c) = lerp(upsample(cost_pyramid_pull[i+1])(x, y, z, c),
                                                cost_pyramid_push[i](x, y, z, c),
                                                0.5f);
    }

    Func filtered_cost;
    filtered_cost(x, y, z) = (cost_pyramid_pull[0](x, y, z, 0) /
                              cost_pyramid_pull[0](x, y, z, 1));

    // Assume the minimum cost slice is the correct depth.
    Func depth;
    {
        RDom r(0, slices);
        depth(x, y) = argmin(filtered_cost(x, y, r))[0];
    }

    Func bokeh_radius;
    bokeh_radius(x, y) = abs(depth(x, y) - focus_depth) * blur_radius_scale;

    Func bokeh_radius_squared;
    bokeh_radius_squared(x, y) = pow(bokeh_radius(x, y), 2);

    // Take a max filter of the bokeh radius to determine the
    // worst-case bokeh radius to consider at each pixel. Makes the
    // sampling more efficient below.
    Func worst_case_bokeh_radius_y, worst_case_bokeh_radius;
    {
        RDom r(-maximum_blur_radius, 2*maximum_blur_radius+1);
        worst_case_bokeh_radius_y(x, y) = maximum(bokeh_radius(x, y + r));
        worst_case_bokeh_radius(x, y) = maximum(worst_case_bokeh_radius_y(x + r, y));
    }

    Func input_with_alpha;
    input_with_alpha(x, y, c) = select(c == 0, cast<float>(left(x, y, 0)),
                                       c == 1, cast<float>(left(x, y, 1)),
                                       c == 2, cast<float>(left(x, y, 2)),
                                       255.0f);

    // Render a blurred image
    Func output;
    output(x, y, c) = input_with_alpha(x, y, c);

    // The sample locations are a random function of x, y, and sample
    // number (not c).
    Expr worst_radius = worst_case_bokeh_radius(x, y);
    Expr sample_u = (random_float() - 0.5f) * 2 * worst_radius;
    Expr sample_v = (random_float() - 0.5f) * 2 * worst_radius;
    sample_u = clamp(cast<int>(sample_u), -maximum_blur_radius, maximum_blur_radius);
    sample_v = clamp(cast<int>(sample_v), -maximum_blur_radius, maximum_blur_radius);
    Func sample_locations;
    sample_locations(x, y, z) = {sample_u, sample_v};

    RDom s(0, aperture_samples);
    sample_u = sample_locations(x, y, z)[0];
    sample_v = sample_locations(x, y, z)[1];
    Expr sample_x = x + sample_u, sample_y = y + sample_v;
    Expr r_squared = sample_u * sample_u + sample_v * sample_v;

    // We use this sample if it's from a pixel whose bokeh influences
    // this output pixel. Here's a crude approximation that ignores
    // some subtleties of occlusion edges and inpaints behind objects.
    Expr sample_is_within_bokeh_of_this_pixel =
        r_squared < bokeh_radius_squared(x, y);

    Expr this_pixel_is_within_bokeh_of_sample =
        r_squared < bokeh_radius_squared(sample_x, sample_y);

    Expr sample_is_in_front_of_this_pixel =
        depth(sample_x, sample_y) < depth(x, y);

    Func sample_weight;
    sample_weight(x, y, z) =
        select((sample_is_within_bokeh_of_this_pixel ||
                sample_is_in_front_of_this_pixel) &&
               this_pixel_is_within_bokeh_of_sample,
               1.0f, 0.0f);

    sample_x = x + sample_locations(x, y, s)[0];
    sample_y = y + sample_locations(x, y, s)[1];
    output(x, y, c) += sample_weight(x, y, s) * input_with_alpha(sample_x, sample_y, c);

    // Normalize
    Func final("final");
    final(x, y, c) = output(x, y, c) / output(x, y, 3);

    Image<uint8_t> in_l = load_image(argv[1]);
    Image<uint8_t> in_r = load_image(argv[2]);
    final.bound(x, 0, in_l.width()).bound(y, 0, in_l.height()).bound(c, 0, 3);

    // std::cerr << in_l.width() << "," << in_l.height() << std::endl;
    Var tx, ty;
    int schedule = atoi(argv[4]);
    switch(schedule) {
    case 0: {
        cost.compute_root();
        cost_confidence.compute_root();

        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        depth.compute_root();

        input_with_alpha.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root();
        worst_case_bokeh_radius.compute_root();
        output.compute_root();
        sample_weight.compute_root();
        sample_locations.compute_root();
        // Time: 0 min
        // Runtime: 6580 ms
        break;
    }
    case 1: {
        // Pick the low hanging fruit first: the cost and cost
        // confidence can be computed with better locality and no
        // redundant compute.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        input_with_alpha.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root();
        worst_case_bokeh_radius.compute_root();
        output.compute_root();
        sample_weight.compute_root();
        sample_locations.compute_root();
        // Time: 5 min
        // Runtime: 5800 ms
        break;
    }
    case 2: {
        // Similar to the above, output and input_with_alpha look like
        // they can be moved inwards without compromise.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c);
        worst_case_bokeh_radius.compute_root();
        output.compute_at(final, x)
            .unroll(c);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c);
        sample_weight.compute_root();
        sample_locations.compute_root();
        // Time: 15 min
        // Runtime: 3500 ms
        break;
    }
    case 3: {
        // It looks like sample_weights and sample_locations can also be moved to be per output pixel.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c);
        worst_case_bokeh_radius.compute_root();
        output.compute_at(final, x)
            .unroll(c);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c);
        sample_weight.compute_at(output, s);
        sample_locations.compute_at(output, x); // I'd like to also compute this at s, but it's crashing Halide.
        // Time: 30 min
        // Runtime: 2066 ms
        break;
    }
    case 4: {
        // worst_case_bokeh_radius is also pointwise.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c);
        worst_case_bokeh_radius.compute_at(final, x);
        output.compute_at(final, x)
            .unroll(c);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c);
        sample_weight.compute_at(output, s);
        sample_locations.compute_at(output, x); // I'd like to also compute this at s, but it's crashing Halide.
        // Time: 31 min
        // Runtime: 2066 ms
        break;
    }
    case 5: {
        // worst_case_bokeh_radius is also pointwise.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        worst_case_bokeh_radius_y.compute_root();

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c);
        worst_case_bokeh_radius.compute_at(final, x);
        output.compute_at(final, x)
            .unroll(c);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c);
        sample_weight.compute_at(output, s);
        sample_locations.compute_at(output, x); // I'd like to also compute this at s, but it's crashing Halide.
        // Time: 35 min
        // Runtime: 2223 ms  (that really shouldn't be slower, but I'm certain it should be an improvement. Moving on...)
        break;
    }
    case 6: {
        // worst_case_bokeh_radius is a stencil, compute the y filter at scanlines.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y);

        cost.compute_at(cost_pyramid_push[0], x);
        cost_confidence.compute_at(cost_pyramid_push[0], x);

        depth.compute_root();

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c);
        worst_case_bokeh_radius.compute_at(final, x);
        output.compute_at(final, x)
            .unroll(c);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c);
        sample_weight.compute_at(output, s);
        sample_locations.compute_at(output, x); // I'd like to also compute this at s, but it's crashing Halide.

        worst_case_bokeh_radius_y.compute_at(final, y);

        // Time: 38 min
        // Runtime: 2173 ms
        break;
    }
    case 7: {
        // Before making redundant compute tradeoffs, let's parallelize and vectorize everything (but the pyramid).
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(z, c, x, y)
            .parallel(y)
            .vectorize(x, 8);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        depth.compute_root()
            .parallel(y)
            .vectorize(x, 8);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .vectorize(x, 8)
            .parallel(y);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 8);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 8);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);
;

        // Time: 50 min
        // Runtime: 1060 ms (I expected more improvement from this...).
        break;
    }
    case 8: {
        // I noticed the loop over c in cost_pyramid_push[0] should be unrolled to simplify a select.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(c, z, x, y)
            .unroll(c)
            .parallel(y)
            .vectorize(x, 8);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        depth.compute_root()
            .parallel(y)
            .vectorize(x, 8);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .vectorize(x, 8)
            .parallel(y);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 8);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 8);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);

        // Time: 51 min
        // Runtime: 1000 ms
        break;
    }
    case 9: {
        // The inputs are 8 bit ints, we should be vectorizing wider in a few places.
        for (int i = 0; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(c, z, x, y)
            .unroll(c)
            .parallel(y)
            .vectorize(x, 32);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        depth.compute_root()
            .parallel(y)
            .vectorize(x, 8);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .vectorize(x, 32)
            .parallel(y);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 32);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 32);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);

        // Time: 55 min
        // Runtime: 920 ms
        break;
    }
    case 10: {
        // I think I've procrastinated the pyramid and the redundant
        // compute as long as possible. Begin by inlining the bottom
        // of the pull pyramid.
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_root();
            cost_pyramid_pull[i].compute_root();
        }

        cost_pyramid_push[0].compute_root()
            .reorder(c, z, x, y)
            .unroll(c)
            .parallel(y)
            .vectorize(x, 32);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        depth.compute_root()
            .parallel(y)
            .vectorize(x, 8);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .vectorize(x, 32)
            .parallel(y);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 32);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 32);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);

        // Time: 59 min
        // Runtime: 550 ms (Wow!)
        break;
    }
    case 11: {
        // Try scheduling the rest of the pyramid.
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_root()
                .reorder(x, z, c, y)
                .parallel(y)
                .vectorize(x, 8);
        }
        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z)
            .unroll(y, 2)
            .parallel(z)
            .vectorize(x, 16);
        for (int i = 2; i < 8; i++) {
            cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], c)
                .vectorize(x, 16)
                .unroll(y, 2);
        }

        cost_pyramid_push[0].compute_root()
            .reorder(c, z, x, y)
            .unroll(c)
            .parallel(y)
            .vectorize(x, 32);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        depth.compute_root()
            .parallel(y)
            .vectorize(x, 16);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .vectorize(x, 32)
            .parallel(y);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 32);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 32);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);

        // Time: 73 min
        // Runtime: 330 ms
        break;
    }
    case 12: {
        // Try tiling the output for the depth stencil.
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_root()
                .reorder(x, z, c, y)
                .parallel(y)
                .vectorize(x, 8);
        }
        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z)
            .unroll(y, 2)
            .parallel(z)
            .vectorize(x, 16);
        for (int i = 2; i < 8; i++) {
            cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], c)
                .vectorize(x, 16)
                .unroll(y, 2);
        }

        cost_pyramid_push[0].compute_root()
            .reorder(c, z, x, y)
            .unroll(c)
            .parallel(y)
            .vectorize(x, 32);

        cost.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);
        cost_confidence.compute_at(cost_pyramid_push[0], x)
            .vectorize(x, 8);

        final.compute_root()
            .reorder(c, x, y)
            .unroll(c)
            .tile(x, y, tx, ty, x, y, 256, 64)
            .vectorize(x, 32)
            .parallel(ty);
        worst_case_bokeh_radius.compute_at(final, x)
            .vectorize(x, 8);
        output.compute_at(final, x)
            .unroll(c)
            .vectorize(x, 32);
        output.update(0)
            .reorder(c, s, x, y)
            .unroll(c)
            .vectorize(x, 32);
        sample_weight.compute_at(output, s);

        sample_locations.compute_at(output, x)
            .vectorize(x, 8);

        worst_case_bokeh_radius_y.compute_at(final, y)
            .vectorize(x, 8);

        depth.compute_at(final, tx)
            .vectorize(x, 16);

        // Time: 80 min
        // Runtime: 375 ms (I can't find tiling that helps. I give up, schedule 11 is the winner.
        break;
    }
    default:
        break;
    }

    final.compile_to_lowered_stmt("final.html", final.infer_arguments(), HTML, get_target_from_environment());

    // Run it

    left_im.set(in_l);
    right_im.set(in_r);
    Image<float> out(in_l.width(), in_l.height(), 3);
    Target target = get_target_from_environment();
    if (schedule == -1) {
        final.compile_jit(target, true);
    } else {
        final.compile_jit(target);
    }

    // std::cout << "runtime: " << std::endl;
    double best = benchmark(4, 4, [&]() { final.realize(out); });
    std::cout << "runtime: " << best * 1e3 << std::endl;

    // save_image(out, argv[3]);

    return 0;
}
