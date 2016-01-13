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
    Expr w = left_im.width(), h = left_im.height();
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
    Func final;
    final(x, y, c) = output(x, y, c) / output(x, y, 3);

    Image<uint8_t> in_l = load_image(argv[1]);
    Image<uint8_t> in_r = load_image(argv[2]);
    final.bound(x, 0, in_l.width()).bound(y, 0, in_l.height()).bound(c, 0, 3);

    int schedule = atoi(argv[4]);
    // Start: 2:10pm
    switch(schedule) {
    case 0:
    {
        // Inline pointwise funcs and boundary conditions. Compute
        // root, parallelize across outermost, vectorize across x,
        // move c innermost and unroll where appropriate.
        cost_pyramid_push[0].compute_root().reorder(c, x, z, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(z);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);
            cost_pyramid_pull[i].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);
        }
        // cost_pyramid_pull[0] and filtered_cost are inlined into depth
        depth.compute_root().vectorize(x, 8).parallel(y);

        // Going to be sampling this a bunch. Not much work per
        // pixel so use strips of scanlines for parallel tasks.
        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        // We also sample this a bunch
        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        // Everything else should probably grouped with final...
        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);

        // Ran this at 2:35. 453ms
    }
    break;
    case 1:
        // Spend a while just reading the pseudocode looking for
        // opportunities to improve it... pyramid blurring is
        // independent per z. Move the pyramid blurring inside
        // cost_pyramid_pull[1]'s loop over z. Also
        // cost_pyramid_push[0] should have the loop over z innermost
        // for cost_confidence to be computed there I think.
    {
        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        // Only 32 tasks might not be enough load balancing
        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        // cost_pyramid_pull[0] and filtered_cost are inlined into depth
        depth.compute_root().vectorize(x, 8).parallel(y);

        // Going to be sampling this a bunch. Not much work per
        // pixel so use strips of scanlines for parallel tasks.
        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        // We also sample this a bunch
        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        // Everything else should probably grouped with final...
        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);

        // Ran this at 2:47. 176ms
    }
    break;
    case 2:
    {
        // Lots of boundary condition junk is happening in the cost
        // function. Might be better off padding each input scanline
        // in the search.
        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);
        diff.compute_at(cost_pyramid_push[0], y).vectorize(x, 16);
        right.compute_at(cost_pyramid_push[0], y).vectorize(_0, 16);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        // Only 32 tasks might not be enough load balancing
        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        // cost_pyramid_pull[0] and filtered_cost are inlined into depth
        depth.compute_root().vectorize(x, 8).parallel(y);

        // Going to be sampling this a bunch. Not much work per
        // pixel so use strips of scanlines for parallel tasks.
        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        // We also sample this a bunch
        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        // Everything else should probably grouped with final...
        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);

        // Ran this at 2:59. 179 ms :(
    }
    break;
    case 3:
    {
        // diff.compute_at was a dumb idea. Cut it.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);
        right.compute_at(cost_pyramid_push[0], y).vectorize(_0, 16);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        // Only 32 tasks might not be enough load balancing
        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        // cost_pyramid_pull[0] and filtered_cost are inlined into depth
        depth.compute_root().vectorize(x, 8).parallel(y);

        // Going to be sampling this a bunch. Not much work per
        // pixel so use strips of scanlines for parallel tasks.
        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        // We also sample this a bunch
        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        // Everything else should probably grouped with final...
        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);

        // Ran this at 3:02. 177ms
        // I am exploring an unproductive part of the schedule space
    }
    break;
    case 4:
    {
        // Current profile is:
        /*
          cost_pyramid_push[0]:  101.080048ms   (29%)
          repeat_edge$2:         2.120529ms     (0%)
          cost:                  47.641014ms    (13%)
          cost_confidence:       1.061826ms     (0%)
          sum$2:                 3.173286ms     (0%)
          sum:                   5.298717ms     (1%)
          cost_pyramid_pull[1]:  13.983347ms    (4%)
          cost_pyramid_push[1]:  15.051983ms    (4%)
          cost_pyramid_push[2]:  9.647482ms     (2%)
          cost_pyramid_push[3]:  7.501913ms     (2%)
          cost_pyramid_push[4]:  3.205331ms     (0%)
          cost_pyramid_push[5]:  2.140458ms     (0%)
          cost_pyramid_push[6]:  0.000000ms     (0%)
          cost_pyramid_push[7]:  0.000000ms     (0%)
          cost_pyramid_pull[7]:  0.000000ms     (0%)
          cost_pyramid_pull[6]:  0.000000ms     (0%)
          cost_pyramid_pull[5]:  0.000000ms     (0%)
          cost_pyramid_pull[4]:  0.000000ms     (0%)
          cost_pyramid_pull[3]:  1.071502ms     (0%)
          cost_pyramid_pull[2]:  2.130642ms     (0%)
          depth:                 9.491812ms     (2%)
          argmin:                22.225187ms    (6%)
          bokeh_radius:          1.059442ms     (0%)
          bokeh_radius_squared:  1.064262ms     (0%)
          input_with_alpha:      1.054206ms     (0%)
          worst_case_bokeh_radius_y: 2.149454ms (0%)
          maximum:               0.000000ms     (0%)
          final:                 6.338049ms     (1%)
          worst_case_bokeh_radius: 3.169680ms   (0%)
          maximum$2:             0.000000ms     (0%)
          output:                80.236969ms    (23%)
        */

        // the argmin is suspiciously slow. cost_pyramid_pull[0] is
        // inlined into it. Maybe that was dumb.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[0], z).vectorize(x, 8);
            cost_pyramid_pull[i].compute_at(cost_pyramid_pull[0], z).vectorize(x, 8);
        }

        // Only 32 tasks might not be enough load balancing
        cost_pyramid_pull[0].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        // filtered_cost is inlined into depth
        depth.compute_root().vectorize(x, 8).parallel(y);

        // Going to be sampling this a bunch. Not much work per
        // pixel so use strips of scanlines for parallel tasks.
        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        // We also sample this a bunch
        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        // Everything else should probably grouped with final...
        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);

        // Ran at 3:11: 184 ms
    }
    break;
    case 5:
    {
        // output still has a loop over c. I'm redundantly recomputing it per c of the output. Oops.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);

        // Ran at 3:15: 235 ms... wtf? Oh, forgot to vectorize the update definition.
    }
    break;
    case 6:
    {
        // output still has a loop over c. I'm redundantly recomputing it per c of the output. Oops.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().vectorize(x);

        // Ran at 3:16: 143 ms
    }
    break;
    case 7:
    {
        // It's possible that computing the sample locations should be vectorized but not the actual sampling.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        sample_locations.compute_at(final, x).vectorize(x);

        // Ran at 3:20: 143 ms again. OK. Maybe let's compute it further in.
    }
    break;
    case 8:
    {
        // Oops, redundant work is still happening in the channels of output.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:29: 119 ms
    }
    break;
    case 9:
    {
        // Should try unrolling output.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius_squared.compute_root().vectorize(x, 8).parallel(y, 8);
        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:31: 121 ms
    }
    break;
    case 10:
    {
        // Inline bokeh_radius_squared

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_root().parallel(y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:33: 119ms
    }
    break;
    case 11:
    {
        // I think I can move the max filter inwards...

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            }
        }

        cost_pyramid_pull[1].compute_root().vectorize(x, 8).reorder(c, z).parallel(z);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:34: 118ms
    }
    break;
    case 12:
    {
        // The upsamples have y%2 in them. and adjacent scanlines use
        // overlapping data, so maybe split y and move that inner var
        // innermost and unroll it.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z)
                    .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);
            }
        }

        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z).parallel(z)
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:40: 118ms again. I guess y%2 is pretty cheap.
    }
    break;
    case 13:
    {
        // The upsample vectors are only size 4 thanks to the interleave code. Let's widen...

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z)
                    .tile(x, y, xi, yi, 16, 2).vectorize(xi).unroll(yi);
            }
        }

        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z).parallel(z)
            .tile(x, y, xi, yi, 16, 2).vectorize(xi).unroll(yi);

        depth.compute_root().vectorize(x, 8).parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:43: 121ms
    }
    break;
    case 14:
    {
        // The most expensive upsample is the one inlined into depth

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z)
                    .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);
            }
        }

        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z).parallel(z)
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);

        depth.compute_root()
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi)
            .parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).vectorize(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:46: 116ms
    }
    break;
    case 15:
    {
        // Stared at it for a while... I don't see why sample_weight
        // should vectorize well. Unroll instead.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], z).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], z)
                    .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);
            }
        }

        cost_pyramid_pull[1].compute_root()
            .reorder(x, y, c, z).parallel(z)
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);

        depth.compute_root()
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi)
            .parallel(y);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).unroll(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:54: 121ms
    }
    break;
    case 16:
    {
        // Some of the parallel fors could probably be coarser-grained
        // when they're across y, and finer-grained when they're
        // across z.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y, 4);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi, t;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], t).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], t)
                    .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);
            }
        }


        cost_pyramid_pull[1].compute_root()
            .fuse(z, c, t).parallel(t)
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);

        depth.compute_root()
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi)
            .parallel(y, 8);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        bokeh_radius.compute_root().vectorize(x, 8).parallel(y, 8);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).unroll(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 3:59: 111ms
    }
    break;
    case 17:
    {
        // bokeh_radius is a pretty simple function of depth. Maybe I should inline it.

        cost_pyramid_push[0].compute_root().reorder(c, z, x, y).bound(c, 0, 2).unroll(c).vectorize(x, 16).parallel(y, 4);
        cost.compute_at(cost_pyramid_push[0], x).vectorize(x);
        cost_confidence.compute_at(cost_pyramid_push[0], x).vectorize(x);

        Var xi, yi, t;
        for (int i = 1; i < 8; i++) {
            cost_pyramid_push[i].compute_at(cost_pyramid_pull[1], t).vectorize(x, 8);
            if (i > 1) {
                cost_pyramid_pull[i].compute_at(cost_pyramid_pull[1], t)
                    .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);
            }
        }


        cost_pyramid_pull[1].compute_root()
            .fuse(z, c, t).parallel(t)
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi);

        depth.compute_root()
            .tile(x, y, xi, yi, 8, 2).vectorize(xi).unroll(yi)
            .parallel(y, 8);

        input_with_alpha.compute_root().reorder(c, x, y).unroll(c).vectorize(x, 8).parallel(y, 8);

        worst_case_bokeh_radius_y.compute_at(final, y).vectorize(x, 8);

        final.compute_root().reorder(c, x, y).bound(c, 0, 3).unroll(c).vectorize(x, 8).parallel(y);
        worst_case_bokeh_radius.compute_at(final, y).vectorize(x, 8);
        output.compute_at(final, x).vectorize(x);
        output.update().reorder(c, x, s).vectorize(x).unroll(c);
        sample_weight.compute_at(output, x).unroll(x);
        sample_locations.compute_at(output, x).vectorize(x);

        // Ran at 4:01: 111ms

        // At this point, I think I've converged.
    }
    break;
    default:
        break;
    }

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

    std::cout << "Running... " << std::endl;
    double best = benchmark(5, 5, [&]() { final.realize(out); });
    std::cout << " took " << best * 1e3 << " msec." << std::endl;

    save_image(out, argv[3]);

    return 0;
}
