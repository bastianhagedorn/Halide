#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>

#include "benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

using std::vector;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n\t./interpolate in.png out.png\n" << std::endl;
        return 1;
    }

    ImageParam input(Float(32), 3);

    // Input must have four color channels - rgba
    input.set_bounds(2, 0, 4);

    const int levels = 4;

    std::vector<Func> downsampled;
    std::vector<Func> downx;
    std::vector<Func> interpolated;
    std::vector<Func> upsampled;
    std::vector<Func> upsampledx;
    for (int i = 0; i < levels; i++) {
        Func down("downsampled_" + std::to_string(i));
        downsampled.push_back(down);
        Func dx("downx_" + std::to_string(i));
        downx.push_back(dx);
        Func interp("interpolated_" + std::to_string(i));
        interpolated.push_back(interp);
        Func up("upsampled_" + std::to_string(i));
        upsampled.push_back(up);
        Func upx("upsampledx_" + std::to_string(i));
        upsampledx.push_back(upx);

    }

    Var x("x"), y("y"), c("c");

    Func clamped = BoundaryConditions::repeat_edge(input);

    // This triggers a bug in llvm 3.3 (3.2 and trunk are fine), so we
    // rewrite it in a way that doesn't trigger the bug. The rewritten
    // form assumes the input alpha is zero or one.
    // downsampled[0](x, y, c) = select(c < 3, clamped(x, y, c) * clamped(x, y, 3), clamped(x, y, 3));
    downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);

    for (int l = 1; l < levels; ++l) {
        Func prev = downsampled[l-1];

        if (l == 4) {
            // Also add a boundary condition at a middle pyramid level
            // to prevent the footprint of the downsamplings to extend
            // too far off the base image. Otherwise we look 512
            // pixels off each edge.
            //Expr w = input.width()/(1 << l);
            //Expr h = input.height()/(1 << l);
            Expr w = 1536/(1 << l);
            Expr h = 2560/(1 << l);
            prev = lambda(x, y, c, prev(clamp(x, 0, w), clamp(y, 0, h), c));
        }

        downx[l](x, y, c) = (prev(x*2-1, y, c) +
                             2.0f * prev(x*2, y, c) +
                             prev(x*2+1, y, c)) * 0.25f;
        downsampled[l](x, y, c) = (downx[l](x, y*2-1, c) +
                                   2.0f * downx[l](x, y*2, c) +
                                   downx[l](x, y*2+1, c)) * 0.25f;
    }
    interpolated[levels-1](x, y, c) = downsampled[levels-1](x, y, c);
    for (int l = levels-2; l >= 0; --l) {
        upsampledx[l](x, y, c) = (interpolated[l+1](x/2, y, c) +
                                  interpolated[l+1]((x+1)/2, y, c)) / 2.0f;
        upsampled[l](x, y, c) =  (upsampledx[l](x, y/2, c) +
                                  upsampledx[l](x, (y+1)/2, c)) / 2.0f;
        interpolated[l](x, y, c) = downsampled[l](x, y, c) + (1.0f - downsampled[l](x, y, 3)) * upsampled[l](x, y, c);
    }

    Func normalize("normalize");
    normalize(x, y, c) = interpolated[0](x, y, c) / interpolated[0](x, y, 3);

    Func final("final");
    final(x, y, c) = normalize(x, y, c);

    //std::cout << "Finished function setup." << std::endl;

    int sched;
    Target target = get_target_from_environment();
    if (argc == 4)
        sched = atoi(argv[3]);

    if (sched > 0) {
        if (target.has_gpu_feature())
            sched = 4;
        else
            sched = 2;
    }

    switch (sched) {
    case 0:
    {
        //std::cout << "Flat schedule." << std::endl;
        for (int l = 0; l < levels; ++l) {
            downsampled[l].compute_root();
            interpolated[l].compute_root();
        }
        normalize.compute_root();
        break;
    }
    case 1:
    {
        //std::cout << "Flat schedule with vectorization." << std::endl;
        for (int l = 0; l < levels; ++l) {
            downsampled[l].compute_root().vectorize(x,4);
            interpolated[l].compute_root().vectorize(x,4);
        }
        normalize.compute_root();
        break;
    }
    case 2:
    {
        Var xi, yi;
        //std::cout << "Flat schedule with parallelization + vectorization." << std::endl;
        for (int l = 1; l < levels-1; ++l) {
            downsampled[l]
                .compute_root()
                .parallel(y, 8)
                .vectorize(x, 4);
            interpolated[l]
                .compute_root()
                .parallel(y, 8)
                .unroll(x, 2)
                .unroll(y, 2)
                .vectorize(x, 8);
        }
        normalize
            .reorder(c, x, y)
            .bound(c, 0, 3)
            .unroll(c)
            .tile(x, y, xi, yi, 2, 2)
            .unroll(xi)
            .unroll(yi)
            .parallel(y, 8)
            .vectorize(x, 8)
            .bound(x, 0, input.width())
            .bound(y, 0, input.height());
        //final.print_loop_nest();
        break;
    }
    case 3:
    {
        //std::cout << "Flat schedule with vectorization sometimes." << std::endl;
        for (int l = 0; l < levels; ++l) {
            if (l + 4 < levels) {
                Var yo,yi;
                downsampled[l].compute_root().vectorize(x,4);
                interpolated[l].compute_root().vectorize(x,4);
            } else {
                downsampled[l].compute_root();
                interpolated[l].compute_root();
            }
        }
        normalize.compute_root();
        break;
    }
    case 4:
    {
        std::cerr << "GPU schedule." << std::endl;

        // Some gpus don't have enough memory to process the entire
        // image, so we process the image in tiles.
        Var yo, yi, xo, xi;

        // We can't compute the entire output stage at once on the GPU
        // - it takes too much GPU memory on some of our build bots,
        // so we wrap the final stage in a CPU stage.
        Func cpu_wrapper = normalize.in();

        cpu_wrapper
            .reorder(c, x, y)
            .bound(c, 0, 3)
            .tile(x, y, xo, yo, xi, yi, input.width()/4, input.height()/4)
            .vectorize(xi, 8);

        normalize
            .compute_at(cpu_wrapper, xo)
            .reorder(c, x, y)
            .gpu_tile(x, y, 16, 16)
            .unroll(c);

        // Start from level 1 to save memory - level zero will be computed on demand
        for (int l = 1; l < levels; ++l) {
            int tile_size = 32 >> l;
            if (tile_size < 1) tile_size = 1;
            if (tile_size > 8) tile_size = 8;
            downsampled[l]
                .compute_root()
                .gpu_tile(x, y, c, tile_size, tile_size, 4);
            if (l == 1 || l == 4) {
                interpolated[l]
                    .compute_at(cpu_wrapper, xo)
                    .gpu_tile(x, y, c, 8, 8, 4);
            } else {
                int parent = l > 4 ? 4 : 1;
                interpolated[l]
                    .compute_at(interpolated[parent], Var::gpu_blocks())
                    .gpu_threads(x, y, c);
            }
        }

        // The cpu wrapper is our new output Func
        normalize = cpu_wrapper;

        break;
    }
    }

    if (sched == -1 || sched == -2)
    {
        final.bound(x, 0, 1536).bound(y, 0, 2560).bound(c, 0, 3);
        final.estimate(x, 0, 1536).estimate(y, 0, 2560).estimate(c, 0, 3);
    }

    if (sched == -2) {
        //target.set_feature(Halide::Target::CUDA);
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }

    // JIT compile the pipeline eagerly, so we don't interfere with timing
    if (sched == -1 || sched == -2)
        final.compile_jit(target, true);
    else
        final.compile_jit(target, false);

    Image<float> in_png = load_image(argv[1]);
    Image<float> out(in_png.width(), in_png.height(), 3);
    //std::cout << in_png.width() << "," << in_png.height() << std::endl;
    assert(in_png.channels() == 4);
    input.set(in_png);

    //std::cout << "Running... " << std::endl;
    double best = benchmark(10, 100, [&]() { final.realize(out);}, [&](){ out.copy_to_host();});
    //std::cout << " took " << best * 1e3 << " msec." << std::endl;
    std::cout << "runtime: " << best * 1e3 << std::endl;

    //vector<Argument> args;
    //args.push_back(input);
    //final.compile_to_assembly("test.s", args, target);

    // save_image(out, argv[2]);

    return 0;
}
