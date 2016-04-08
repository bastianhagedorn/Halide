#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>
#include "benchmark.h"

int main(int argc, char **argv) {
    int size = 2048;
    Image<float> A(size, size);
    Image<float> B(size, size);
    Image<float> C(size, size);

    for (int y = 0; y < A.height(); y++) {
        for (int x = 0; x < A.width(); x++) {
            A(x, y) = rand() & 0xfff;
        }
    }

    for (int y = 0; y < B.height(); y++) {
        for (int x = 0; x < B.width(); x++) {
            B(x, y) = rand() & 0xfff;
        }
    }

    Var x, y;

    Func prod("prod");
    RDom r(0, size);

    prod(x, y) = 0.0f;
    prod(x, y) += A(x, r.x) * B(r.x, y);

    Func out;
    out(x, y) = prod(x, y);

    //out.bound(x, 0, size).bound(y, 0, size);
    out.estimate(x, 0, size).estimate(y, 0, size);

    int sched = atoi(argv[1]);

    Target target = get_target_from_environment();

    if (sched == 0) {
        if (target.has_gpu_feature()) {
            Var xi, yi, xii, yii;
            out.tile(x, y, xi, yi, 8, 8).unroll(xi).unroll(yi).gpu_tile(x, y, 8, 8);
            prod.compute_at(out, Var::gpu_threads()).update().reorder(x, y, r.x);

            // This schedule as-is is terrible - 518ms

            // Not unrolled, a mat mul does 2 loads per
            // multiply-add. We unroll an 8x8 block so that the common
            // loads will be shared. This means we do 16 loads for 64
            // multiply adds, which is a huge win.


            // prod.update().unroll(x).unroll(y);
            // 53ms

            // We then also use Z-order within each 8x8 unrolled block
            // to minimize register pressure and avoid the big hit of
            // 8 high-latency loads up-front. This is surprisingly
            // effective.
            //
            // prod.update().tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi)
            //    .tile(x, y, xii, yii, 2, 2).unroll(xii).unroll(yii)
            //    .unroll(x).unroll(y);
            // 46ms

            // We also vectorize the innermost pair of float loads so
            // that we use 64-bit memory accesses to A instead of 32-bit.

            Var t;
            prod.update()
                .tile(x, y, xi, yi, 2, 2).vectorize(xi).unroll(yi)
                .tile(x, y, xii, yii, 2, 2).unroll(xii).unroll(yii)
                .unroll(x).unroll(y);

            // 36ms

            // Still not as fast as apps/linear_algebra on the CPU on
            // the same machine (28ms). There are probably way more
            // tricks a good CUDA programmer can pull out
            // here. Counting a multiply-add as two ops, this is 477
            // GFlops on a card that's supposed to be capable of
            // 1728. In terms of memory bandwidth we're doing 16 loads
            // in the inner loop, which executes 2048*2048*2048 /
            // (8*8) times, which is 238 GB/s on a card that
            // advertises 86.4. So I guess the cache is working.

            // If we assume perfect cache hits for threads in a block,
            // then each thread block handles a 64x64 tile of output,
            // so it touches 64*2048 values from each matrix, which is
            // 64*2048*4*2 bytes. There are (2048*2048)/(64*64) total
            // blocks, so the total number of bytes loaded with
            // perfect caching per block is
            // 2048*2048*2048*64*4*2/(64*64), which implies 29.8
            // GB/s. So we're getting good but not great caching.
        } else {
            Var xi, yi, xii, yii;
            // Tile the output domain
            prod.compute_at(out, x).vectorize(x);
            prod.update().reorder(x, y, r).vectorize(x).unroll(y);
            out.tile(x, y, xi, yi, 16, 4).vectorize(xi).unroll(yi).parallel(y);
            out.print_loop_nest();
        }
    }

    if (sched == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::Debug);
    }

    if (sched == -1 || sched == -2)
        out.compile_jit(target, true);
    else
        out.compile_jit(target, false);

    target.set_features({Target::NoAsserts, Target::NoRuntime, Target::NoBoundsQuery});
    //out.compile_to_assembly("/dev/stderr", {A, B}, target);

    std::vector<Func> outs;
    outs.push_back(out);
    Buffer buf(C);
    double best = benchmark(6, 3, [&]() { out.realize(buf); buf.device_sync(); });
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
