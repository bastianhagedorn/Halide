#include "layers.h"
#include<stdio.h>
#include "benchmark.h"

#ifndef p_N
#define p_N 4     // number of samples/batch_size
#endif

#ifndef p_d_w
#define p_d_w 128 // data width
#endif

#ifndef p_d_h
#define p_d_h 128 // data height
#endif

#ifndef p_ch
#define p_ch 64   // number of input channels
#endif

#ifndef p_n_f
#define p_n_f 64  // number of filters
#endif

#ifndef p_f_w
#define p_f_w 3   // filter width
#endif

#ifndef p_f_h
#define p_f_h 3   // filter height
#endif

int main(int argc, char **argv) {

    Image<float> data(p_d_w, p_d_h, p_ch, p_N);
    DataLayer * d_layer = new DataLayer(p_d_h, p_d_w, p_ch, p_N, data);
    fprintf(stderr, "data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));

    int pad = (p_f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated
    float reg = 0.1;
    Convolutional * conv  = new Convolutional(p_n_f, p_f_w, p_f_h, pad,
                                              stride, reg, d_layer);
    fprintf(stderr, "conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
                                                conv->out_dim_size(1),
                                                conv->out_dim_size(2),
                                                conv->out_dim_size(3));
    Func f_in_bound;
    f_in_bound = BoundaryConditions::repeat_edge(d_layer->forward, 0, p_d_w,
                                                 0, p_d_h);

    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));

    Image<float> W(p_f_w, p_f_h, p_ch, p_n_f), b(p_n_f);

    Var x, y, z, n, par;
    Var i_B, o_B, x_t, y_t, z_t;
    // Simple convolution
    Func f_conv("conv");
    RDom r(0, p_f_w, 0, p_f_h, 0, p_ch);

    f_conv(x, y, z, n) = b(z);

    f_conv(x, y, z, n) += W(r.x, r.y, r.z, z) *
                           f_in_bound(x + r.x - pad,
                                      y + r.y - pad,
                                      r.z, n);

    Func f_ReLU("ReLU");
    f_ReLU(x, y, z, n) = max(0, f_conv(x, y, z, n));

    /*
    f_ReLU.bound(x, 0, conv->out_dim_size(0)).
           bound(y, 0, conv->out_dim_size(1)).
           bound(z, 0, conv->out_dim_size(2)).
           bound(n, 0, conv->out_dim_size(3));
    */

    f_ReLU.estimate(x, 0, conv->out_dim_size(0)).
           estimate(y, 0, conv->out_dim_size(1)).
           estimate(z, 0, conv->out_dim_size(2)).
           estimate(n, 0, conv->out_dim_size(3));

    int vec_len = 8;
    int o_block_size = 32;
    int y_block = 32;
    int sched = atoi(argv[1]);

    Target target = get_target_from_environment();

    if (sched == 0) {
        if (target.has_gpu_feature()) {
            Var ni, no, xi, xo, yi, yo, zi, zo;
            f_ReLU.compute_root()
                .split(x, xo, xi, 8)
                .split(y, yo, yi, 8)
                .split(z, zo, zi, 16)
                .reorder(xi, yi, zi, n, xo, yo, zo)
                .gpu_threads(xi, yi, zi)
                .gpu_blocks(xo, yo, zo);

            f_conv.compute_at(f_ReLU, n)
                .gpu_threads(x, y, z)
                .update()
                .unroll(r.x)
                .unroll(r.y)
                .gpu_threads(x, y, z);

            Var v0 = f_in_bound.args()[0];
            Var v1 = f_in_bound.args()[1];
            Var v2 = f_in_bound.args()[2];
            Var v0o, v0i, v1o, v1i, v2o, v2i;
            f_in_bound.compute_at(f_ReLU, n)
                .split(v0, v0o, v0i, 2)
                .split(v1, v1o, v1i, 2)
                .split(v2, v2o, v2i, 4)
                .reorder(v0i, v1i, v2i, v0o, v1o, v2o)
                .unroll(v0i)
                .unroll(v1i)
                .gpu_threads(v0o, v1o, v2o);
        } else {
            // blocking spatially with vectorization
            //f_in_bound.compute_at(f_conv, par);
            //f_in_bound.compute_at(f_conv, z_t);
            f_in_bound.compute_root().parallel(f_in_bound.args()[3]);
            //f_in_bound.compute_root().parallel(f_in_bound.args()[2]);
            f_conv.compute_root();
            f_conv.fuse(z, n, par).parallel(par);
            f_conv.update().reorder(x, y, r.z);
            f_conv.update().split(y, y, y_t, y_block);
            f_conv.update().split(z, z, z_t, o_block_size);
            f_conv.update().reorder(y_t, z_t, y, r.z, z);
            f_conv.update().vectorize(x, vec_len);
            f_conv.update().unroll(r.x);
            f_conv.update().unroll(r.y);
            f_conv.update().fuse(z, n, par).parallel(par);
            //f_conv.update().reorder(x, r.x, r.y).vectorize(x, vec_len);
            //f_conv.update().fuse(z, n, par).parallel(par);
            //f_conv.update().fuse(y, par, par).parallel(par);
            //f_conv.update().parallel(z);
            //f_conv.print_loop_nest();
        }
    } else if (sched == 1) { //autoschedule version for middle choice
       RVar z_r;
       f_in_bound.compute_root().parallel(f_in_bound.args()[3]);
       f_conv.compute_root().vectorize(x, 8);
       f_conv.reorder(z, y, n).fuse(y, n, par).parallel(par);
       //f_conv.update().reorder(x, y, r.z);
       f_conv.update().reorder(z, r.z, y, n);
       f_conv.update().split(r.z, r.z, z_r, 32).reorder(x, z_r, z, r.z);
       f_conv.update().vectorize(x, vec_len);
       f_conv.update().unroll(r.x);
       f_conv.update().unroll(r.y);
       f_conv.update().parallel(n);
       f_conv.update().parallel(y);

       f_ReLU.reorder(n, z).parallel(z).vectorize(x, 8);
       //f_ReLU.print_loop_nest();
    }


    if (sched == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::CUDA);
        //target.set_feature(Halide::Target::Debug);
    }

    if (sched == -1 || sched == -2)
        f_ReLU.compile_jit(target, true);
    else
        f_ReLU.compile_jit(target, false);

    Buffer buf(conv_out);
    double best = benchmark(5, 20, [&]() { f_ReLU.realize(buf);}, [&](){buf.device_sync();});
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
