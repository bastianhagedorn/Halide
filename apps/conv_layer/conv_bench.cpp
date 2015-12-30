#include "layers.h"
#include<stdio.h>
#include "benchmark.h"

int main(int argc, char **argv) {

    int N = 4; // number of samples/batch_size
    int d_w = 128; // data width
    int d_h = 128; // data height
    int ch = 64; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));

    int n_f = 64; // number of filters
    int f_w = 3;  // filter width
    int f_h = 3;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated
    float reg = 0.1;
    Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
                                              stride, reg, d_layer);
    printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
                                                conv->out_dim_size(1),
                                                conv->out_dim_size(2),
                                                conv->out_dim_size(3));
    Func f_in_bound;
    f_in_bound = BoundaryConditions::constant_exterior(
                                    d_layer->forward, 0,
                                    0, d_w, 0, d_h);

    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));

    Image<float> W(f_w, f_h, ch, n_f), b(n_f);

    Var x, y, z, n, par;
    Var i_B, o_B, x_t, y_t, z_t;
    // Simple convolution
    Func f_conv("conv");
    RDom r(0, f_w, 0, f_h, 0, ch);

    f_conv(x, y, z, n) = b(z);

    f_conv(x, y, z, n) += W(r.x, r.y, r.z, z) *
                           f_in_bound(x + r.x - pad,
                                      y + r.y - pad,
                                      r.z, n);

    int vec_len = 8;
    int o_block_size = 32;
    int y_block = 32;
    int sched = atoi(argv[1]);

    if (sched == 0) {
       // blocking spatially with vectorization
       //f_in_bound.compute_at(f_conv, par);
       f_in_bound.compute_at(f_conv, z_t);
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
       //f_conv.update().fuse(y, par, par).parallel(par);
       //f_conv.update().parallel(z);
       //f_conv.print_loop_nest();
    }

    Target target = get_target_from_environment();
    if (sched == -1)
        f_conv.compile_jit(target, true);
    else
        f_conv.compile_jit(target, false);

    std::vector<Func> simple_outs;
    simple_outs.push_back(f_conv);
    double best = benchmark(3, 1, [&]() { f_conv.realize(conv_out); });
    std::cerr << best * 1e3 << std::endl;
}
