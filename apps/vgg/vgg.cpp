#include<stdio.h>
#include "layers.h"
#include "benchmark.h"

int main(int argc, char **argv) {

    int sched = atoi(argv[1]);

    // Network structure
    // data -> conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1 ->
    // conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2 ->
    // conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3 ->
    // conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4 ->
    // conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5 ->
    // fc6-> relu6 -> droupout6-> fc7 -> relu7 -> dropout7 -> fc8 -> loss

    std::vector<Layer*> network;
    float reg = 0.001;

    // Description of the neural network

    int N = 16; // number of samples/batch_size
    int d_w = 224; // data width
    int d_h = 224; // data height
    int ch = 3; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    Image<int> labels(N);

    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    network.push_back(d_layer);
    fprintf(stderr, "data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));
    int n_f_1 = 64; // number of filters
    int f_w = 3;  // filter width
    int f_h = 3;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated

    Convolutional * conv1_1  = new Convolutional(n_f_1, f_w, f_h, pad,
                                              stride, reg, d_layer, sched);
    network.push_back(conv1_1);
    fprintf(stderr, "conv1_1 out size %d x %d x %d x %d\n", conv1_1->out_dim_size(0),
                                                   conv1_1->out_dim_size(1),
                                                   conv1_1->out_dim_size(2),
                                                   conv1_1->out_dim_size(3));

    ReLU * relu1_1 = new ReLU(conv1_1, sched);
    network.push_back(relu1_1);

    Convolutional * conv1_2  = new Convolutional(n_f_1, f_w, f_h, pad,
                                              stride, reg, relu1_1, sched);
    network.push_back(conv1_2);
    fprintf(stderr, "conv1_2 out size %d x %d x %d x %d\n", conv1_2->out_dim_size(0),
                                                   conv1_2->out_dim_size(1),
                                                   conv1_2->out_dim_size(2),
                                                   conv1_2->out_dim_size(3));
    ReLU * relu1_2 = new ReLU(conv1_2, sched);
    network.push_back(relu1_2);

    int p_w = 2; // pooling width
    int p_h = 2; // pooling height
    int p_stride = 2; // pooling stride

    MaxPooling * pool1 = new MaxPooling(p_w, p_h, p_stride, relu1_2, sched);
    network.push_back(pool1);
    fprintf(stderr, "pool1 out size %d x %d x %d x %d\n", pool1->out_dim_size(0),
                                                 pool1->out_dim_size(1),
                                                 pool1->out_dim_size(2),
                                                 pool1->out_dim_size(3));

    int n_f_2 = 128;
    Convolutional * conv2_1  = new Convolutional(n_f_2, f_w, f_h, pad,
                                              stride, reg, pool1, sched);
    network.push_back(conv2_1);
    fprintf(stderr, "conv2_1 out size %d x %d x %d x %d\n", conv2_1->out_dim_size(0),
                                                   conv2_1->out_dim_size(1),
                                                   conv2_1->out_dim_size(2),
                                                   conv2_1->out_dim_size(3));

    ReLU * relu2_1 = new ReLU(conv2_1, sched);
    network.push_back(relu2_1);

    Convolutional * conv2_2  = new Convolutional(n_f_2, f_w, f_h, pad,
                                              stride, reg, relu2_1, sched);
    network.push_back(conv2_2);
    fprintf(stderr, "conv2_2 out size %d x %d x %d x %d\n", conv2_2->out_dim_size(0),
                                                   conv2_2->out_dim_size(1),
                                                   conv2_2->out_dim_size(2),
                                                   conv2_2->out_dim_size(3));
    ReLU * relu2_2 = new ReLU(conv2_2, sched);
    network.push_back(relu2_2);

    MaxPooling * pool2 = new MaxPooling(p_w, p_h, p_stride, relu2_2, sched);
    network.push_back(pool2);
    fprintf(stderr, "pool2 out size %d x %d x %d x %d\n", pool2->out_dim_size(0),
                                                 pool2->out_dim_size(1),
                                                 pool2->out_dim_size(2),
                                                 pool2->out_dim_size(3));

    int n_f_3 = 256;
    Convolutional * conv3_1  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, pool2, sched);
    network.push_back(conv3_1);
    fprintf(stderr, "conv3_1 out size %d x %d x %d x %d\n", conv3_1->out_dim_size(0),
                                                   conv3_1->out_dim_size(1),
                                                   conv3_1->out_dim_size(2),
                                                   conv3_1->out_dim_size(3));

    ReLU * relu3_1 = new ReLU(conv3_1, sched);
    network.push_back(relu3_1);

    Convolutional * conv3_2  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, relu3_1, sched);
    network.push_back(conv3_2);
    fprintf(stderr, "conv3_2 out size %d x %d x %d x %d\n", conv3_2->out_dim_size(0),
                                                   conv3_2->out_dim_size(1),
                                                   conv3_2->out_dim_size(2),
                                                   conv3_2->out_dim_size(3));
    ReLU * relu3_2 = new ReLU(conv3_2, sched);
    network.push_back(relu3_2);

    Convolutional * conv3_3  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, relu3_2, sched);
    network.push_back(conv3_3);
    fprintf(stderr, "conv3_3 out size %d x %d x %d x %d\n", conv3_3->out_dim_size(0),
                                                   conv3_3->out_dim_size(1),
                                                   conv3_3->out_dim_size(2),
                                                   conv3_3->out_dim_size(3));
    ReLU * relu3_3 = new ReLU(conv3_3, sched);
    network.push_back(relu3_3);

    MaxPooling * pool3 = new MaxPooling(p_w, p_h, p_stride, relu3_3, sched);
    network.push_back(pool3);
    fprintf(stderr, "pool3 out size %d x %d x %d x %d\n", pool3->out_dim_size(0),
                                                 pool3->out_dim_size(1),
                                                 pool3->out_dim_size(2),
                                                 pool3->out_dim_size(3));

    int n_f_4 = 512;
    Convolutional * conv4_1  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, pool3, sched);
    network.push_back(conv4_1);
    fprintf(stderr, "conv4_1 out size %d x %d x %d x %d\n", conv4_1->out_dim_size(0),
                                                   conv4_1->out_dim_size(1),
                                                   conv4_1->out_dim_size(2),
                                                   conv4_1->out_dim_size(3));

    ReLU * relu4_1 = new ReLU(conv4_1, sched);
    network.push_back(relu4_1);

    Convolutional * conv4_2  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, relu4_1, sched);
    network.push_back(conv4_2);
    fprintf(stderr, "conv4_2 out size %d x %d x %d x %d\n", conv4_2->out_dim_size(0),
                                                   conv4_2->out_dim_size(1),
                                                   conv4_2->out_dim_size(2),
                                                   conv4_2->out_dim_size(3));
    ReLU * relu4_2 = new ReLU(conv4_2, sched);
    network.push_back(relu4_2);

    Convolutional * conv4_3  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, relu4_2, sched);
    network.push_back(conv4_3);
    fprintf(stderr, "conv4_3 out size %d x %d x %d x %d\n", conv4_3->out_dim_size(0),
                                                   conv4_3->out_dim_size(1),
                                                   conv4_3->out_dim_size(2),
                                                   conv4_3->out_dim_size(3));
    ReLU * relu4_3 = new ReLU(conv4_3, sched);
    network.push_back(relu4_3);

    MaxPooling * pool4 = new MaxPooling(p_w, p_h, p_stride, relu4_3, sched);
    network.push_back(pool4);
    fprintf(stderr, "pool4 out size %d x %d x %d x %d\n", pool4->out_dim_size(0),
                                                 pool4->out_dim_size(1),
                                                 pool4->out_dim_size(2),
                                                 pool4->out_dim_size(3));

    int n_f_5 = 512;
    Convolutional * conv5_1  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, pool4, sched);
    network.push_back(conv5_1);
    fprintf(stderr, "conv5_1 out size %d x %d x %d x %d\n", conv5_1->out_dim_size(0),
                                                   conv5_1->out_dim_size(1),
                                                   conv5_1->out_dim_size(2),
                                                   conv5_1->out_dim_size(3));

    ReLU * relu5_1 = new ReLU(conv5_1, sched);
    network.push_back(relu5_1);

    Convolutional * conv5_2  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, relu5_1, sched);
    network.push_back(conv5_2);
    fprintf(stderr, "conv5_2 out size %d x %d x %d x %d\n", conv5_2->out_dim_size(0),
                                                   conv5_2->out_dim_size(1),
                                                   conv5_2->out_dim_size(2),
                                                   conv5_2->out_dim_size(3));
    ReLU * relu5_2 = new ReLU(conv5_2, sched);
    network.push_back(relu5_2);

    Convolutional * conv5_3  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, relu5_2, sched);
    network.push_back(conv5_3);
    fprintf(stderr, "conv5_3 out size %d x %d x %d x %d\n", conv5_3->out_dim_size(0),
                                                   conv5_3->out_dim_size(1),
                                                   conv5_3->out_dim_size(2),
                                                   conv5_3->out_dim_size(3));
    ReLU * relu5_3 = new ReLU(conv5_3, sched);
    network.push_back(relu5_3);

    MaxPooling * pool5 = new MaxPooling(p_w, p_h, p_stride, relu5_3, sched);
    network.push_back(pool5);
    fprintf(stderr, "pool5 out size %d x %d x %d x %d\n", pool5->out_dim_size(0),
                                                 pool5->out_dim_size(1),
                                                 pool5->out_dim_size(2),
                                                 pool5->out_dim_size(3));
    Flatten * flatten = new Flatten(pool5, sched);
    network.push_back(flatten);
    fprintf(stderr, "flatten out size %d x %d\n", flatten->out_dim_size(0),
                                         flatten->out_dim_size(1));

    int fc6_out_dim = 4096;

    Affine * fc6 = new Affine(fc6_out_dim, reg, flatten, sched);
    network.push_back(fc6);
    fprintf(stderr, "fc6 out size %d x %d\n", fc6->out_dim_size(0),
                                     fc6->out_dim_size(1));

    ReLU * relu6 = new ReLU(fc6, sched);
    network.push_back(relu6);

    // TODO Add drop out

    int fc7_out_dim = 4096;

    Affine * fc7 = new Affine(fc7_out_dim, reg, relu6, sched);
    network.push_back(fc7);
    fprintf(stderr, "fc7 out size %d x %d\n", fc7->out_dim_size(0),
                                     fc7->out_dim_size(1));

    ReLU * relu7 = new ReLU(fc7, sched);
    network.push_back(relu7);

    // TODO Add drop out

    int C = 1000;
    Affine * fc8 = new Affine(C, reg, relu7, sched);
    network.push_back(fc8);
    fprintf(stderr, "fc8 out size %d x %d\n", fc8->out_dim_size(0),
                                     fc8->out_dim_size(1));

    SoftMax * softm = new SoftMax(fc8, sched);
    network.push_back(softm);
    fprintf(stderr, "softm out size %d x %d\n", softm->out_dim_size(0),
                                       softm->out_dim_size(1));

    Func acc = softm->loss(Func(labels));
    Image<float> scores(C, N), loss(1);

    softm->forward.bound(softm->forward.args()[0], 0, C).
                   bound(softm->forward.args()[1], 0, N);
    acc.bound(acc.args()[0], 0, 1);

    softm->forward.estimate(softm->forward.args()[0], 0, C).
                   estimate(softm->forward.args()[1], 0, N);
    acc.estimate(acc.args()[0], 0, 1);

    std::vector<Func> test_outs;
    test_outs.push_back(acc);
    test_outs.push_back(softm->forward);
    Pipeline test(test_outs);

    Target target = get_target_from_environment();
    if (sched == -2) {
        target.set_feature(Halide::Target::CUDACapability35);
        //target.set_feature(Halide::Target::CUDA);
        //target.set_feature(Halide::Target::Debug);
    }

    if (sched == -1 || sched == -2)
        test.compile_jit(target, true);
    else
        test.compile_jit(target, false);

    double best = benchmark(5, 3, [&]() { test.realize({loss, scores}); scores.copy_to_host();});
    std::cout << "runtime: " << best * 1e3 << std::endl;
}
