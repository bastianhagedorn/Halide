#!/bin/bash
rm -f gpu_auto.txt
rm -f gpu_naive.txt
touch gpu_auto.txt
touch gpu_naive.txt
echo "bilateral_grid:" >> gpu_auto.txt
echo "bilateral_grid:" >> gpu_naive.txt
cd bilateral_grid
make clean
make filter_auto_gpu
make filter_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "blur:" >> gpu_auto.txt
echo "blur:" >> gpu_naive.txt
cd blur
make clean
make test_auto_gpu
make test_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "camera_pipe:" >> gpu_auto.txt
echo "camera_pipe:" >> gpu_naive.txt
cd camera_pipe
make clean
make process_auto_gpu
make process_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "conv_layer:" >> gpu_auto.txt
echo "conv_layer:" >> gpu_naive.txt
cd conv_layer
make clean
make conv_bench
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "harris:" >> gpu_auto.txt
echo "harris:" >> gpu_naive.txt
cd harris
make clean
make filter_auto_gpu
make filter_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "hist:" >> gpu_auto.txt
echo "hist:" >> gpu_naive.txt
cd hist
make clean
make filter_auto_gpu
make filter_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -


echo "interpolate:" >> gpu_auto.txt
echo "interpolate:" >> gpu_naive.txt
cd interpolate
make clean
make interpolate
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "lens_blur:" >> gpu_auto.txt
echo "lens_blur:" >> gpu_naive.txt
cd lens_blur
make clean
make lens_blur
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "local_laplacian:" >> gpu_auto.txt
echo "local_laplacian:" >> gpu_naive.txt
cd local_laplacian
make clean
make process_auto_gpu
make process_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "mat_mul:" >> gpu_auto.txt
echo "mat_mul:" >> gpu_naive.txt
cd mat_mul
make clean
make mat_mul
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "max_filter:" >> gpu_auto.txt
echo "max_filter:" >> gpu_naive.txt
cd max_filter
make clean
make max_filter
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "non_local_means:" >> gpu_auto.txt
echo "non_local_means:" >> gpu_naive.txt
cd non_local_means
make clean
make filter_auto_gpu
make filter_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "unsharp:" >> gpu_auto.txt
echo "unsharp:" >> gpu_naive.txt
cd unsharp
make clean
make filter_auto_gpu
make filter_naive_gpu
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "vgg:" >> gpu_auto.txt
echo "vgg:" >> gpu_naive.txt
cd vgg
make clean
make vgg
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -
