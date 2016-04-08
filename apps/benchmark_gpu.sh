#!/bin/bash
rm -f gpu_auto.txt
rm -f gpu_naive.txt
rm -f gpu_ref.txt

touch gpu_auto.txt
touch gpu_naive.txt
touch gpu_ref.txt
echo "bilateral_grid:" >> gpu_auto.txt
echo "bilateral_grid:" >> gpu_naive.txt
echo "bilateral_grid:" >> gpu_ref.txt
cd bilateral_grid
make clean
make filter_auto_gpu
make filter_naive_gpu
make filter_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "blur:" >> gpu_auto.txt
echo "blur:" >> gpu_naive.txt
echo "blur:" >> gpu_ref.txt
cd blur
make clean
make test_auto_gpu
make test_naive_gpu
make test_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "camera_pipe:" >> gpu_auto.txt
echo "camera_pipe:" >> gpu_naive.txt
echo "camera_pipe:" >> gpu_ref.txt
cd camera_pipe
make clean
make process_auto_gpu
make process_naive_gpu
make process_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "conv_layer:" >> gpu_auto.txt
echo "conv_layer:" >> gpu_naive.txt
echo "conv_layer:" >> gpu_ref.txt
cd conv_layer
make clean
make conv_bench
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "harris:" >> gpu_auto.txt
echo "harris:" >> gpu_naive.txt
echo "harris:" >> gpu_ref.txt
cd harris
make clean
make filter_auto_gpu
make filter_naive_gpu
make filter_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "hist:" >> gpu_auto.txt
echo "hist:" >> gpu_naive.txt
echo "hist:" >> gpu_ref.txt
cd hist
make clean
make filter_auto_gpu
make filter_naive_gpu
make filter_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -


echo "interpolate:" >> gpu_auto.txt
echo "interpolate:" >> gpu_naive.txt
echo "interpolate:" >> gpu_ref.txt
cd interpolate
make clean
make interpolate
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "lens_blur:" >> gpu_auto.txt
echo "lens_blur:" >> gpu_naive.txt
echo "lens_blur:" >> gpu_ref.txt
cd lens_blur
make clean
make lens_blur
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "local_laplacian:" >> gpu_auto.txt
echo "local_laplacian:" >> gpu_naive.txt
echo "local_laplacian:" >> gpu_ref.txt
cd local_laplacian
make clean
make process_auto_gpu
make process_naive_gpu
make process_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "mat_mul:" >> gpu_auto.txt
echo "mat_mul:" >> gpu_naive.txt
echo "mat_mul:" >> gpu_ref.txt
cd mat_mul
make clean
make mat_mul
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "max_filter:" >> gpu_auto.txt
echo "max_filter:" >> gpu_naive.txt
echo "max_filter:" >> gpu_ref.txt
cd max_filter
make clean
make max_filter
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "non_local_means:" >> gpu_auto.txt
echo "non_local_means:" >> gpu_naive.txt
echo "non_local_means:" >> gpu_ref.txt
cd non_local_means
make clean
make filter_auto_gpu
make filter_naive_gpu
make filter_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "unsharp:" >> gpu_auto.txt
echo "unsharp:" >> gpu_naive.txt
echo "unsharp:" >> gpu_ref.txt
cd unsharp
make clean
make filter_auto_gpu
make filter_naive_gpu
make filter_ref
./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -

echo "vgg:" >> gpu_auto.txt
echo "vgg:" >> gpu_naive.txt
echo "vgg:" >> gpu_ref.txt
cd vgg
make clean
make vgg
#./test.sh ref | tail -1 >> ../gpu_ref.txt
./test.sh auto_gpu | tail -1 >> ../gpu_auto.txt
./test.sh naive_gpu | tail -1 >> ../gpu_naive.txt
cd -
