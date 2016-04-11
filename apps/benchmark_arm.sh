#!/bin/bash
threads=$1
rm -f fig8_auto.txt
rm -f fig8_naive.txt
rm -f fig8_ref.txt

touch fig8_auto.txt
touch fig8_naive.txt
touch fig8_ref.txt
echo "bilateral_grid:" >> fig8_auto.txt
echo "bilateral_grid:" >> fig8_naive.txt
echo "bilateral_grid:" >> fig8_ref.txt
cd bilateral_grid
make clean
make filter_auto
make filter_naive
make filter_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "blur:" >> fig8_auto.txt
echo "blur:" >> fig8_naive.txt
echo "blur:" >> fig8_ref.txt
cd blur
make clean
make test_auto
make test_naive
make test_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "camera_pipe:" >> fig8_auto.txt
echo "camera_pipe:" >> fig8_naive.txt
echo "camera_pipe:" >> fig8_ref.txt
cd camera_pipe
make clean
make process_auto
make process_naive
make process_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "conv_layer:" >> fig8_auto.txt
echo "conv_layer:" >> fig8_naive.txt
echo "conv_layer:" >> fig8_ref.txt
cd conv_layer
make clean
make conv_bench
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "harris:" >> fig8_auto.txt
echo "harris:" >> fig8_naive.txt
echo "harris:" >> fig8_ref.txt
cd harris
make clean
make filter_auto
make filter_naive
make filter_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "hist:" >> fig8_auto.txt
echo "hist:" >> fig8_naive.txt
echo "hist:" >> fig8_ref.txt
cd hist
make clean
make filter_auto
make filter_naive
make filter_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -


echo "interpolate:" >> fig8_auto.txt
echo "interpolate:" >> fig8_naive.txt
echo "interpolate:" >> fig8_ref.txt
cd interpolate
make clean
make interpolate
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "lens_blur:" >> fig8_auto.txt
echo "lens_blur:" >> fig8_naive.txt
echo "lens_blur:" >> fig8_ref.txt
cd lens_blur
make clean
make lens_blur
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "local_laplacian:" >> fig8_auto.txt
echo "local_laplacian:" >> fig8_naive.txt
echo "local_laplacian:" >> fig8_ref.txt
cd local_laplacian
make clean
make process_auto
make process_naive
make process_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "mat_mul:" >> fig8_auto.txt
echo "mat_mul:" >> fig8_naive.txt
echo "mat_mul:" >> fig8_ref.txt
cd mat_mul
make clean
make mat_mul
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "max_filter:" >> fig8_auto.txt
echo "max_filter:" >> fig8_naive.txt
echo "max_filter:" >> fig8_ref.txt
cd max_filter
make clean
make max_filter
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "non_local_means:" >> fig8_auto.txt
echo "non_local_means:" >> fig8_naive.txt
echo "non_local_means:" >> fig8_ref.txt
cd non_local_means
make clean
make filter_auto
make filter_naive
make filter_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "unsharp:" >> fig8_auto.txt
echo "unsharp:" >> fig8_naive.txt
echo "unsharp:" >> fig8_ref.txt
cd unsharp
make clean
make filter_auto
make filter_naive
make filter_ref
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -

echo "vgg:" >> fig8_auto.txt
echo "vgg:" >> fig8_naive.txt
echo "vgg:" >> fig8_ref.txt
cd vgg
make clean
make vgg
./test.sh ref $threads | tail -1 >> ../fig8_ref.txt
./test.sh auto $threads | tail -1 >> ../fig8_auto.txt
./test.sh naive $threads | tail -1 >> ../fig8_naive.txt
cd -
