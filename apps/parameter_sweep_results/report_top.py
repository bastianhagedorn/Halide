import sys
import os
for app in ['bilateral_grid', 'blur', 'camera_pipe', 'conv_layer', 'harris', 'hist', \
            'interpolate', 'lens_blur', 'local_laplacian', 'mat_mul', 'max_filter', \
            'non_local_means', 'unsharp', 'vgg']:
    os.chdir(app)
    os.system('python ../../min_run.py > sorted')
    print app
    os.system('head -1 sorted')
    os.chdir('../')
