import matplotlib.pyplot as plt
times = {}
for app in ["blur/", "local_laplacian/", "interpolate/", "bilateral_grid/", \
            "camera_pipe/"] :

    f = open(app+"ref_perf.txt")
    times[app] = [ float(l) for l in f ]
    print times[app]
