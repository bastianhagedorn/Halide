import matplotlib.pyplot as plt
times = {}
for mode in ["ref_perf.txt", "auto_perf.txt"]:
    for app in ["blur/", "local_laplacian/", "interpolate/", "bilateral_grid/", \
                "camera_pipe/"] :

        f = open(app+mode)
        times[app] = [ float(l) for l in f ]
        print times[app]
    print
