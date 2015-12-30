import matplotlib.pyplot as plt
times_ref = {}
times_auto = {}
speed_up = {}
p = 1
for app in ["blur/", "unsharp/", "harris/", "local_laplacian/", "interpolate/", "bilateral_grid/", \
            "camera_pipe/", "conv_layer/", "cost_function_test/", "overlap_test/", \
            "split_test/", "tile_vs_inline_test/", "data_dependent_test/",\
            "parallel_test/"] :
        f = open(app+ "ref_perf.txt")
        times_ref[app] = [ float(l) for l in f ]
        f.close()
        f = open(app+ "auto_perf.txt")
        times_auto[app] = [ float(l) for l in f ]
        f.close()
        speed_up[app] = []
        for i in xrange(0, len(times_auto[app])):
            speed_up[app].append(times_ref[app][i]/times_auto[app][i])

        plt.tight_layout()
        ax = plt.subplot(4, 4, p)
        ax.bar([1, 2, 3, 4], speed_up[app], alpha=0.4, align='center')
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speed Up")
        ax.set_ylim([0, 1])
        ax.set_xticks([1, 2, 3 , 4])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(app)
        p = p+1
        print speed_up[app]
plt.show()
