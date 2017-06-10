import tfplot

def draw_trajectory(gt, pred):
    a, b = gt, pred
    time_step, coord_size = a.shape
    try:
        ay, ax = a[:, 2], a[:, 3]     # just draw the first player
        by, bx = b[:, 2], b[:, 3]     # just draw the first player
    except:
        ay, ax = a[:, 0], a[:, 1]
        by, bx = b[:, 0], b[:, 1]

    fig, axes = tfplot.subplots(1, 2, figsize=(8, 4))
    # gt
    axes[0].axis([0, 1, 0, 1])
    axes[0].scatter(ax, ay, c=range(time_step), cmap='jet')
    axes[0].set_title("GT")
    # pred
    axes[1].axis([0, 1, 0, 1])
    axes[1].scatter(bx, by, c=range(time_step), cmap='jet')
    axes[1].set_title("pred")
    return fig


def draw_trajectory_multiple(gt, *pred_list):
    time_step, coord_size = gt.shape
    try:
        ay_ax_list = [(a[:, 2], a[:, 3]) for a in [gt] + list(pred_list)]
    except:
        ay_ax_list = [(a[:, 0], a[:, 1]) for a in [gt] + list(pred_list)]

    N = len(ay_ax_list)
    sqrtN = int(N**0.5 + 1e-9)
    H, W = sqrtN, sqrtN
    if H * W < N: W += 1

    fig, axes = tfplot.subplots(H, W, figsize=(12, 12), squeeze=False)
    for h in range(H):
        for w in range(W):
            k = h * H + w
            ax = axes[h, w]
            ax.axis([0, 1, 0, 1])
            ax.scatter(ay_ax_list[k][1], ay_ax_list[k][0], c=range(time_step), cmap='jet')
            if k == 0:
                ax.set_title("GT")
                ax.set_axis_bgcolor('gray')
            else: ax.set_title("pred %d" % k)
    return fig


__all__ = (
    'draw_trajectory',
    'draw_trajectory_multiple',
)
