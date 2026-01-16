import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def plot_triplets(triplets, left_row_names, right_row_names,col_names=("Frame", "Ground truth", "SAMAnnot"),fig_dpi=200,save_path=None,save_dpi=400,w_gap=0.01,h_gap=0.2,header_gap=0.0,show=False):
    fig, axes = plt.subplots(len(triplets), 3,figsize=(12,2.75*len(triplets)),dpi=fig_dpi,gridspec_kw={"wspace": w_gap, "hspace": h_gap},)
    for r in range(len(triplets)):
        for c in range(3):
            img = mpimg.imread(triplets[r][c])
            axes[r, c].imshow(img, aspect="auto")
            axes[r, c].set_axis_off()
        axes[r, 0].text(-0.025, 0.5, left_row_names[r],transform=axes[r, 0].transAxes,rotation=90,va="center", ha="right",fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.03, top=0.90)
    for j, cname in enumerate(col_names):
        pos = axes[0, j].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_text = pos.y1 + header_gap
        fig.text(x_center, y_text, cname, ha="center", va="bottom", fontsize=18)
    for r in range(len(triplets)):
        pos = axes[r, 2].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_text = pos.y0 - 0.005

        t = fig.text(
            x_center, y_text, right_row_names[r],
            ha="center", va="top",
            fontsize=18,
            zorder=100,
        )
        t.set_clip_on(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)
if __name__ == "__main__":
    triplets = [
        ("./drone_00006_fr.png", "./drone_00006_gt.png", "./drone_00006_pr.png"),
        ("./boxing_fisheye_00012_fr.png", "./boxing_fisheye_00012_gt.png", "./boxing_fisheye_00012_pr.png"),
        ("./schoolgirls_00012_fr.png", "./schoolgirls_00012_gt.png", "./schoolgirls_00012_pr.png"),
    ]
    row_names = [
        "drone",
        "boxing-fisheye",
        "schoolgirls",
    ]
    iou_val = [
        "mIoU: 0.8188",
        "mIoU: 0.9099",
        "mIoU: 0.7473",
    ]
    triplets = [
        ("./bear_00000_fr.png", "./bear_00000_gt.png", "./bear_00000_pr.png"),
        ("./sheep_00000_fr.png", "./sheep_00000_gt.png", "./sheep_00000_pr.png"),
        ("./color_run_00006_fr.png", "./color_run_00006_gt.png", "./color_run_00006_pr.png"),
        ("./classic_car_00043_fr.png", "./classic_car_00043_gt.png", "./classic_car_00043_pr.png"),
        ("./cat_girl_00001_fr.png", "./cat_girl_00001_gt.png", "./cat_girl_00001_pr.png"),
        ("./bus_00000_fr.png", "./bus_00000_gt.png", "./bus_00000_pr.png"),
        ("./pigs_00046_fr.png", "./pigs_00046_gt.png", "./pigs_00046_pr.png"),
    ]
    row_names = [
        "bear",
        "sheep",
        "color-run",
        "classic-car",
        "cat-girl",
        "bus",
        "pigs",
    ]
    iou_val = [
        "mIoU: 0.9703",
        "mIoU: 0.8361",
        "mIoU: 0.9252",
        "mIoU: 0.9265",
        "mIoU: 0.9460",
        "mIoU: 0.9295",
        "mIoU: 0.8914",
    ]
    plot_triplets(triplets,row_names, iou_val,col_names=("Frame", "Ground truth (DAVIS)", "SAMAnnot"),w_gap=0.01,h_gap=0.175,save_path="grid_u_app.png",save_dpi=250,header_gap=0.0)