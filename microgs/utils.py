import torch
import render

def export_results(initial_img, final_img, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    for a, im, t in zip(ax, [initial_img, final_img, target],
                        ["Initial", "Post-Training", "Target"]):
        a.imshow(im.clamp(0, 1).numpy()); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig("render.png", dpi=120)
    print("render.png")

def log_finite_compare(p, cam, H, W_img, bg, target, eps=1e-6):
    # Compare with Finite Differences:
    loss = ((render.render(p, cam, H, W_img, bg) - target) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        p["mu"] += eps
        top = ((render.render(p, cam, H, W_img, bg) - target) ** 2).mean()
        p["mu"] -= 2 * eps
        bot = ((render.render(p, cam, H, W_img, bg) - target) ** 2).mean()
        p["mu"] += eps
    print(f"autograd:{p['mu'].grad[0].item():.6e}, finite diff:{((top-bot)/(2*eps)).item():.6e}")
