import torch

torch.set_default_dtype(torch.float64)


# Claude-Generated: makes a dictionary for the parameters of a given Gaussian
def make_params(mu, log_scale, quat, opacity_logit, rgb_logit):
    return {k: torch.tensor(v, requires_grad=True) for k, v in dict(
        mu=mu, log_scale=log_scale, quat=quat,
        opacity_logit=opacity_logit, rgb_logit=rgb_logit).items()}

# Claude-Generated: quaternion -> 3x3 rotation matrix
# The quaternion is what is stored and trained per-guassian,
# we need to turn it in to something we can render
def quat_to_rotmat(q):
    w, x, y, z = q / q.norm()
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]),
    ])

# Get sigma: T this matrix is PSD because of the quadratic form
def covariance(log_scale, quat):
    # cov = rot * scale * scaleT * rotT
    R = quat_to_rotmat(quat)
    S = torch.diag(torch.exp(log_scale))
    return R @ S @ S.T @ R.T

def project(mu, sigma, cam_rot, cam_trans, focal_x, focal_y, cx, cy):
    t = cam_rot @ mu + cam_trans; # center in world coords
    tx, ty, tz = t;

    # pinhole projection
    mean2d = torch.stack([focal_x * tx / tz + cx, focal_y * ty / tz + cy])

    # ???
    J = torch.stack([                              # d(pixel)/d(camera point), evaluated at t
        torch.stack([focal_x / tz, torch.zeros(()), -focal_x * tx / tz**2]),
        torch.stack([torch.zeros(()), focal_y / tz, -focal_y * ty / tz**2]),
    ])

    J_rotated = J @ cam_rot
    cov2d = J_rotated @ sigma @ J_rotated.T

    # add 0.3 px^2 in each dir so the blob is never smaller than a pixel
    cov2d = cov2d + 0.3 * torch.eye(2) 

    # also return tz for alpha culling
    return mean2d, cov2d, tz


def render(params, cam, H, W_img, bg):
    Sigma = covariance(params["log_scale"], params["quat"])
    mean2d, cov2d, depth = project(params["mu"], Sigma, **cam)

    #???
    conic = torch.linalg.inv(cov2d)

    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W_img), indexing="ij")
    # move from pixel corner to center
    d = torch.stack([xs, ys], -1) + 0.5 - mean2d

    # compute squared Mahalanobis distance: dT inv(sigma') d
    # power = -0.5 (a*dx^2 + 2b*dx*dy + c*dy^2)
    power = -0.5 * torch.einsum("hwi,ij,hwj->hw", d, conic, d)
    opacity = torch.sigmoid(params["opacity_logit"])
    alpha = opacity * torch.exp(power)
    # alpha(p) = o * exp(-1/2 d^T inv(Sigma') d)
    
    # if the object is too close, just set alpha = 0 (transparent)
    if depth <= 0.2:
        alpha = torch.zeros_like(alpha)

    # stand-in for SH degree 0
    color = torch.sigmoid(params["rgb_logit"])

    img = alpha[..., None] * color + (1-alpha[...,None]) * bg
    return img

# ----------------------------------------------------------------------

def default_camera(H, W_img):
    return dict(cam_rot=torch.eye(3), cam_trans=torch.tensor([0., 0., 4.]),   # camera 4 units back
                focal_x=60.0, focal_y=60.0, cx=W_img/2, cy=H/2)


if __name__ == "__main__":
    H, W_img = 64, 64
    cam = default_camera(H, W_img)
    bg=torch.zeros(3)

    target_p = make_params(mu=[0.4, -0.2, 0.0], log_scale=[-0.7, -2.0, -2.0],
                           quat=[0.92, 0.0, 0.0, 0.38], opacity_logit=2.0,
                           rgb_logit=[2.0, 0.3, -2.0])

    # no_grad turns off recording for those methods (not used in autograd)
    with torch.no_grad():
        target = render(target_p, cam, H, W_img, bg)
    
    # Starting gaussian, randomly initialized:
    p = make_params(mu=[-0.3, 0.3, 0.0], log_scale=[-1.5, -1.5, -1.5],
                quat=[1.0, 0.0, 0.0, 0.0], opacity_logit=0.0,
                rgb_logit=[0.0, 0.0, 0.0])

    # get our initial image for later comparison
    initial_img = render(p, cam, H, W_img, bg).detach()

    # Compare with Finite Differences:
    # MSE
    loss = ((render(p, cam, H, W_img, bg) - target) ** 2).mean()
    loss.backward()
    eps = 1e-6
    with torch.no_grad():
        p["mu"] += eps
        top = ((render(p, cam, H, W_img, bg) - target) ** 2).mean()
        p["mu"] -= 2 * eps
        bot = ((render(p, cam, H, W_img, bg) - target) ** 2).mean()
        p["mu"] += eps
    print(f"autograd:{p['mu'].grad[0].item():.6e}, finite diff:{((top-bot)/(2*eps)).item():.6e}")


    # fit the Gaussian with ADAM:
    opt = torch.optim.Adam(p.values(), lr=0.05)
    for it in range(400):
        # Reset the grads so that previous addition doesnt stack
        opt.zero_grad()
        # Forward prop
        loss = ((render(p, cam, H, W_img, bg) - target) ** 2).mean()
        # Backprop
        loss.backward()
        # Take our adam step
        opt.step()

        if it % 100 == 0 or it == 399:
            print(f"iter {it:3d}  loss {loss.item():.2e}")

    final_img = render(p, cam, H, W_img, bg).detach()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    for a, im, t in zip(ax, [initial_img, final_img, target],
                        ["Initial", "Adam, 400 steps", "Target"]):
        a.imshow(im.clamp(0, 1).numpy()); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig("render.png", dpi=120)
    print("render.png")
