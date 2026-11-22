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
    mean2d = torch.stack([focal_x * tx / tz * cx, focal_y * ty / tz * cy])

    # ???
    J = torch.stack([                              # d(pixel)/d(camera point), evaluated at t
        torch.stack([fx / tz, torch.zeros(()), -fx * tx / tz**2]),
        torch.stack([torch.zeros(()), fy / tz, -fy * ty / tz**2]),
    ])

    J_rotated = J @ cam_rot
    cov2d = J_rotated @ Sigma @ J_rotated.T

    # add 0.3 px^2 in each dir so the blob is never smaller than a pixel
    cov2d = cov2d + 0.3 * torch.eye(2) 

    # also return tz for alpha culling
    return mean2d, cov2d, tz


def render(params, cam, H, w_img, bg):
    Sigma = covariance(params["log_scale"], params["quat"])
    mean2d, cov2, tz = project(params["mu"], Sigma, **cam)

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

# Claude-Generated: Makes a dict with camera vars
def default_camera(H, W_img):
    return dict(W=torch.eye(3), tvec=torch.tensor([0., 0., 4.]),   # camera 4 units back
                fx=60.0, fy=60.0, cx=W_img/2, cy=H/2)


if __name__ == "__main__":
    H, W_img = 64, 64
    cam = default_camera(H, W_img)
    bg=torch.zeros(3)


