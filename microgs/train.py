import torch
import render
import gaussian
import utils
import camera

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    H, W_img = 64, 64
    cam = camera.default_camera(H, W_img)
    bg=torch.zeros(3)

    target_p = gaussian.make_params(mu=[0.4, -0.2, 0.0], log_scale=[-0.7, -2.0, -2.0],
                           quat=[0.92, 0.0, 0.0, 0.38], opacity_logit=2.0,
                           rgb_logit=[2.0, 0.3, -2.0])

    # no_grad turns off recording for those methods (not used in autograd)
    with torch.no_grad():
        target = render.render(target_p, cam, H, W_img, bg)
    
    # Starting gaussian, randomly initialized:
    p = gaussian.make_params(mu=[-0.3, 0.3, 0.0], log_scale=[-1.5, -1.5, -1.5],
                quat=[1.0, 0.0, 0.0, 0.0], opacity_logit=0.0,
                rgb_logit=[0.0, 0.0, 0.0])

    initial_img = render.render(p, cam, H, W_img, bg).detach()

    # fit the Gaussian with ADAM:
    opt = torch.optim.Adam(p.values(), lr=0.05)
    for it in range(400):
        # Reset the grads so that previous addition doesnt stack
        opt.zero_grad()

        loss = ((render.render(p, cam, H, W_img, bg) - target) ** 2).mean()
        loss.backward()
        opt.step()

        if it % 100 == 0 or it == 399:
            print(f"iter {it:3d}  loss {loss.item():.2e}")

    final_img = render.render(p, cam, H, W_img, bg).detach()

    utils.export_results(initial_img, final_img, target)

