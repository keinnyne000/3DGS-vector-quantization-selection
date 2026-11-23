import torch

def default_camera(H, W_img):
    return dict(cam_rot=torch.eye(3), cam_trans=torch.tensor([0., 0., 4.]),   # camera 4 units back
                focal_x=60.0, focal_y=60.0, cx=W_img/2, cy=H/2)
