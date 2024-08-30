import argparse
import torch
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F
from torch.nn.functional import grid_sample

from tqdm import tqdm
import matplotlib.pyplot as plt

def warp_with_flow(img, flow, device="cuda"):
    B, C, H, W = img.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).to(torch.float32).to(device)
    vgrid = (grid + flow).round().to(torch.int32)

    vgrid_new = []
    mask_remaped = []
    for i in range(B):
        remap_xy = grid[i].detach().cpu().numpy()
        uv_new = vgrid[i].detach().cpu().numpy()
        mask = (uv_new[0] >= 0) & (uv_new[1] >= 0) & (
            uv_new[0] < W) & (uv_new[1] < H)
        uv_new_ = uv_new[:, mask]
        remap_xy[:, uv_new_[1], uv_new_[0]] = remap_xy[:, mask]
        remap_xy = torch.tensor(remap_xy).to(device)
        vgrid_new.append(remap_xy)

        mask = torch.zeros((H, W)).to(torch.bool)
        mask[uv_new_[1], uv_new_[0]] = True
        mask_remaped.append(mask)

    vgrid_new = torch.stack(vgrid_new, dim=0)
    vgrid = vgrid_new
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

    warped_image = grid_sample(img, vgrid.permute(
        0, 2, 3, 1), mode='nearest', padding_mode='zeros', align_corners=True)
    mask_remaped = torch.stack(mask_remaped, dim=0)

    warped_image = warped_image.permute(0, 2, 3, 1)
    warped_image[~mask_remaped] = 0
    warped_image = warped_image.permute(0, 3, 1, 2)

    return warped_image, (~mask_remaped).float()

@torch.no_grad()
def get_mawe(video_path: str, model, coeff):
    video = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
    video = F.resize(video, size=[720, 720], antialias=False)
    _, _, height, width = video.shape

    def preprocess(left_frames, right_frames):
        left_frames = F.resize(left_frames, size=[height, width], antialias=False)
        right_frames = F.resize(right_frames, size=[height, width], antialias=False)
        return transforms(left_frames, right_frames)

    left_frames, right_frames = video[:-1], video[1:]

    warp_errors = []
    motions = []
    for video_idx in range(len(left_frames)):
        left_frame, right_frame = left_frames[video_idx:video_idx+1], right_frames[video_idx:video_idx+1]
        left_frame, right_frame = left_frame.to(device), right_frame.to(device)
        left_tensor, right_tensor = preprocess(left_frame, right_frame)
        flows = model(left_tensor.to(device), right_tensor.to(device))[-1]

        norms = torch.linalg.vector_norm(flows, dim=1)
        mean_of = torch.mean(norms) #/ (height * width)
        motions.append(mean_of.item())
        
        warped, masks = warp_with_flow(left_frame.to(torch.float32), flows)
        known_regions = 1 - \
            masks.unsqueeze(1).repeat(1, 3, 1, 1).to(torch.float32).to(device)
        diff = (right_frame - warped) * known_regions

        N = torch.sum(known_regions)
        if N == 0:
            N = diff.shape[0] * diff.shape[1] * diff.shape[2] * diff.shape[3]
        warp_errors.append((torch.sum(torch.pow(diff, 2)) / N).item())
    warp_errors = torch.tensor(warp_errors)
    motions = torch.tensor(motions)
    return (warp_errors / (coeff * motions)).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--coeff", default=9.5, type=float)
    args = parser.parse_args()

    coeff = args.coeff
    video_path = args.video_path

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    mawe = get_mawe(video_path.as_posix(), model, coeff)
    print(f"MAWE for {video_path} is {mawe:0.2f}")
