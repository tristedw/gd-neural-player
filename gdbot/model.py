import torch
import torch.nn as nn

from .vision import OBS_H, OBS_W, STACK


class DashNet(nn.Module):
    """Small CNN: stack of frames in, logit for 'hold the button' out."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, STACK, OBS_H, OBS_W)).numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(path, device):
    net = DashNet().to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state["model"] if "model" in state else state)
    net.eval()
    return net


def save(net, path, **meta):
    torch.save({"model": net.state_dict(), **meta}, path)
