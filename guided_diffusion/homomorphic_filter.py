import torch
import torch.nn as nn

class LearnableHomomorphicFilter(nn.Module):
    def __init__(self, gamma_low_init=0.3, gamma_high_init=3, cutoff_init=20, c=1.5, device='cuda', verbose=True):
        super().__init__()
        self.device = device
        self.gamma_high = nn.Parameter(torch.tensor([gamma_high_init], dtype=torch.float32, device=device))
        self.cutoff     = nn.Parameter(torch.tensor([cutoff_init],     dtype=torch.float32, device=device))
        self.gamma_low  = nn.Parameter(torch.tensor([gamma_low_init],  dtype=torch.float32, device=device))
        self.c = float(c)
        self.verbose = verbose

        self.register_buffer("_init_gamma_high", torch.tensor([gamma_high_init], dtype=torch.float32, device=device))
        self.register_buffer("_init_cutoff",     torch.tensor([cutoff_init],     dtype=torch.float32, device=device))
        self.register_buffer("_init_gamma_low",  torch.tensor([gamma_low_init],  dtype=torch.float32, device=device))
        self.register_buffer("_last_t", torch.tensor([-1], dtype=torch.int64, device=device))

    @torch.no_grad()
    def _reset_params_to_init(self):
        self.gamma_low.data.copy_(self._init_gamma_low)
        self.gamma_high.data.copy_(self._init_gamma_high)
        self.cutoff.data.copy_(self._init_cutoff)

    def forward(self, x, t_val: int):
        if (self._last_t.item() == 0) and (int(t_val) == 9):
            if self.verbose:
                print("Start sampling new image, init params.")
            self._reset_params_to_init()

        self._last_t.fill_(int(t_val))

        if self.verbose:
            print(f"[Learnable Param] gL={self.gamma_low.item():.4f}, gH={self.gamma_high.item():.4f}, D0={self.cutoff.item():.4f}")

        B, C, H, W = x.shape
        x_in = (x + 1.0) / 2.0
        x_fft = torch.fft.fft2(x_in, dim=(-2, -1))
        amp, phase = torch.abs(x_fft), torch.angle(x_fft)

        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing='ij')
        d2 = (yy - H//2)**2 + (xx - W//2)**2

        gL, gH, D0 = self.gamma_low, self.gamma_high, self.cutoff
        H_uv = (gH - gL) * (1 - torch.exp(-self.c * d2 / (D0**2 + 1e-12))) + gL
        H_uv = H_uv.unsqueeze(0).unsqueeze(0)
        dc = gL.clamp_min(1e-6)
        H_uv = H_uv / dc

        amp_filtered = amp * H_uv
        x_filtered = torch.fft.ifft2(amp_filtered * torch.exp(1j * phase), dim=(-2, -1))
        x_filtered = torch.abs(x_filtered)

        max_per = x_filtered.amax(dim=2, keepdim=True).amax(dim=3, keepdim=True)
        x_filtered = x_filtered / (max_per + 1e-8)
        x_out = x_filtered * 2.0 - 1.0
        return x_out



















