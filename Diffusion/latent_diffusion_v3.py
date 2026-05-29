import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# AUTOENCODER (latent feature map, NOT vector)
# ============================================================

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=2, base=32):
        super().__init__()

        self.down1 = ConvBlock(in_channels, base)
        self.down2 = ConvBlock(base, base * 2)
        self.down3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool3d(2) # each time the D, H, W are divided by 2

    def forward(self, x):
        x = self.down1(x) 
        x = self.down2(self.pool(x)) # pooling results in D/2, H/2, W/2
        x = self.down3(self.pool(x)) # second pooling results in D/4, H/4, W/4
        return x #, (x1, x2) keep skips commented out as they are not used for the decoder


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=2, base=32):
        super().__init__()

        # input: base*4 latent
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 2, base * 2)   
        
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base, base)          

        self.out_img = nn.Conv3d(base, 1, 1)
        self.out_mask = nn.Conv3d(base, 1, 1)

    def forward(self, x):

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        img = self.out_img(x)
        mask = torch.sigmoid(self.out_mask(x))

        return torch.cat([img, mask], dim=1)

class AutoEncoderModel(nn.Module):
    def __init__(self, in_channels=2, base=32):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base)
        self.decoder = UNetDecoder(base=base)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AutoencoderLoss(nn.Module):
    def __init__(self, lambda_mask=0.1):
        super().__init__()
        self.lambda_mask = lambda_mask

    def forward(self, recon, target):

        img_loss = F.l1_loss(
            recon[:, 0:1],
            target[:, 0:1]
        )

        mask_loss = F.binary_cross_entropy(
            recon[:, 1:2].clamp(1e-6, 1 - 1e-6),
            target[:, 1:2]
        )

        return img_loss + self.lambda_mask * mask_loss


# ============================================================
# TIME EMBEDDING
# ============================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) *
            torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ============================================================
# REAL 3D CONDITIONAL DIFFUSION U-NET (FIXED)
# ============================================================

class FiLM(nn.Module):
    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        self.to_gamma = nn.Conv3d(cond_channels, feature_channels, 1)
        self.to_beta  = nn.Conv3d(cond_channels, feature_channels, 1)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return x * (1 + gamma) + beta


class DiffusionUNet(nn.Module):
    """
    FiLM-conditioned diffusion U-Net:
    P(z_target | z_source)
    """

    def __init__(self, channels=32, cond_channels=32, base=64, time_dim=128):
        super().__init__()

        self.time_emb = SinusoidalEmbedding(time_dim)

        # input: z_t + time
        in_ch = channels + time_dim

        # -----------------------------
        # condition encoder (CRITICAL)
        # -----------------------------
        self.cond_encoder = nn.Sequential(
            nn.Conv3d(cond_channels, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base, base, 3, padding=1),
            nn.SiLU(),
        )

        # -----------------------------
        # Encoder
        # -----------------------------
        self.conv1 = ConvBlock(in_ch, base)
        self.film1 = FiLM(base, base)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = ConvBlock(base, base * 2)
        self.film2 = FiLM(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        # -----------------------------
        # Bottleneck
        # -----------------------------
        self.mid = ConvBlock(base * 2, base * 2)
        self.mid_film = FiLM(base, base * 2)

        # -----------------------------
        # Decoder
        # -----------------------------
        self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec2 = ConvBlock(base * 3, base)
        self.film_dec2 = FiLM(base, base)

        self.up1 = nn.ConvTranspose3d(base, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.film_dec1 = FiLM(base, base)

        self.out = nn.Conv3d(base, channels, 1)

    def forward(self, z_t, z_source, t):

        # -------------------------
        # time embedding
        # -------------------------
        t_emb = self.time_emb(t)[:, :, None, None, None]
        t_emb = t_emb.expand(-1, -1, *z_t.shape[2:])

        # -------------------------
        # condition encoding
        # -------------------------
        cond = self.cond_encoder(z_source)

        # multi-scale conditioning
        cond1 = cond
        cond2 = F.avg_pool3d(cond, 2)
        cond3 = F.avg_pool3d(cond, 4)

        # -------------------------
        # input
        # -------------------------
        h = torch.cat([z_t, t_emb], dim=1)

        # -------------------------
        # encoder
        # -------------------------
        h1 = self.conv1(h)
        h1 = self.film1(h1, cond1)
        h1p = self.pool1(h1)

        h2 = self.conv2(h1p)
        h2 = self.film2(h2, cond2)
        h2p = self.pool2(h2)

        # -------------------------
        # bottleneck
        # -------------------------
        mid = self.mid(h2p)
        mid = self.mid_film(mid, cond3)

        # -------------------------
        # decoder
        # -------------------------
        x = self.up2(mid)
        x = self.dec2(torch.cat([x, h2], dim=1))
        x = self.film_dec2(x, cond2)

        x = self.up1(x)
        x = self.dec1(torch.cat([x, h1], dim=1))
        x = self.film_dec1(x, cond1)

        return self.out(x)

# ============================================================
# DIFFUSION SCHEDULER (FIXED BROADCASTING)
# ============================================================

class DiffusionScheduler(nn.Module):
    def __init__(self, T=1000):
        super().__init__()

        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)
        self.T = T

    def q_sample(self, x0, noise, t):
        a = self.alpha_hat[t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


# ============================================================
# FULL MODEL (DiffusionCT CORRECT FORM)
# ============================================================

class ReconstructionModel(nn.Module):
    """
    DiffusionCT-style conditional latent diffusion
    """

    def __init__(self, in_channels=2, base=32):
        super().__init__()

        self.autoencoder = AutoEncoderModel(in_channels, base)

        self.diffusion = DiffusionUNet(
            channels=base * 4,
            cond_channels=base * 4,
        )

        self.scheduler = DiffusionScheduler()

    def freeze_autoencoder(self):

        self.autoencoder.eval()

        for p in self.autoencoder.parameters():
            p.requires_grad = False

    def load_pretrained_autoencoder(self, autoencoder_model):
        """
        Copy pretrained autoencoder weights.
        """

        self.autoencoder.load_state_dict(autoencoder_model.state_dict())

    def forward(self, ap, pvp, mode="ap_to_pvp"):

        # ====================================================
        # Encode
        # ====================================================

        with torch.no_grad():

            z_ap = self.autoencoder.encoder(ap)

            z_pvp = self.autoencoder.encoder(pvp)

        # ====================================================
        # Direction selection
        # ====================================================

        if mode == "ap_to_pvp":

            z_source = z_ap
            z_target = z_pvp
            # target_skips = skips_pvp

        elif mode == "pvp_to_ap":

            z_source = z_pvp
            z_target = z_ap
            # target_skips = skips_ap

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ====================================================
        # Diffusion
        # ====================================================

        B = z_target.shape[0]           # z_pvp when ap_to_pvp

        t = torch.randint(0, self.scheduler.T, (B,), device=ap.device,)

        noise = torch.randn_like(z_target)

        z_t = self.scheduler.q_sample(z_target, noise, t)  # get the noisy latent given the timesteps and the noise added

        pred_noise = self.diffusion(z_t, z_source, t)

        # ====================================================
        # Reconstruct latent
        # ====================================================

        a = self.scheduler.alpha_hat[t].view(B, 1, 1, 1, 1)

        z_pred = (z_t - torch.sqrt(1 - a) * pred_noise) / torch.sqrt(a)

        # ====================================================
        # Decode reconstructed latent
        # ====================================================

        with torch.no_grad():

            recon = self.autoencoder.decoder(z_pred)

        return pred_noise, noise, z_pred, z_target, recon
    
    @torch.no_grad()
    def generate_image(
        self,
        source,
        mode="ap_to_pvp",
    ):
        """
        Generate translated phase volume using reverse diffusion.

        Args:
            source:
                Tensor [B, 2, D, H, W]
                source phase (AP or PVP)

            mode:
                "ap_to_pvp" or "pvp_to_ap"

        Returns:
            recon:
                Generated image [B, 2, D, H, W]
        """

        self.eval()

        device = source.device

        # ========================================================
        # Encode source
        # ========================================================

        z_source  = self.autoencoder.encoder(source)

        # ========================================================
        # Initialize latent with pure Gaussian noise
        # ========================================================

        z = torch.randn_like(z_source)

        # ========================================================
        # Reverse diffusion process
        # ========================================================

        for t in reversed(range(self.scheduler.T)):

            t_tensor = torch.full(
                (z.shape[0],),
                t,
                device=device,
                dtype=torch.long,
            )

            # --------------------------------------------
            # Predict noise
            # --------------------------------------------

            pred_noise = self.diffusion(z, z_source, t_tensor)

            # --------------------------------------------
            # DDPM reverse step
            # --------------------------------------------


            alpha_t = self.scheduler.alphas[t]
            alpha_hat_t = self.scheduler.alpha_hat[t]
            beta_t = self.scheduler.betas[t]

            alpha_t = alpha_t.view(1,1,1,1,1)
            alpha_hat_t = alpha_hat_t.view(1,1,1,1,1)
            beta_t = beta_t.view(1,1,1,1,1)

            # ----------------------------
            # predict x0
            # ----------------------------
            x0_pred = (z - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)

            # ----------------------------
            # DDPM posterior mean (CORRECT FORM)
            # ----------------------------
            mean = (
                torch.sqrt(alpha_t) * z
                + (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * x0_pred
            )

            # ----------------------------
            # noise
            # ----------------------------
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)

            z = mean + torch.sqrt(beta_t) * noise

        # ========================================================
        # Decode final latent
        # ========================================================

        recon = self.autoencoder.decoder(z)

        return recon


# ============================================================
# LOSS
# ============================================================

class ReconstructionLoss(nn.Module):
    def forward(self, pred_noise, true_noise):
        return F.mse_loss(pred_noise, true_noise)