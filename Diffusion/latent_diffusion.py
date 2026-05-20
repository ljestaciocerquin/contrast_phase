import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18


class ResNetEncoder(nn.Module):
    """3D ResNet10 encoder for AP and PVP volumes.

    Input:  [batch_size, channels, depth, height, width]
    Output: latent representation [batch_size, latent_dim]
    """

    def __init__(self, latent_dim=512, in_channels=1):
        super().__init__()

        backbone = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            pretrained=False,
        )

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        z = self.fc(x)
        return z


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal time embeddings for conditioning the DDPM on the diffusion step t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class LatentDDPM(nn.Module):
    """DDPM operating in latent space."""

    def __init__(self, latent_dim=512, time_dim=128):
        super().__init__()

        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)

        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim * 2 + time_dim, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, latent_dim),
        )

    def forward(self, z_noisy, z_condition, t):

        t_embed = self.time_embedding(t)

        x = torch.cat([
            z_noisy,
            z_condition,
            t_embed,
        ], dim=1)

        predicted_noise = self.denoiser(x)
        return predicted_noise


class DiffusionScheduler(nn.Module):
    """Defines noise schedule."""

    def __init__(self, timesteps=1000):
        super().__init__()

        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)

    def add_noise(self, z, noise, t):
        """
        z: [B, ...]
        t: [B] (timesteps)
        """
        t = t.long()

        alpha_hat_t = self.alpha_hat[t]

        sqrt_alpha_hat = torch.sqrt(alpha_hat_t).view(-1, *([1] * (z.ndim - 1)))
        sqrt_one_minus = torch.sqrt(1 - alpha_hat_t).view(-1, *([1] * (z.ndim - 1)))

        noisy_z = sqrt_alpha_hat * z + sqrt_one_minus * noise
        return noisy_z


class Decoder3D(nn.Module):
    """Decode a latent vector back to a 3D AP/PVP volume."""

    def __init__(self, latent_dim=512, out_channels=1, output_size=(128, 128, 128)):
        super().__init__()

        self.output_size = output_size
        self.start_shape = (512, 4, 4, 4)

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self.start_shape)

        x = self.decoder(x)

        if x.shape[2:] != self.output_size:
            x = F.interpolate(
                x,
                size=self.output_size,
                mode="trilinear",
                align_corners=False,
            )

        return x


class ReconstructionModel(nn.Module):
    """AP <-> PVP latent diffusion model"""

    def __init__(self, latent_dim=512, in_channels=1, output_size=(128, 128, 128)):
        super().__init__()

        self.encoder = ResNetEncoder(latent_dim, in_channels)
        self.ddpm = LatentDDPM(latent_dim)
        self.scheduler = DiffusionScheduler(1000)
        self.decoder = Decoder3D(latent_dim, in_channels, output_size)

    def forward(self, ap, pvp, mode="ap_to_pvp"):

        z_ap = self.encoder(ap)
        z_pvp = self.encoder(pvp)

        t = torch.randint(
            0,
            self.scheduler.timesteps,
            (z_ap.shape[0],),
            device=ap.device,
        )

        if mode == "ap_to_pvp":
            z_source, z_target = z_ap, z_pvp
        else:
            z_source, z_target = z_pvp, z_ap

        noise = torch.randn_like(z_target) # initialize random noise to be added to the target latent representation

        z_noisy = self.scheduler.add_noise(z_target, noise, t)

        predicted_noise = self.ddpm(z_noisy, z_source, t)

        # z_reconstructed = z_noisy - predicted_noise

        alpha_hat_t = self.scheduler.alpha_hat[t].view(-1, 1)

        z_reconstructed = (z_noisy - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)

        recon = self.decoder(z_reconstructed)

        return recon, predicted_noise, noise, z_ap, z_pvp


    @torch.no_grad()
    def generate(self, source_image, steps=1000):

        self.eval()
        device = source_image.device

        z_condition = self.encoder(source_image)
        z = torch.randn_like(z_condition)

        for t in reversed(range(steps)):

            t_batch = torch.full((z.shape[0],), t, device=device, dtype=torch.float32)

            alpha = self.scheduler.alphas[t]
            alpha_hat = self.scheduler.alpha_hat[t]
            beta = self.scheduler.betas[t]

            predicted_noise = self.ddpm(z, z_condition, t_batch)

            z = (1 / torch.sqrt(alpha)) * (
                z - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            )

            if t > 0:
                z = z + torch.sqrt(beta) * torch.randn_like(z)

        return self.decoder(z)


class ReconstructionLoss(nn.Module):
    def __init__(self, lambda_latent=0.1):
        super().__init__()
        self.lambda_latent = lambda_latent

    def forward(self, reconstructed, target, z_source, z_target):

        image_loss = F.l1_loss(
            reconstructed[:, 0:1],
            target[:, 0:1]
        )

        mask_loss = F.binary_cross_entropy(
            reconstructed[:, 1:2],
            target[:, 1:2]
        )

        return image_loss + self.lambda_latent * mask_loss