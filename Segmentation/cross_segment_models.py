import torch
import torch.nn as nn


# =========================================================
# BASIC 3D SEGMENTATION BACKBONE
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SimpleSegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)      # [B, 32, D, H, W]
        self.pool1 = nn.MaxPool3d(2)                # [B, 32, D/2, H/2, W/2]

        self.enc2 = ConvBlock(32, 64)               # [B, 64, D/2, H/2, W/2]
        self.pool2 = nn.MaxPool3d(2)                # [B, 64, D/4, H/4, W/4]

        # smallest spatial size, largest channel depth
        self.bottleneck = ConvBlock(64, 128)        # [B, 128, D/4, H/4, W/4]

        # reconstructing the spatial detai => decrease channels, increase spatial size
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)     # spatial dimensions: D/4 -> D/2
        self.dec2 = ConvBlock(128, 64)                          # channels: 128 -> 64

        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)      # spatial dimensions: D/2 -> D
        self.dec1 = ConvBlock(64, 32)                           # channels: 64 -> 32

        # convert feature vectors into segmentation logits 
        # (binary segmentation => single dimension)
        # No sigmoid because it is later applied through the BCE and Dice losses
        self.out = nn.Conv3d(32, out_channels, 1)              # channels: 32 -> 1 (original)

    # Feature extractor
    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))

        return x1, x2, xb

    def decode(self, x1, x2, xb):
        x = self.up2(xb)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out(x)

    def forward(self, x):
        x1, x2, xb = self.encode(x)
        return self.decode(x1, x2, xb)


# =========================================================
# 1. LATE FUSION MODEL
# =========================================================

class LateFusionSegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.ap_model = SimpleSegNet(in_channels=1)
        self.pvp_model = SimpleSegNet(in_channels=1)

    def forward(self, ap_img, pvp_img):

        ap_mask = self.ap_model(ap_img) if ap_img is not None else None
        pvp_mask = self.pvp_model(pvp_img) if pvp_img is not None else None

        if ap_mask is None:
            return torch.sigmoid(pvp_mask)

        if pvp_mask is None:
            return torch.sigmoid(ap_mask)

        fused = torch.maximum(ap_mask, pvp_mask)

        # return torch.sigmoid(fused)
        return fused

    

# =========================================================
# 2. ATTENTION-BASED FUSION MODEL
# =========================================================


class AttentionFusionBlock(nn.Module):
    """
    Pass the concatenated AP and PVP features through a 2-layer CNN 
    that outputs the weights for each set of features. 
    Optimize the weights in the training process. 
    """
    def __init__(self, channels):
        super().__init__()

        self.att = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 2, 1)  # AP vs PVP weights
        )

    def forward(self, ap_feat, pvp_feat):

        # # CASE 1: only AP
        # if ap_exists and not pvp_exists:
        #     return ap_feat

        # # CASE 2: only PVP
        # if pvp_exists and not ap_exists:
        #     return pvp_feat

        # # CASE 3: both exist → safe to fuse
        combined = torch.cat([ap_feat, pvp_feat], dim=1)
        att = self.att(combined)
        att = torch.softmax(att, dim=1)
        w_ap = att[:, 0:1]
        w_pvp = att[:, 1:2]

        return w_ap * ap_feat + w_pvp * pvp_feat
    
class EncBlock(nn.Module):
    """
    Encoder block of the U-Net-based attention segmentation model
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down
    
class DecBlock(nn.Module):
    """
    Decoder block of the U-Net-based attention segmentation model.
    Skip connections integrated in the decoding
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, 2, stride=2)

        self.block = nn.Sequential(
            nn.Conv3d(out_c * 2, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class AttentionFusionSegmentation(nn.Module):

    def __init__(self, in_channels=1, base=32):
        super().__init__()

        # -------------------------
        # Learnable null embeddings for missing phases
        # -------------------------
        self.null_ap1  = nn.Parameter(torch.zeros(1, base, 1, 1, 1))
        self.null_ap2  = nn.Parameter(torch.zeros(1, base * 2, 1, 1, 1))
        self.null_ap3  = nn.Parameter(torch.zeros(1, base * 4, 1, 1, 1))

        self.null_pvp1 = nn.Parameter(torch.zeros(1, base, 1, 1, 1))
        self.null_pvp2 = nn.Parameter(torch.zeros(1, base * 2, 1, 1, 1))
        self.null_pvp3 = nn.Parameter(torch.zeros(1, base * 4, 1, 1, 1))

        # -------------------------
        # AP encoder
        # -------------------------
        self.ap1 = EncBlock(in_channels, base)
        self.ap2 = EncBlock(base, base * 2)
        self.ap3 = EncBlock(base * 2, base * 4)

        # -------------------------
        # PVP encoder
        # -------------------------
        self.pvp1 = EncBlock(in_channels, base)
        self.pvp2 = EncBlock(base, base * 2)
        self.pvp3 = EncBlock(base * 2, base * 4)

        # -------------------------
        # Attention fusion at each scale
        # -------------------------
        self.fuse1 = AttentionFusionBlock(base)
        self.fuse2 = AttentionFusionBlock(base * 2)
        self.fuse3 = AttentionFusionBlock(base * 4)

        # -------------------------
        # Decoder
        # -------------------------
        self.dec2 = DecBlock(base * 4, base * 2)
        self.dec1 = DecBlock(base * 2, base)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, ap, pvp):

        ap_exists = ap is not None
        pvp_exists = pvp is not None

        if not ap_exists and not pvp_exists:
            raise ValueError("At least one modality must be present")

        # -------------------------
        # Encode AP (if given)
        # -------------------------

        ap1 = ap2 = ap3 = None

        if ap_exists:
            ap1, apd1 = self.ap1(ap)
            ap2, apd2 = self.ap2(apd1)
            ap3, _    = self.ap3(apd2)


        # -------------------------
        # Encode PVP (if given)
        # -------------------------

        p1 = p2 = p3 = None

        if pvp_exists:
            p1, pd1 = self.pvp1(pvp)
            p2, pd2 = self.pvp2(pd1)
            p3, _   = self.pvp3(pd2)

        # -------------------------
        # Create missing AP features
        # -------------------------

        if not ap_exists:

            ap1 = self.null_ap1.expand_as(p1)
            ap2 = self.null_ap2.expand_as(p2)
            ap3 = self.null_ap3.expand_as(p3)

        # -------------------------
        # Create missing PVP features
        # -------------------------
        if not pvp_exists:

            p1 = self.null_pvp1.expand_as(ap1)
            p2 = self.null_pvp2.expand_as(ap2)
            p3 = self.null_pvp3.expand_as(ap3)


        # -------------------------
        # Cross-modal fusion with masking
        # -------------------------
        # Each encoder block output for each phase is passed through the 
        # attention fusion block that weights the contribution of each contrast image
        f1 = self.fuse1(ap1, p1)
        f2 = self.fuse2(ap2, p2)
        f3 = self.fuse3(ap3, p3)

        # -------------------------
        # Decode (U-Net)
        # -------------------------

        x = self.dec2(f3, f2)
        x = self.dec1(x, f1)

        return self.out(x)
    


# =========================================================
# 3. UNIMODAL MODEL WITH CROSS-PHASE SUPERVISION
# =========================================================



class BidirectionalCrossPhaseSegmentation(nn.Module):

    def __init__(self):

        super().__init__()

        self.ap_model = SimpleSegNet(in_channels=1)
        self.pvp_model = SimpleSegNet(in_channels=1)

    def forward(self, ap_img, pvp_img):

        ap_mask = self.ap_model(ap_img) if ap_img is not None else None
        pvp_mask = self.pvp_model(pvp_img) if pvp_img is not None else None

        # ----------------------------
        # Single-modality inference
        # ----------------------------

        if ap_mask is None:
            return torch.sigmoid(pvp_mask)

        if pvp_mask is None:
            return torch.sigmoid(ap_mask)

        # ----------------------------
        # Dual-modality case (training or full input)
        # ----------------------------
        # fused = torch.cat([ap_features, pvp_features], dim=1)
        return {
            "ap_mask": ap_mask,
            "pvp_mask": pvp_mask,
            "fused_mask": torch.maximum(ap_mask, pvp_mask)

        }