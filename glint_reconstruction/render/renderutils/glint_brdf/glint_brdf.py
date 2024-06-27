import torch
import imageio

from .glint_utils import *

"""
This class is modified from Glints2023.hlsl in https://drive.google.com/file/d/1YQDxlkZFEwV6ZeaXCUYMhB4P-3ODS32e/view.
Credit to Deliot et al. (https://thomasdeliot.wixsite.com/blog/single-post/hpg23-real-time-rendering-of-glinty-appearance-using-distributed-binomial-laws-on-anisotropic-grids).
"""


class GlintBRDF:
    def __init__(
        self,
        _LogMicrofacetDensity,
        _MicrofacetRoughness,
        _DensityRandomization,
        noise_3d,
        noise_4d,
        resolution,
    ):
        # constants
        self.DEG2RAD = 0.01745329251
        self.RAD2DEG = 57.2957795131
        self.EPSILON = 1e-7
        self.ZERO = torch.tensor(0.0, device="cuda")
        self.ONE = torch.tensor(1.0, device="cuda")
        self._ScreenSpaceScale = 2.75
        self.targetNDF = 50000.0
        self.maxNDF = 100000.0
        self._MicrofacetRoughness = (
            _MicrofacetRoughness  # should be fixed after the first 2 passes
        )
        self.noise_3d = noise_3d
        self.noise_4d = noise_4d
        self.resolution = int(resolution[0])  # Assuming a square image -> H = W
        self.batch_size = int(_MicrofacetRoughness.shape[0] / self.resolution**2)

        # below should be the parameters approximated from the counting method
        self._LogMicrofacetDensity = _LogMicrofacetDensity
        self._DensityRandomization = _DensityRandomization

    def CustomRand4Texture(self, slope, slopeRandOffset):
        slope2 = (
            torch.abs(slope)
            / torch.clamp(self._MicrofacetRoughness.unsqueeze(-1), min=self.EPSILON)
        )[None, :, None, :].repeat(
            4, 1, 3, 1
        )  # (4, n, 3, 2)
        slope2 = slope2 + (slopeRandOffset * self.resolution)
        slopeLerp = torch.frac(slope2)

        slopeCoord = (
            torch.floor(slope2).to(torch.int32) % self.resolution
        )  # (4, n, 3, 2)

        # For uniform, although the range is 0-1, the distribution is not uniform in the original code
        # For gaussian, the scale is obtained empirically from running the original code
        uniform = self.noise_4d[
            slopeCoord.view(-1, 2)[..., 0].to(torch.long), slopeCoord.view(-1, 2)[..., 1].to(torch.long)
        ].view(4, -1, 3, 4)
        gaussian = -4.5 + 9 * self.noise_4d[
            slopeCoord.view(-1, 2)[..., 1].to(torch.long), slopeCoord.view(-1, 2)[..., 0].to(torch.long)
        ].view(4, -1, 3, 4)

        return uniform, gaussian, slopeLerp.to("cuda")

    def GenerateAngularBinomialValueForSurfaceCell(
        self,
        randB,
        randG,
        slopeLerp,
        footprintOneHitProba,
        binomialSmoothWidth,
        footprintMean,
        footprintSTD,
        microfacetCount,
    ):
        gating = torch.where(
            randB < footprintOneHitProba.unsqueeze(-1), self.ONE, self.ZERO
        )  # so that this line is differentiable

        gating = torch.where(
            binomialSmoothWidth.unsqueeze(-1) > self.EPSILON,
            torch.clamp(
                RemapTo01(
                    randB,
                    (footprintOneHitProba + binomialSmoothWidth).unsqueeze(-1),
                    (footprintOneHitProba - binomialSmoothWidth).unsqueeze(-1),
                ),
                0.0,
                1.0,
            ),
            gating,
        )  # (4, n, 3, 4)

        # Compute gauss
        gauss = randG * footprintSTD.unsqueeze(-1) + footprintMean.unsqueeze(-1)
        gauss = torch.clamp(
            torch.floor(gauss), self.ZERO, microfacetCount.unsqueeze(-1)
        )

        results = gating * (1.0 + gauss)  # (4, n, 3, 4)
        res = BilinearLerp(results, slopeLerp)  # (4, n, 3)

        return res

    def SampleGlintGridSimplex(
        self, uv, gridSeed, slope, footprintAreas, targetNDF, gridWeights
    ):
        # Get surface space glint simplex grid cell
        gridToSkewedGrid = torch.tensor(
            [[1.0, -0.57735027], [0.0, 1.15470054]], device="cuda"
        )
        skewedCoord = torch.matmul(gridToSkewedGrid, uv.transpose(1, 2)).transpose(
            1, 2
        )  # (4, n, 2)

        temp = torch.cat(
            (
                torch.frac(skewedCoord),
                torch.zeros((4, skewedCoord.shape[1], 1)).to("cuda"),
            ),
            dim=-1,
        )  # (4, n, 3)
        temp[..., 2] = 1.0 - temp[..., 0] - temp[..., 1]

        s = torch.where(-temp[..., 2] >= 0.0, self.ONE, self.ZERO)
        s2 = 2.0 * s - 1.0

        baseId = toIntApprox(torch.floor(skewedCoord))[:, :, None, :].repeat(
            1, 1, 3, 1
        )  # (4, n, 3, 2)
        glints = baseId + torch.cat(
            (
                torch.cat((s.unsqueeze(-1), s.unsqueeze(-1)), dim=-1)[:, :, None, :],
                torch.cat((s.unsqueeze(-1), (1.0 - s).unsqueeze(-1)), dim=-1)[
                    :, :, None, :
                ],
                torch.cat(((1.0 - s).unsqueeze(-1), s.unsqueeze(-1)), dim=-1)[
                    :, :, None, :
                ],
            ),
            dim=-2,
        )  # (4, n, 3, 2)

        barycentrics = torch.stack(
            (-temp[..., 2] * s2, s - temp[..., 1] * s2, s - temp[..., 0] * s2), dim=-1
        )  # (4, n, 3)

        # Get per surface cell per slope cell random numbers
        rands = pcg3dFloat(
            torch.cat(
                (
                    toIntApprox(glints) + 2147483648,
                    gridSeed[:, :, None, :].repeat(1, 1, 3, 1),
                ),
                dim=-1,
            ).cpu()
        )  # (4, n, 3, 3)
        # (4, n, 3, 3)
        rands[..., 1:] = normalise_to(
            rands[..., 1:]
            + self.noise_3d[:, :, :, None].repeat(1, self.batch_size, 1, 3)[..., 1:],
            0.0,
            1.0,
        )  # (4, n, 3, 3)

        randSlopesB, randSlopesG, slopeLerps = self.CustomRand4Texture(
            slope, rands[..., 1:]
        )

        # Compute microfacet count with randomization
        logDensityRand = torch.clamp(
            sampleNormalDistribution(
                rands[..., 0],  # (4, n, 3)
                self._LogMicrofacetDensity,
                self._DensityRandomization,
            ),
            0.0,
            50.0,
        )  # (4, n, 3)

        microfacetCount = torch.clamp(
            footprintAreas * torch.exp(logDensityRand), min=self.EPSILON
        )  # (4, n, 3)

        # Compute binomial properties
        hitProba = torch.clamp(
            self._MicrofacetRoughness * targetNDF, 0.0, 1.0
        )  # probability of hitting desired half vector in NDF distribution0
        # torch.unsqueeze(gridWeights.T, -1) has shape (4, n, 1)
        microfacetCountBlended = torch.clamp(
            microfacetCount * torch.unsqueeze(gridWeights.T, -1), min=0.0
        )  # (4, n, 3)
        footprintOneHitProba = (
            1.0 - (1.0 - hitProba).unsqueeze(-1) ** microfacetCountBlended
        )  # probability of hitting at least one microfacet in footprint

        footprintMean = (microfacetCountBlended - 1.0) * hitProba.unsqueeze(
            -1
        )  # Expected value of number of hits in the footprint given already one hit
        footprintSTD = torch.sqrt(
            torch.clamp(
                footprintMean * (1.0 - hitProba.unsqueeze(-1)),
                min=self.EPSILON,  # this can't be 0.0 otherwise it will produce invalid values during backpropagation
            )
        )  # Standard deviation of number of hits in the footprint given already one hit

        binomialSmoothWidth = (
            0.1
            * torch.clamp(footprintOneHitProba * 10, 0.0, 1.0)
            * torch.clamp((1.0 - footprintOneHitProba) * 10, 0.0, 1.0)
        )  # (4, n, 3)

        # Generate numbers of reflecting microfacets
        results = self.GenerateAngularBinomialValueForSurfaceCell(
            randSlopesB,
            randSlopesG,
            slopeLerps,
            footprintOneHitProba,
            binomialSmoothWidth,
            footprintMean,
            footprintSTD,
            microfacetCountBlended,
        )  # (4, n, 3)

        # Interpolate result for glint grid cell
        results = results / microfacetCount  # (4, n, 3)

        res = torch.sum(results * barycentrics, dim=-1)

        return res
    
    def GetAnisoCorrectingGridTetrahedron(self, center_special_case, theta_bin_lerp, ratio_lerp, lod_lerp):
        n = center_special_case.shape[0]
        device = center_special_case.device
        
        # Initialize tensors to store p0, p1, p2, p3
        p0 = torch.empty((n, 3), device=device)
        p1 = torch.empty((n, 3), device=device)
        p2 = torch.empty((n, 3), device=device)
        p3 = torch.empty((n, 3), device=device)
        
        # Define all possible points
        a = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        b = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        c = torch.tensor([1, 1, 0], dtype=torch.float32, device=device)
        d = torch.tensor([0, 1, 1], dtype=torch.float32, device=device)
        e = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        f = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        
        c_normal = torch.tensor([0.5, 1, 0], dtype=torch.float32, device=device)
        d_normal = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
        e_normal = c
        f_normal = d
        g_normal = e
        h_normal = torch.tensor([0.5, 1, 1], dtype=torch.float32, device=device)
        i_normal = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
        j_normal = f

        # Compute conditions for center special case
        upper_pyramid = lod_lerp > (1.0 - ratio_lerp)
        lower_pyramid = lod_lerp < (1.0 - ratio_lerp)

        left_up_tetrahedron_center = RemapTo01(lod_lerp, 1.0 - ratio_lerp, 1.0) > theta_bin_lerp
        left_up_tetrahedron_a = RemapTo01(lod_lerp, 1.0 - ratio_lerp, 1.0) > RemapTo01(theta_bin_lerp * 2.0, 0.0, ratio_lerp)
        left_up_tetrahedron_b = RemapTo01(lod_lerp, 0.0, 1.0 - ratio_lerp) > RemapTo01(theta_bin_lerp, 0.5 - (1.0 - ratio_lerp) * 0.5, 0.5 + (1.0 - ratio_lerp) * 0.5)
        left_up_tetrahedron_c = RemapTo01(lod_lerp, 1.0 - ratio_lerp, 1.0) > RemapTo01((theta_bin_lerp - 0.5) * 2.0, 1.0 - ratio_lerp, 1.0)

        # Compute conditions for the normal case
        prism_a = (theta_bin_lerp < 0.5) & ((theta_bin_lerp * 2.0) < ratio_lerp)
        prism_b = (1.0 - ((theta_bin_lerp - 0.5) * 2.0)) > ratio_lerp
        prism_c = ~(prism_a | prism_b)

        # Center special case
        cond1 = center_special_case & upper_pyramid & left_up_tetrahedron_center
        cond2 = center_special_case & upper_pyramid & ~left_up_tetrahedron_center
        cond3 = center_special_case & ~upper_pyramid
        # Normal case
        # Prism A
        cond4 = ~center_special_case & prism_a & upper_pyramid & left_up_tetrahedron_a
        cond5 = ~center_special_case & prism_a & upper_pyramid & ~left_up_tetrahedron_a
        cond6 = ~center_special_case & prism_a & ~upper_pyramid
        # Prism B
        cond7 = ~center_special_case & prism_b & lower_pyramid & left_up_tetrahedron_b
        cond8 = ~center_special_case & prism_b & lower_pyramid & ~left_up_tetrahedron_b
        cond9 = ~center_special_case & prism_b & ~lower_pyramid
        # Prism C
        cond10 = ~center_special_case & prism_c & upper_pyramid & left_up_tetrahedron_c
        cond11 = ~center_special_case & prism_c & upper_pyramid & ~left_up_tetrahedron_c
        cond12 = ~center_special_case & prism_c & ~upper_pyramid

        p0 = torch.where(cond1.unsqueeze(1) | cond4.unsqueeze(1), a, p0)
        p0 = torch.where(cond2.unsqueeze(1), f, p0)
        p0 = torch.where(cond3.unsqueeze(1) | cond6.unsqueeze(1) | cond7.unsqueeze(1), b, p0)
        p0 = torch.where(cond5.unsqueeze(1) | cond9.unsqueeze(1) | cond10.unsqueeze(1), c_normal, p0)
        p0 = torch.where(cond8.unsqueeze(1) | cond12.unsqueeze(1), d_normal, p0)
        p0 = torch.where(cond11.unsqueeze(1), e_normal, p0)

        p1 = torch.where(cond1.unsqueeze(1) | cond2.unsqueeze(1), e, p1)
        p1 = torch.where(cond3.unsqueeze(1) | cond5.unsqueeze(1) | cond6.unsqueeze(1), a, p1)
        p1 = torch.where(cond4.unsqueeze(1), f_normal, p1)
        p1 = torch.where(cond7.unsqueeze(1) | cond9.unsqueeze(1), g_normal, p1)
        p1 = torch.where(cond8.unsqueeze(1), b, p1)
        p1 = torch.where(cond10.unsqueeze(1), j_normal, p1)
        p1 = torch.where(cond11.unsqueeze(1), i_normal, p1)
        p1 = torch.where(cond12.unsqueeze(1), e_normal, p1)

        p2 = torch.where(cond1.unsqueeze(1), d, p2)
        p2 = torch.where(cond2.unsqueeze(1) | cond3.unsqueeze(1), c, p2)
        p2 = torch.where(cond4.unsqueeze(1) | cond5.unsqueeze(1) | cond9.unsqueeze(1) | cond10.unsqueeze(1), h_normal, p2)
        p2 = torch.where(cond6.unsqueeze(1) | cond8.unsqueeze(1) | cond11.unsqueeze(1) | cond12.unsqueeze(1), c_normal, p2)
        p2 = torch.where(cond7.unsqueeze(1), i_normal, p2)

        p3 = torch.where(cond1.unsqueeze(1), f, p3)
        p3 = torch.where(cond2.unsqueeze(1), a, p3)
        p3 = torch.where(cond3.unsqueeze(1), e, p3)
        p3 = torch.where(cond4.unsqueeze(1) | cond5.unsqueeze(1) | cond6.unsqueeze(1), g_normal, p3)
        p3 = torch.where(cond7.unsqueeze(1), c_normal, p3)
        p3 = torch.where(cond8.unsqueeze(1) | cond9.unsqueeze(1) | cond10.unsqueeze(1) | cond12.unsqueeze(1), i_normal, p3)
        p3 = torch.where(cond11.unsqueeze(1), j_normal, p3)

        # Stack results to get shape (4, n, 3)
        return torch.stack([p0, p1, p2, p3], dim=0)

    def sample_glints(self, localHalfVector, uv, duvdx, duvdy):
        ellipseMajor, ellipseMinor = GetGradientEllipse(duvdx, duvdy)
        # ellipseRatio must be >= 1.0 otherwise it doesn't make sense
        ellipseRatio = torch.clamp(
            torch.norm(ellipseMajor, dim=-1) / torch.clamp(
            torch.norm(ellipseMinor, dim=-1), min=self.EPSILON
        ), min=1.0)

        # SHARED GLINT NDF VALUES
        halfScreenSpaceScaler = self._ScreenSpaceScale * 0.5
        slope = localHalfVector[..., :2]  # Orthogrtaphic slope projected grid
        print('slope: ', slope.shape)
        rescaledTargetNDF = self.targetNDF / max(self.maxNDF, self.EPSILON)

        # MANUAL LOD COMPENSATION
        lod = torch.log2(torch.norm(ellipseMinor, dim=-1) * halfScreenSpaceScaler)
        lod0 = toIntApprox(lod)
        lod1 = lod0 + 1.0
        divLod0 = torch.pow(2.0, lod0)
        divLod1 = torch.pow(2.0, lod1)
        lodLerp = torch.frac(lod)
        footprintAreaLOD0 = torch.pow(torch.exp2(lod0), 2.0)
        footprintAreaLOD1 = torch.pow(torch.exp2(lod1), 2.0)
        print('footprintAreaLOD: ', footprintAreaLOD0.shape) # n*n

        # MANUAL ANISOTROPY RATIO COMPENSATION
        ratio0 = torch.clamp(torch.pow(2.0, toIntApprox(torch.log2(ellipseRatio))), min=1.0)
        ratio1 = ratio0 * 2.0
        ratioLerp = torch.clamp(Remap(ellipseRatio, ratio0, ratio1, 0.0, 1.0), min=0.0, max=1.0)

        # MANUAL ANISOTROPY ROTATION COMPENSATION
        v2 = torch.nn.functional.normalize(ellipseMajor, dim=-1)
        theta = (
            torch.atan2(
                -v2[..., 0],
                v2[..., 1],
            )
            * self.RAD2DEG
        )
        thetaGrid = 90.0 / torch.clamp(ratio0, min=2.0)
        thetaBin = toIntApprox(theta / thetaGrid) * thetaGrid
        thetaBin += thetaGrid / 2.0
        thetaBin0 = torch.where(theta < thetaBin, thetaBin - thetaGrid / 2.0, thetaBin)
        thetaBinH = thetaBin0 + thetaGrid / 4.0
        thetaBin1 = thetaBin0 + thetaGrid / 2.0
        thetaBinLerp = Remap(theta, thetaBin0, thetaBin1, 0.0, 1.0)
        thetaBin0 = torch.where(thetaBin0 <= 0.0, 180.0 + thetaBin0, thetaBin0)

        # TETRAHEDRONIZATION OF ROTATION + RATIO + LOD GRID
        # centerSpecialCase = torch.where(ratio0 == 1.0, ratio0, self.ZERO)
        centerSpecialCase = ratio0 == 1.0
        divLods = torch.stack((divLod0, divLod1), dim=-1)
        footprintAreas = torch.stack((footprintAreaLOD0, footprintAreaLOD1), dim=-1)
        ratios = torch.stack((ratio0, ratio1), dim=-1)
        thetaBins = torch.stack(
            (thetaBin0, thetaBinH, thetaBin1, torch.zeros(thetaBin0.shape).to("cuda")),
            dim=-1,
        )  # added 0.0 for center singularity case
        tetras = self.GetAnisoCorrectingGridTetrahedron(
            centerSpecialCase, thetaBinLerp, ratioLerp, lodLerp
        )
        print('tetras: ', tetras.shape)
        # Account for center singularity in barycentric computation
        thetaBinLerp = torch.where(
            centerSpecialCase,
            Remap01To(thetaBinLerp, 0.0, ratioLerp),
            thetaBinLerp,
        )
        tetraBarycentricWeights = GetBarycentricWeightsTetrahedron(
            torch.stack((thetaBinLerp, ratioLerp, lodLerp), dim=-1), tetras
        )  # Compute barycentric coordinates within chosen tetrahedron
        print('tetraBarycentricWeights: ', tetraBarycentricWeights.shape)

        # PREPARE NEEDED ROTATIONS
        tetras[..., 0] *= 2
        tetras[..., 0] = torch.where(
            centerSpecialCase,
            torch.where(tetras[..., 1] == 0.0, 3.0, tetras[..., 0]),
            tetras[..., 0],
        )

        # selections based on tetra values
        thetaBins_tetras = torch.gather(
            thetaBins[None,].repeat(4, 1, 1),
            dim=-1,
            index=tetras[..., 0].unsqueeze(-1).to(torch.int64),
        )
        divLods_tetras = torch.gather(
            divLods[None,].repeat(4, 1, 1),
            dim=-1,
            index=tetras[..., 2].unsqueeze(-1).to(torch.int64),
        )
        ratios_tetras = torch.gather(
            ratios[None,].repeat(4, 1, 1),
            dim=-1,
            index=tetras[..., 1].unsqueeze(-1).to(torch.int64),
        )
        footprintAreas_tetras = torch.gather(
            footprintAreas[None,].repeat(4, 1, 1),
            dim=-1,
            index=tetras[..., 2].unsqueeze(-1).to(torch.int64),
        )

        uvRots = RotateUV(uv, thetaBins_tetras * self.DEG2RAD, torch.full((2,), 0.0))

        # note that modulo is not differentiable
        gridSeeds = (
            normalise_to(
                HashWithoutSine13(
                    torch.cat(
                        (divLods_tetras, thetaBins_tetras % 360, ratios_tetras), dim=-1
                    )
                )
                + HashWithoutSine13(self.noise_3d).repeat(1, self.batch_size, 1),
                0.0,
                1.0,
            )
            * 4294967296.0  # 2**32
        )  # (4, n, 1)

        # SAMPLE GLINT GRIDS
        samples = self.SampleGlintGridSimplex(
            uvRots
            / remove_zeros(divLods_tetras)
            / remove_zeros(
                torch.cat(
                    (torch.ones(ratios_tetras.shape).to("cuda"), ratios_tetras), dim=-1
                )
            ),
            gridSeeds,
            slope,
            ratios_tetras * footprintAreas_tetras,
            rescaledTargetNDF,
            tetraBarycentricWeights,
        )  # (4, n, 1) or equivalently (4, n). For simplicity the latter is used.

        res = (
            torch.sum(samples, dim=0)
            * (1.0 / torch.clamp(self._MicrofacetRoughness, self.EPSILON))
            * self.maxNDF
        )

        res = torch.clamp(res, min=0.0)

        assert is_valid(res)
        return res
