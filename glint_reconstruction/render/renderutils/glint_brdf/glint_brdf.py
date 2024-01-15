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
            slopeCoord.view(-1, 2)[..., 0], slopeCoord.view(-1, 2)[..., 1]
        ].view(4, -1, 3, 4)
        gaussian = -4.5 + 9 * self.noise_4d[
            slopeCoord.view(-1, 2)[..., 1], slopeCoord.view(-1, 2)[..., 0]
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

    def GetAnisoCorrectingGridTetrahedron(
        self, centerSpecialCase, thetaBinLerp, ratioLerp, lodLerp
    ):
        centerSpecialCase = centerSpecialCase.to(torch.bool)

        upper_pyramid_mask = lodLerp > 1.0 - ratioLerp
        lower_pyramid_mask = lodLerp < 1.0 - ratioLerp

        left_up_tetrahedron_mask = (
            RemapTo01(lodLerp, 1.0 - ratioLerp, 1.0) > thetaBinLerp
        )

        prismA_mask = (thetaBinLerp < 0.5) & (thetaBinLerp * 2.0 < ratioLerp)
        prismB_mask = 1.0 - ((thetaBinLerp - 0.5) * 2.0) > ratioLerp
        prismC_mask = (~prismA_mask) & (~prismB_mask)

        left_up_tetrahedron_mask_prismA = RemapTo01(
            lodLerp, 1.0 - ratioLerp, 1.0
        ) > RemapTo01(thetaBinLerp * 2.0, 0.0, ratioLerp)
        left_up_tetrahedron_mask_prismB = RemapTo01(
            lodLerp, 0.0, 1.0 - ratioLerp
        ) > RemapTo01(
            thetaBinLerp,
            0.5 - (1.0 - ratioLerp) * 0.5,
            0.5 + (1.0 - ratioLerp) * 0.5,
        )
        left_up_tetrahedron_mask_prismC = RemapTo01(
            lodLerp, 1.0 - ratioLerp, 1.0
        ) > RemapTo01((thetaBinLerp - 0.5) * 2.0, 1.0 - ratioLerp, 1.0)

        ps = torch.ones((len(centerSpecialCase), 4, 3)).to("cuda")

        # centerSpecialCase
        M = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # 0: a
                [0.0, 0.0, 0.0],  # 1: b
                [1.0, 1.0, 0.0],  # 2: c
                [0.0, 1.0, 1.0],  # 3: d
                [0.0, 0.0, 1.0],  # 4: e
                [1.0, 1.0, 1.0],  # 5: f
            ]
        ).to("cuda")
        special_cond_1 = (
            centerSpecialCase & upper_pyramid_mask & left_up_tetrahedron_mask
        )
        special_cond_2 = (
            centerSpecialCase & upper_pyramid_mask & ~left_up_tetrahedron_mask
        )
        special_cond_3 = centerSpecialCase & ~upper_pyramid_mask
        # p0
        ps[special_cond_1][:, 0, :] = M[0]
        ps[special_cond_2][:, 0, :] = M[5]
        ps[special_cond_3][:, 0, :] = M[1]
        # p1
        ps[centerSpecialCase & upper_pyramid_mask][:, 1, :] = M[4]
        ps[special_cond_3][:, 1, :] = M[0]
        # p2
        ps[special_cond_1][:, 2, :] = M[3]
        ps[special_cond_2][:, 2, :] = M[2]
        ps[special_cond_3][:, 2, :] = M[2]
        # p3
        ps[special_cond_1][:, 3, :] = M[5]
        ps[special_cond_2][:, 3, :] = M[0]
        ps[special_cond_3][:, 3, :] = M[4]

        # normal case
        M = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # 0: a
                [0.0, 0.0, 0.0],  # 1: b
                [0.5, 1.0, 0.0],  # 2: c
                [1.0, 0.0, 0.0],  # 3: d
                [1.0, 1.0, 0.0],  # 4: e
                [0.0, 1.0, 1.0],  # 5: f
                [0.0, 0.0, 1.0],  # 6: g
                [0.5, 1.0, 1.0],  # 7: h
                [1.0, 0.0, 1.0],  # 8: i
                [1.0, 1.0, 1.0],  # 9: j
            ]
        ).to("cuda")
        # cond general
        normal_upper = ~centerSpecialCase & upper_pyramid_mask
        normal_lower = ~centerSpecialCase & lower_pyramid_mask
        # cond prism A
        normal_cond_1_prismA = (
            normal_upper & prismA_mask & left_up_tetrahedron_mask_prismA
        )
        normal_cond_2_prismA = (
            normal_upper & prismA_mask & ~left_up_tetrahedron_mask_prismA
        )
        normal_cond_3_prismA = ~centerSpecialCase & prismA_mask & ~upper_pyramid_mask
        # cond prism B
        normal_cond_1_prismB = (
            normal_lower & prismB_mask & left_up_tetrahedron_mask_prismB
        )
        normal_cond_2_prismB = (
            normal_lower & prismB_mask & ~left_up_tetrahedron_mask_prismB
        )
        normal_cond_3_prismB = ~centerSpecialCase & prismB_mask & ~lower_pyramid_mask
        # cond prism C
        normal_cond_1_prismC = (
            normal_upper & prismC_mask & left_up_tetrahedron_mask_prismC
        )
        normal_cond_2_prismC = (
            normal_upper & prismC_mask & ~left_up_tetrahedron_mask_prismC
        )
        normal_cond_3_prismC = ~centerSpecialCase & prismC_mask & ~upper_pyramid_mask

        # p0
        # prism A
        ps[normal_cond_1_prismA][:, 0, :] = M[0]
        ps[normal_cond_2_prismA][:, 0, :] = M[2]
        ps[normal_cond_3_prismA][:, 0, :] = M[1]
        # prism B
        ps[normal_cond_1_prismB][:, 0, :] = M[1]
        ps[normal_cond_2_prismB][:, 0, :] = M[3]
        ps[normal_cond_3_prismB][:, 0, :] = M[2]
        # prism C
        ps[normal_cond_1_prismC][:, 0, :] = M[2]
        ps[normal_cond_2_prismC][:, 0, :] = M[4]
        ps[normal_cond_3_prismC][:, 0, :] = M[3]

        # p1
        # prism A
        ps[normal_cond_1_prismA][:, 1, :] = M[5]
        ps[normal_cond_2_prismA][:, 1, :] = M[0]
        ps[normal_cond_3_prismA][:, 1, :] = M[0]
        # prism B
        ps[normal_cond_1_prismB][:, 1, :] = M[6]
        ps[normal_cond_2_prismB][:, 1, :] = M[1]
        ps[normal_cond_3_prismB][:, 1, :] = M[6]
        # prism C
        ps[normal_cond_1_prismC][:, 1, :] = M[9]
        ps[normal_cond_2_prismC][:, 1, :] = M[8]
        ps[normal_cond_3_prismC][:, 1, :] = M[4]

        # p2
        # prism A
        ps[~centerSpecialCase & prismA_mask & upper_pyramid_mask][:, 2, :] = M[7]
        ps[normal_cond_3_prismA][:, 2, :] = M[2]
        # prism B
        ps[normal_cond_1_prismB][:, 2, :] = M[8]
        ps[normal_cond_2_prismB][:, 2, :] = M[2]
        ps[normal_cond_3_prismB][:, 2, :] = M[7]
        # prism C
        ps[normal_cond_1_prismC][:, 2, :] = M[7]
        ps[normal_cond_2_prismC][:, 2, :] = M[2]
        ps[normal_cond_3_prismC][:, 2, :] = M[2]

        # p3
        # prism A
        ps[~centerSpecialCase & prismA_mask][:, 3, :] = M[6]
        # prism B
        ps[normal_cond_1_prismB][:, 3, :] = M[2]
        ps[normal_cond_2_prismB][:, 3, :] = M[8]
        ps[normal_cond_3_prismB][:, 3, :] = M[8]
        # prism C
        ps[normal_cond_1_prismC][:, 3, :] = M[8]
        ps[normal_cond_2_prismC][:, 3, :] = M[9]
        ps[normal_cond_3_prismC][:, 3, :] = M[8]

        return ps.permute(1, 0, 2)

    def sample_glints(self, localHalfVector, uv, duvdx, duvdy):
        ellipseMajor, ellipseMinor = GetGradientEllipse(duvdx, duvdy)
        ellipseRatio = torch.norm(ellipseMajor, dim=-1) / torch.clamp(
            torch.norm(ellipseMinor, dim=-1), min=self.EPSILON
        )

        # SHARED GLINT NDF VALUES
        halfScreenSpaceScaler = self._ScreenSpaceScale * 0.5
        slope = localHalfVector[..., :2]  # Orthogrtaphic slope projected grid
        rescaledTargetNDF = self.targetNDF / max(self.maxNDF, self.EPSILON)

        # MANUAL LOD COMPENSATION
        lod = torch.log2(torch.norm(ellipseMinor, dim=-1) * halfScreenSpaceScaler)
        lod0 = toIntApprox(lod)
        lod1 = lod0 + 1
        divLod0 = 2.0**lod0
        divLod1 = 2.0**lod1
        lodLerp = torch.frac(lod)
        footprintAreaLOD0 = torch.exp2(lod0) ** 2.0
        footprintAreaLOD1 = torch.exp2(lod1) ** 2.0

        # MANUAL ANISOTROPY RATIO COMPENSATION
        ratio0 = torch.clamp(2.0 ** toIntApprox(torch.log2(ellipseRatio)), 1.0)
        ratio1 = ratio0 * 2.0
        ratioLerp = torch.clamp(Remap(ellipseRatio, ratio0, ratio1, 0.0, 1.0), 0.0, 1.0)

        # MANUAL ANISOTROPY ROTATION COMPENSATION
        v2 = torch.nn.functional.normalize(ellipseMajor, dim=-1)
        theta = (
            torch.atan2(
                -v2[..., 0],
                v2[..., 1],
            )
            * self.RAD2DEG
        )
        thetaGrid = 90.0 / torch.clamp(ratio0, 2.0)
        thetaBin = toIntApprox(theta / thetaGrid) * thetaGrid
        thetaBin += thetaGrid / 2.0
        thetaBin0 = torch.where(theta < thetaBin, thetaBin - thetaGrid / 2.0, thetaBin)
        thetaBinH = thetaBin0 + thetaGrid / 4.0
        thetaBin1 = thetaBinH * 2.0
        thetaBinLerp = Remap(theta, thetaBin0, thetaBin1, 0.0, 1.0)
        thetaBin0 = torch.where(thetaBin0 <= 0.0, 180.0 + thetaBin0, thetaBin0)

        # TETRAHEDRONIZATION OF ROTATION + RATIO + LOD GRID
        centerSpecialCase = torch.where(ratio0 == 1.0, ratio0, self.ZERO)
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
        # Account for center singularity in barycentric computation
        thetaBinLerp = torch.where(
            centerSpecialCase == 1.0,
            Remap01To(thetaBinLerp, 0.0, ratioLerp),
            thetaBinLerp,
        )
        tetraBarycentricWeights = GetBarycentricWeightsTetrahedron(
            torch.stack((thetaBinLerp, ratioLerp, lodLerp), dim=-1), tetras
        )  # Compute barycentric coordinates within chosen tetrahedron

        # PREPARE NEEDED ROTATIONS
        tetras[..., 0] *= 2

        # if (centerSpecialCase): # Account for center singularity (if center vertex => no rotation)
        three = torch.tensor(3.0).to("cuda")
        tetras[..., 0] = torch.where(
            centerSpecialCase == 1.0,
            torch.where(tetras[..., 1] == 0.0, three, tetras[..., 0]),
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
