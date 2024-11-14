import torch
from torch.utils.checkpoint import checkpoint

from .glint_utils import *

"""
This class is modified from Glints2023.hlsl in https://drive.google.com/file/d/1YQDxlkZFEwV6ZeaXCUYMhB4P-3ODS32e/view.
Credit to Deliot et al. (https://thomasdeliot.wixsite.com/blog/single-post/hpg23-real-time-rendering-of-glinty-appearance-using-distributed-binomial-laws-on-anisotropic-grids).
"""
class GlintBRDF:
    def __init__(
        self,
        glint_params,
        _MicrofacetRoughness,
        pcg3dFloat,
        noise_4d
    ):
        # constants
        self.EPSILON = 1e-7
        self.targetNDF = 0.5
        self.maxNDF = 1.0
        self._MicrofacetRoughness = _MicrofacetRoughness  
        self.pcg3dFloat = pcg3dFloat
        self.noise_4d = noise_4d
        self.resolution = noise_4d.shape[0]  # Assuming a square image -> H = W

        # below should be the parameters approximated from the counting method
        self._LogMicrofacetDensity = glint_params[0]
        self._DensityRandomization = glint_params[1]
        self._ScreenSpaceScale = glint_params[2]
    
    @torch.no_grad()
    def sampleNoiseMap4d(self, slope, batch_size=1024):
        slopeCoord = torch.remainder(torch.floor(slope).to(torch.long), self.resolution)
    
        # Extract the indices from the second tensor
        indices_x = slopeCoord[..., 0]  # Shape: [4, n*n]
        indices_y = slopeCoord[..., 1]  # Shape: [4, n*n]

        # Determine the total number of elements to process
        total_elements = indices_x.size(1)

        # Prepare a list to hold the results
        results = []

        # Process in smaller batches
        for start_idx in range(0, total_elements, batch_size):
            end_idx = min(start_idx + batch_size, total_elements)

            # Slice the indices for the current batch
            batch_indices_x = indices_x[:, start_idx:end_idx]
            batch_indices_y = indices_y[:, start_idx:end_idx]

            # Gather the values for the current batch
            batch_result = self.noise_4d[batch_indices_x, batch_indices_y]  # Shape: [4, batch_size, 4]

            # Append the result to the list
            results.append(batch_result)

        # Concatenate the results along the second dimension
        final_result = torch.cat(results, dim=1)  # Shape: [4, n*n, 4]

        return final_result

    def CustomRand4Texture(self, slope, slopeRandOffset):
        slope2 = torch.abs(slope) / torch.clamp(self._MicrofacetRoughness.unsqueeze(-1), min=self.EPSILON) + slopeRandOffset * self.resolution
        slopeLerp = torch.frac(slope2)
       
        outUniform = self.sampleNoiseMap4d(slope2).requires_grad_()
        # Note that there in the original code we are not suposed to use outUniform 
        # but rather RandXorshiftFloat(rngState01)
        outGaussian = sampleNormalDistribution(outUniform, 0.0, 1.0) 

        return outUniform, outGaussian, slopeLerp
        
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
            torch.where(randB < footprintOneHitProba.unsqueeze(-1), 0.0, 1.0),
        ) 

        # Compute gauss
        gauss = torch.clamp(
            torch.floor(randG * footprintSTD.unsqueeze(-1) + footprintMean.unsqueeze(-1)), 
            min=torch.tensor(0.0).to('cuda'), 
            max=microfacetCount.unsqueeze(-1)
        ) 
 
        return BilinearLerp(gating * (1.0 + gauss), slopeLerp)
    
    def computePcg3dFloat_in_batches(self, input_tensor, batch_size=16):
        output_list = []
        n = input_tensor.size(1)  # Get the size of the second dimension

        for i in range(0, n, batch_size):
            # Slice the input tensor to create a smaller batch
            batch = input_tensor[:, i:i + batch_size, :]
            # Process the batch using the model
            # Remember to change the input in SampleGlintGridSimplex!!!!
            # Use the original np version
            output_batch = pcg3dFloat_np(batch.cpu().numpy()).to('cuda')
            # TODO: Use MLP version
            # output_batch = checkpoint(self.pcg3dFloat, batch)
            # Store the output
            output_list.append(output_batch)

        # Concatenate all the outputs along the second dimension
        output_tensor = torch.cat(output_list, dim=1)
        
        return output_tensor

    def SampleGlintGridSimplex( 
        self, uv, gridSeed, slope, footprintAreas, targetNDF, gridWeights
    ):
        # Get surface space glint simplex grid cell
        gridToSkewedGrid = torch.tensor(
            [[1.0, -0.57735027], [0.0, 1.15470054]], device="cuda", dtype=torch.float32
        )

        skewedCoord = torch.einsum('ij,bkj->bki', gridToSkewedGrid, uv) # (4, n, 2)
        
        baseId = torch.floor(skewedCoord) # (4, n, 2)
        
        x = skewedCoord[..., 0]
        y = skewedCoord[..., 1]
        temp = torch.stack((x, y, 1.0 - x - y), dim=-1)

        s = torch.where(-temp[..., 2] < 0.0, 0.0, 1.0)
        s2 = 2.0 * s - 1.0
        
        glint0 = torch.abs(baseId + toIntApprox(torch.stack((s, s), dim=-1)) + 2147483648.0)
        glint1 = torch.abs(baseId + toIntApprox(torch.stack((s, 1.0 - s), dim=-1)) + 2147483648.0)
        glint2 = torch.abs(baseId + toIntApprox(torch.stack((1.0 - s, s), dim=-1)) + 2147483648.0)
        
        barycentrics = torch.stack(
            (-temp[..., 2] * s2, s - temp[..., 1] * s2, s - temp[..., 0] * s2), dim=-1
        )  # (4, n, 3)
        is_valid(barycentrics)
        # Get per surface cell per slope cell random numbers
        # TODO: Use this version if you are using MLP for pcg3dFloat
        # rand0 = self.computePcg3dFloat_in_batches((torch.cat((glint0, gridSeed.unsqueeze(-1)), dim=-1) / 4294967296.0))
        # rand1 = self.computePcg3dFloat_in_batches((torch.cat((glint1, gridSeed.unsqueeze(-1)), dim=-1) / 4294967296.0))
        # rand2 = self.computePcg3dFloat_in_batches((torch.cat((glint2, gridSeed.unsqueeze(-1)), dim=-1) / 4294967296.0))
        # Use this version if you are using the original implementation
        rand0 = self.computePcg3dFloat_in_batches((torch.cat((glint0, gridSeed.unsqueeze(-1)), dim=-1)))
        rand1 = self.computePcg3dFloat_in_batches((torch.cat((glint1, gridSeed.unsqueeze(-1)), dim=-1)))
        rand2 = self.computePcg3dFloat_in_batches((torch.cat((glint2, gridSeed.unsqueeze(-1)), dim=-1)))
 
        rand0SlopesB, rand0SlopesG, slopeLerp0 = self.CustomRand4Texture(slope, rand0[..., 1:])
        rand1SlopesB, rand1SlopesG, slopeLerp1 = self.CustomRand4Texture(slope, rand1[..., 1:])
        rand2SlopesB, rand2SlopesG, slopeLerp2 = self.CustomRand4Texture(slope, rand2[..., 1:])

        # Compute microfacet count with randomization
        randX = torch.stack((rand0[..., 0], rand1[..., 0], rand2[..., 0]), dim=2)
        logDensityRand = torch.clamp(
            sampleNormalDistribution(
                randX,  
                self._LogMicrofacetDensity.unsqueeze(-1),
                self._DensityRandomization.unsqueeze(-1),
            ),
            0.0,
            50.0,
        )  # (4, n, 3)
        microfacetCount = torch.clamp(
            footprintAreas.unsqueeze(-1) * torch.exp(logDensityRand), min=self.EPSILON
        )  # (4, n, 3)
        microfacetCountBlended = microfacetCount * torch.unsqueeze(gridWeights.T, -1) # (4, n, 3)

        # Compute binomial properties
        # probability of hitting desired half vector in NDF distribution
        hitProba = self._MicrofacetRoughness * targetNDF 
        # probability of hitting at least one microfacet in footprint
        footprintOneHitProba = torch.clamp(1.0 - torch.pow((1.0 - hitProba).unsqueeze(-1), microfacetCountBlended), min=0.0) # Although in the original code there is no clamp, but it should be like this if you look back
        # Expected value of number of hits in the footprint given already one hit
        footprintMean = (microfacetCountBlended - 1.0) * hitProba.unsqueeze(-1)  
        # Standard deviation of number of hits in the footprint given already one hit
        footprintSTD = torch.sqrt(
            torch.clamp(
                footprintMean * (1.0 - hitProba.unsqueeze(-1)),
                min=self.EPSILON,  # this can't be <= 0.0 otherwise it will produce invalid values during backpropagation
            )
        )  
        
        binomialSmoothWidth = (
            0.1
            * torch.clamp(footprintOneHitProba * 10.0, 0.0, 1.0)
            * torch.clamp((1.0 - footprintOneHitProba) * 10, 0.0, 1.0)
        )  # (4, n, 3)

        # Generate numbers of reflecting microfacets
        result0 = self.GenerateAngularBinomialValueForSurfaceCell(rand0SlopesB, rand0SlopesG, slopeLerp0, footprintOneHitProba[...,0], binomialSmoothWidth[...,0], footprintMean[...,0], footprintSTD[...,0], microfacetCountBlended[...,0])
        result1 = self.GenerateAngularBinomialValueForSurfaceCell(rand1SlopesB, rand1SlopesG, slopeLerp1, footprintOneHitProba[...,1], binomialSmoothWidth[...,1], footprintMean[...,1], footprintSTD[...,1], microfacetCountBlended[...,1])
        result2 = self.GenerateAngularBinomialValueForSurfaceCell(rand2SlopesB, rand2SlopesG, slopeLerp2, footprintOneHitProba[...,2], binomialSmoothWidth[...,2], footprintMean[...,2], footprintSTD[...,2], microfacetCountBlended[...,2])

        # Interpolate result for glint grid cell
        results = torch.stack((result0, result1, result2), dim=2) / microfacetCount # (4, n, 3)

        return torch.sum(results * barycentrics, dim=-1)
    
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
    
    @torch.no_grad()
    def GetAnisoCorrectingGridTetrahedron_NonDiff(self, center_special_case, theta_bin_lerp, ratio_lerp, lod_lerp):
        # Initialize output tensors
        n = theta_bin_lerp.shape[0]
        p0 = torch.zeros((n, 3))
        p1 = torch.zeros((n, 3))
        p2 = torch.zeros((n, 3))
        p3 = torch.zeros((n, 3))

        # Define the vertices
        vecs_special_case = {
            "a": torch.tensor([0, 1, 0]),
            "b": torch.tensor([0, 0, 0]),
            "c": torch.tensor([1, 1, 0]),
            "d": torch.tensor([0, 1, 1]),
            "e": torch.tensor([0, 0, 1]),
            "f": torch.tensor([1, 1, 1])
        }

        vecs_normal_case = {
            "a": torch.tensor([0, 1, 0]),
            "b": torch.tensor([0, 0, 0]),
            "c": torch.tensor([0.5, 1, 0]),
            "d": torch.tensor([1, 0, 0]),
            "e": torch.tensor([1, 1, 0]),
            "f": torch.tensor([0, 1, 1]),
            "g": torch.tensor([0, 0, 1]),
            "h": torch.tensor([0.5, 1, 1]),
            "i": torch.tensor([1, 0, 1]),
            "j": torch.tensor([1, 1, 1])
        }

        for i in range(n):
            if center_special_case[i]:  # Special Case (center of blending pattern)
                if lod_lerp[i] > 1.0 - ratio_lerp[i]:  # Upper pyramid
                    if RemapTo01(lod_lerp[i], 1.0 - ratio_lerp[i], 1.0) > theta_bin_lerp[i]:  # Left-up tetrahedron
                        p0[i], p1[i], p2[i], p3[i] = vecs_special_case["a"], vecs_special_case["e"], vecs_special_case["d"], vecs_special_case["f"]
                    else:  # Right-down tetrahedron
                        p0[i], p1[i], p2[i], p3[i] = vecs_special_case["f"], vecs_special_case["e"], vecs_special_case["c"], vecs_special_case["a"]
                else:  # Lower tetrahedron
                    p0[i], p1[i], p2[i], p3[i] = vecs_special_case["b"], vecs_special_case["a"], vecs_special_case["c"], vecs_special_case["e"]
            else:  # Normal case
                if theta_bin_lerp[i] < 0.5 and theta_bin_lerp[i] * 2.0 < ratio_lerp[i]:  # Prism A
                    if lod_lerp[i] > 1.0 - ratio_lerp[i]:  # Upper pyramid
                        if RemapTo01(lod_lerp[i], 1.0 - ratio_lerp[i], 1.0) > RemapTo01(theta_bin_lerp[i] * 2.0, 0.0, ratio_lerp[i]):  # Left-up tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["a"], vecs_normal_case["f"], vecs_normal_case["h"], vecs_normal_case["g"]
                        else:  # Right-down tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["c"], vecs_normal_case["a"], vecs_normal_case["h"], vecs_normal_case["g"]
                    else:  # Lower tetrahedron
                        p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["b"], vecs_normal_case["a"], vecs_normal_case["c"], vecs_normal_case["g"]
                elif 1.0 - ((theta_bin_lerp[i] - 0.5) * 2.0) > ratio_lerp[i]:  # Prism B
                    if lod_lerp[i] < 1.0 - ratio_lerp[i]:  # Lower pyramid
                        if RemapTo01(lod_lerp[i], 0.0, 1.0 - ratio_lerp[i]) > RemapTo01(theta_bin_lerp[i], 0.5 - (1.0 - ratio_lerp[i]) * 0.5, 0.5 + (1.0 - ratio_lerp[i]) * 0.5):  # Left-up tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["b"], vecs_normal_case["g"], vecs_normal_case["i"], vecs_normal_case["c"]
                        else:  # Right-down tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["d"], vecs_normal_case["b"], vecs_normal_case["c"], vecs_normal_case["i"]
                    else:  # Upper tetrahedron
                        p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["c"], vecs_normal_case["g"], vecs_normal_case["h"], vecs_normal_case["i"]
                else:  # Prism C
                    if lod_lerp[i] > 1.0 - ratio_lerp[i]:  # Upper pyramid
                        if RemapTo01(lod_lerp[i], 1.0 - ratio_lerp[i], 1.0) > RemapTo01((theta_bin_lerp[i] - 0.5) * 2.0, 1.0 - ratio_lerp[i], 1.0):  # Left-up tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["c"], vecs_normal_case["j"], vecs_normal_case["h"], vecs_normal_case["i"]
                        else:  # Right-down tetrahedron
                            p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["e"], vecs_normal_case["i"], vecs_normal_case["c"], vecs_normal_case["j"]
                    else:  # Lower tetrahedron
                        p0[i], p1[i], p2[i], p3[i] = vecs_normal_case["d"], vecs_normal_case["e"], vecs_normal_case["c"], vecs_normal_case["i"]

        return torch.stack([p0, p1, p2, p3], dim=0).to("cuda")
    
    def calculate_grid_seeds(self, divLods, thetaBins, ratios):
        tmp = torch.stack([divLods, thetaBins, ratios], dim=-1)

        # The results are supposed to be unit while HashWithoutSine13 returns a float
        return torch.clamp(torch.trunc(HashWithoutSine13(tmp) * 4294967296.0), min=0.0, max=4294967295.0)

    def sample_glints(self, localHalfVector, uv, duvdx, duvdy):
        ellipseMajor, ellipseMinor = GetGradientEllipse(duvdx, duvdy) # shape (n, 2)
        ellipseRatio = torch.norm(ellipseMajor, dim=-1) / (torch.norm(ellipseMinor, dim=-1) + self.EPSILON)

        # SHARED GLINT NDF VALUES
        halfScreenSpaceScaler = self._ScreenSpaceScale * 0.5 
        slope = localHalfVector[..., :2]  # Orthogrtaphic slope projected grid
        rescaledTargetNDF = self.targetNDF / (self.maxNDF + self.EPSILON)

        # MANUAL LOD COMPENSATION
        lod = torch.log2(torch.norm(ellipseMinor, dim=-1) * halfScreenSpaceScaler)
        lod0 = torch.floor(lod)
        lod1 = lod0 + 1.0
        divLod0 = torch.pow(2.0, lod0)
        divLod1 = torch.pow(2.0, lod1)
        lodLerp = torch.frac(lod)
        footprintAreaLOD0 = torch.pow(torch.exp2(lod0), 2.0)
        footprintAreaLOD1 = torch.pow(torch.exp2(lod1), 2.0)

        # MANUAL ANISOTROPY RATIO COMPENSATION
        ratio0 = torch.clamp(torch.pow(2.0, torch.floor(torch.log2(ellipseRatio))), min=1.0)
        ratio1 = ratio0 * 2.0
        ratioLerp = torch.clamp(Remap(ellipseRatio, ratio0, ratio1, 0.0, 1.0), min=0.0, max=1.0)

        # MANUAL ANISOTROPY ROTATION COMPENSATION
        v2 = torch.nn.functional.normalize(ellipseMajor, dim=-1)
        theta = torch.rad2deg(
            torch.atan2(-v2[..., 0], v2[..., 1])
        )
        thetaGrid = 90.0 / torch.clamp(ratio0, min=2.0)
        thetaBin = torch.trunc(theta / thetaGrid) * thetaGrid
        thetaBin += thetaGrid / 2.0
        thetaBin0 = torch.where(theta < thetaBin, thetaBin - thetaGrid / 2.0, thetaBin)
        thetaBinH = thetaBin0 + thetaGrid / 4.0
        thetaBin1 = thetaBin0 + thetaGrid / 2.0
        thetaBinLerp = Remap(theta, thetaBin0, thetaBin1, 0.0, 1.0)
        thetaBin0 = torch.where(thetaBin0 <= 0.0, 180.0 + thetaBin0, thetaBin0)

        # TETRAHEDRONIZATION OF ROTATION + RATIO + LOD GRID
        centerSpecialCase = ratio0 == 1.0 # Note that gradients will not flow through ratio0
        divLods = torch.stack((divLod0, divLod1), dim=-1) # shape (n, 2)
        footprintAreas = torch.stack((footprintAreaLOD0, footprintAreaLOD1), dim=-1)
        ratios = torch.stack((ratio0, ratio1), dim=-1)
        thetaBins = torch.stack(
            (thetaBin0, thetaBinH, thetaBin1, torch.zeros(thetaBin0.shape).to("cuda")),
            dim=-1,
        ) # shape (n, 4)
        # TODO: Differentiable version
        # tetras = self.GetAnisoCorrectingGridTetrahedron(
        #     centerSpecialCase, thetaBinLerp, ratioLerp, lodLerp
        # ) # shape (4, n, 3)
        # Non-differentiable version
        tetras = self.GetAnisoCorrectingGridTetrahedron_NonDiff(centerSpecialCase, thetaBinLerp, ratioLerp, lodLerp)
     
        # Account for center singularity in barycentric computation
        thetaBinLerp = torch.where(
            centerSpecialCase,
            Remap01To(thetaBinLerp, 0.0, ratioLerp),
            thetaBinLerp,
        )
        tetraBarycentricWeights = GetBarycentricWeightsTetrahedron(
            torch.stack((thetaBinLerp, ratioLerp, lodLerp), dim=-1), tetras
        )  # Compute barycentric coordinates within chosen tetrahedron

        # PREPARE NEEDED ROTATIONS
        tetras[..., 0] *= 2.0
        tetras[..., 0] = torch.where(
            centerSpecialCase,
            torch.where(tetras[..., 1] == 0.0, 3.0, tetras[..., 0]),
            tetras[..., 0],
        ) # shape (4, n, 3)
        
        # Non-differentiable version
        with torch.no_grad():
            n = thetaBins.shape[0]  
            
            x_indices = tetras[..., 0].long() 
            z_indices = tetras[..., 2].long()
            y_indices = tetras[..., 1].long()
            
            thetaBins_selected = torch.stack([thetaBins[torch.arange(n), x_indices[i]] for i in range(4)], dim=0)
            divLods_selected = torch.stack([divLods[torch.arange(n), z_indices[i]] for i in range(4)], dim=0)
            ratios_selected = torch.stack([ratios[torch.arange(n), y_indices[i]] for i in range(4)], dim=0)
            footprintAreas_selected = torch.stack([footprintAreas[torch.arange(n), z_indices[i]] for i in range(4)], dim=0)
        
        # Get the selected thetaBins
        # TODO: Differentiable version ###############################################
        # thetaBins_edges = torch.tensor([0, 1, 2, 3], device='cuda', dtype=torch.float32).view(1, 1, 4)
        # thetaBins_idx = torch.nn.functional.gumbel_softmax(
        #         thetaBins_edges - tetras[...,0].unsqueeze(2),
        #         tau=0.5,
        #         hard=True
        #     ) # (4, n, 4)
        
        # thetaBins_selected = torch.zeros(4, thetaBins.shape[0], device=thetaBins.device)
        # for i in range(4):
        #     # Use for loop to optimise memory usage
        #     # Multiply tensor1 with the corresponding slice of tensor2 and sum over the last dimension
        #     thetaBins_selected[i] = torch.sum(thetaBins.unsqueeze(0) * thetaBins_idx[i], dim=-1)
        
        # # Get the selected divLods
        # binary_edges = torch.tensor([0, 1], device='cuda', dtype=torch.float32).view(1, 1, 2)
        # divLods_idx = torch.nn.functional.gumbel_softmax(
        #         binary_edges - tetras[...,2].unsqueeze(2),
        #         tau=0.5,
        #         hard=True
        #     ) # (4, n, 2)
        # divLods_selected = torch.einsum('ij,kij->ki', divLods, divLods_idx)
        
        # # Get the selected ratios
        # ratios_dix = torch.nn.functional.gumbel_softmax(
        #         binary_edges - tetras[...,1].unsqueeze(2),
        #         tau=0.5,
        #         hard=True
        #     ) # (4, n, 2)
        # ratios_selected = torch.einsum('ij,kij->ki', ratios, ratios_dix) 
        
        # # Get the selected footprintAreas
        # footprintAreas_idx = torch.nn.functional.gumbel_softmax(
        #     binary_edges - tetras[...,2].unsqueeze(2),
        #     tau=0.5,
        #     hard=True
        # )
        # footprintAreas_selected = torch.einsum('ij,kij->ki', footprintAreas, footprintAreas_idx)
        ########################################################################################
        
        # selections based on tetra values
        uvRots = RotateUV(uv, torch.deg2rad(thetaBins_selected))
        ratios_selected_reshaped = ratios_selected.unsqueeze(-1)
        denom = torch.cat((torch.ones(ratios_selected_reshaped.shape, device='cuda'), ratios_selected_reshaped), dim=-1)
        final_uv = uvRots / divLods_selected.unsqueeze(-1) / denom # Be careful of the division by zero
        assert is_valid(final_uv)
        
        gridSeeds = self.calculate_grid_seeds(torch.log2(divLods_selected), torch.remainder(thetaBins_selected, 360), ratios_selected)

        # SAMPLE GLINT GRIDS
        samples = self.SampleGlintGridSimplex(
            final_uv,
            gridSeeds,
            slope,
            ratios_selected * footprintAreas_selected,
            rescaledTargetNDF,
            tetraBarycentricWeights
        )

        res = torch.sum(samples, dim=0) * (1.0 / torch.clamp(self._MicrofacetRoughness, self.EPSILON)) * self.maxNDF

        assert is_valid(res)
        return res
