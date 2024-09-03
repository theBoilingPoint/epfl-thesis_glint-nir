# Student Information
- Name: Xinran Tao
- PennKey: tpaulina

# Results
## Requested Screenshots
### 0% metallic, 0% rough, RBG = 1 1 1
![](./imgs/0m0r111.png)

### 100% metallic, 0% rough, RGB = 1 1 1
![](./imgs/1m0r111.png)

### 100% metallic, 25% rough, RGB = 1 1 1
![](./imgs/1m0.25r111.png)

### cerberus.json
![](./imgs/cerberus.png)

## Extra Credits Attempted
Three scenes are created inside the `custom_scenes.zip` file downloadable from this [link](https://drive.google.com/file/d/12q-GpdH_bxvoTGN7g_2EOeUX3xWLs0F3/view?usp=sharing). Please unzip it and put everything under the `models` folder. This step assumes that you have unzipped the file provided on Canvas and put everything in their rightful place. 

Below are the screenshots of the custom scenes.

# Foam
This scene shows an industrial black foam material.

![](./imgs/f1.png)

![](./imgs/f2.png)

![](./imgs/f3.png)

# Pretzel
This scene shows a pretzel-like material.

![](./imgs/p1.png)

![](./imgs/p2.png)

![](./imgs/p3.png)

# Cursed
This scene shows a cursed material.

![](./imgs/c1.png)

![](./imgs/c2.png)

![](./imgs/c3.png)

Physically-Based Shaders Part II: Environment Maps
======================

**University of Pennsylvania, CIS 561: Advanced Computer Graphics, Homework 7**

Overview
------------
You will take what you learned in part I of this physically-based shader assignment and
combine it with the pre-computation of irradiance applied to the plastic-metallic BRDF.
Recall that the overall formula for this BSDF is <img src="https://render.githubusercontent.com/render/math?math=\color{grey}f(\omega_o, \omega_i) = k_D f_{Lambert}( \omega_o, \omega_i)"> + <img src="https://render.githubusercontent.com/render/math?math=\color{grey}k_S f_{Cook-Torrance}(\omega_o, \omega_i)">

Here are some example screenshots of what your implementation should look like with varying amounts of roughness and metallicness:

![](defaultAttribs.png)

![](fullMetal0Rough.png) ![](fullMetal25Rough.png)

![](fullMetal50Rough.png) ![](fullMetal75Rough.png)

![](fullMetalFullRough.png)

The Light Transport Equation
--------------
#### L<sub>o</sub>(p, &#969;<sub>o</sub>) = L<sub>e</sub>(p, &#969;<sub>o</sub>) + &#8747;<sub><sub>S</sub></sub> f(p, &#969;<sub>o</sub>, &#969;<sub>i</sub>) L<sub>i</sub>(p, &#969;<sub>i</sub>) V(p', p) |dot(&#969;<sub>i</sub>, N)| _d_&#969;<sub>i</sub>

* __L<sub>o</sub>__ is the light that exits point _p_ along ray &#969;<sub>o</sub>.
* __L<sub>e</sub>__ is the light inherently emitted by the surface at point _p_
along ray &#969;<sub>o</sub>.
* __&#8747;<sub><sub>S</sub></sub>__ is the integral over the sphere of ray
directions from which light can reach point _p_. &#969;<sub>o</sub> and
&#969;<sub>i</sub> are within this domain.
* __f__ is the Bidirectional Scattering Distribution Function of the material at
point _p_, which evaluates the proportion of energy received from
&#969;<sub>i</sub> at point _p_ that is reflected along &#969;<sub>o</sub>.
* __L<sub>i</sub>__ is the light energy that reaches point _p_ from the ray
&#969;<sub>i</sub>. This is the recursive term of the LTE.
* __V__ is a simple visibility test that determines if the surface point _p_' from
which &#969;<sub>i</sub> originates is visible to _p_. It returns 1 if there is
no obstruction, and 0 is there is something between _p_ and _p_'. This is really
only included in the LTE when one generates &#969;<sub>i</sub> by randomly
choosing a point of origin in the scene rather than generating a ray and finding
its intersection with the scene.
* The __absolute-value dot product__ term accounts for Lambert's Law of Cosines.

Downloading texture and scene files
-----------------------------------
In order to save space in your repository, we have uploaded a `.zip` file to Canvas
containing the environment map `.hdr` files and `.json` scene files you can load into
your program. Download this `.zip` and extract its contents into the same folder your `.pro`
is in. We have also added a `.gitignore` that will ignore these folders when you push your
code to Github. Make sure, then, that you explicitly `git add` any custom scene files you
write as part of your extra credit.

Updating this README (5 points)
-------------
Make sure that you fill out this `README.md` file with your name and PennKey,
along with your example screenshots. For this assignment, you should take screenshots of your OpenGL window with the following configurations:
- 0% metallic, 0% rough, RBG = 1 1 1
- 100% metallic, 0% rough, RGB = 1 1 1
- 100% metallic, 25% rough, RGB = 1 1 1
- cerberus.json, with a camera angle that shows the model in profile

Where to write code
-----------------
All code written for this point-light based PBR shader will be implemented in three different `.glsl` files:
- `pbr.frag.glsl` (same name as last assignment, different code)
- `diffuseConvolution.frag.glsl`
- `glossyConvolution.frag.glsl`

A useful resource
-----------------
While we are implementing the paper [Real Shading in Unreal 4](http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf), you can find an excellent in-depth tutorial on its implementation on LearnOpenGL.org. For this assignment, you will want to refer to the following two articles:
- [Diffuse irradiance with image-based lighting](https://learnopengl.com/PBR/IBL/Diffuse-irradiance)
- [Specular image-based lighting](https://learnopengl.com/PBR/IBL/Specular-IBL)

Shader variable terminology (10 points)
----------------------------
While you are encouraged to refer to the articles on LearnOpenGL.org linked above, we want to make sure
that you are doing more than simply copying the code listed there. As such, you must assign specific names to variables that serve particular purposes, as the code linked above uses different names:
- Name: `wo` | Purpose: The ray traveling from the point being shaded to the camera
- Name: `wi` | Purpose: The ray traveling from the point being shaded to the source of irradiance
- Name: `wh` | Purpose: The microfacet surface normal one would use to reflect `wo` in the direction of `wi`.
- Name: `R` | Purpose: The innate material color used in the Fresnel reflectance function

Please make sure to use this terminology in all three `.glsl` files where appropriate.

Diffuse Irradiance precomputation (25 points)
--------------
Implement the `main` function of `diffuseConvolution.frag.glsl` so that it takes samples of
`u_EnvironmentMap` across the set of directions within the hemisphere aligned with the input surface
normal. Please carefully read the comments in this shader file in order to determine what the surface
normal is in the context of this shader.

When you have implemented the diffuse irradiance map, you can view it as your scene's background by
modifying the code in `MyGL::renderEnvironmentMap()`. Below are two examples of what you should see
(excluding the appearance of the sphere, as you have not yet implemented the PBR shader).

Default environment map:

![](diffuseIrradianceAtelier.png)

Fireplace environment map:

![](diffuseIrradianceFireplace.png)


Diffuse irradiance in the PBR shader (5 points)
----------
Now that you have your diffuse `Li` term, you can implement one portion of `pbr.frag.glsl`.
Using the surface normal direction in the shader, sample the diffuse irradiance cubemap
(`u_DiffuseIrradianceMap`) and combine it with your material's albedo. If you set this as your
output color, and set your material properties to the ones shown below, you should be able to
match these screenshots (also, don't forget to apply the Reinhard operator and gamma correction):

Default environment map:

![](diffuseMaterialAtelier.png)

Fireplace environment map:

![](diffuseMaterialFireplace.png)

Fireplace environment map (viewed from another angle):

![](diffuseMaterialFireplaceBack.png)

Glossy Irradiance precomputation (30 points)
----------
Implement the `main` function of `glossyConvolution.frag.glsl` so that it takes samples of
`u_EnvironmentMap` via importance sampling based on the GGX normal distribution function. Please carefully read the comments in this shader file in order to determine how you should orient the
glossy BRDF lobe you'll be importance sampling.

When you have implemented the glossy irradiance map, you can view it as your scene's background by
modifying the code in `MyGL::renderEnvironmentMap()`, and uncommenting the invocation of `textureLod` in `envMap.frag.glsl`. Below are two examples of what you should see
(excluding the appearance of the sphere, as you have not yet implemented the other portion of the PBR shader).

Default environment map:

![](glossyIrradianceAtelier.png)

Panorama environment map:

![](glossyIrradiancePanorama.png)

Glossy irradiance in the PBR shader (15 points)
----------------
Now that you have your glossy `Li` term, you can implement the other portion of `pbr.frag.glsl`.
You must compute the following attributes of your Cook-Torrance BRDF:
- Fresnel term using the Schlick approximation
- The combined D and G terms by sampling `u_BRDFLookupTexture` based on the `absdot` term and `roughness`.

Then, use `wi` to sample `u_GlossyIrradianceMap` using `textureLod`, along with `roughness` to determine
the mip level.

You must also compute `kD` based on your `kS` term, and multiply it with your diffuse component.

If you combine your diffuse and glossy colors into your output, and set your material properties to the ones shown below, you should be able to match the screenshots shown in the very beginning of these instructions.

Normal mapping and displacement mapping (10 points)
------------------
Modify `pbr.frag.glsl` and `pbr.vert.glsl` to do the following:
- Alter `N` to align with the direction given by `u_NormalMap` when `u_UseNormalMap` is true
- Alter `displacedPos` so that it is displaced along the surface normal by `u_DisplacementMagnitude` times a scalar obtained from `u_DisplacementMap` when `u_UseDisplacementMap` is true. 

Extra credit (15 points maximum)
-----------
Create your own custom JSON scenes and load them into your project. Create texture maps for as many material attributes as you can for more interesting surface appearance. The more interesting and numerous your scenes are, the more points you'll receive. You'll also have created excellent material for your
demo reel in the process!

Submitting your project
--------------
Along with your project code, make sure that you fill out this `README.md` file
with your name and PennKey, along with your test renders.

Rather than uploading a zip file to Canvas, you will simply submit a link to
the committed version of your code you wish us to grade. If you click on the
__Commits__ tab of your repository on Github, you will be brought to a list of
commits you've made. Simply click on the one you wish for us to grade, then copy
and paste the URL of the page into the Canvas submission form.
