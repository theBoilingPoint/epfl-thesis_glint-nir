#include "mygl.h"
#include <glm_includes.h>

#include <iostream>
#include <QApplication>
#include <QKeyEvent>
#include <QDir>

#include <QFileDialog>
#include <stdexcept>
#include "texture.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>

MyGL::MyGL(QWidget *parent)
    :   OpenGLContext(parent),
        modelMatrix_default(glm::mat4(1.f)),
        modelMatrix(glm::mat4(1.f)),
        rotationX(0.f), 
        rotationY(0.f), 
        rotationZ(0.f),
        m_geomSquare(this),
        m_geomMesh(this), m_textureAlbedo(this), m_textureMetallic(this),
        m_textureNormals(this), m_textureRoughness(this), m_textureAO(this),
        m_textureDisplacement(this),
        m_geomCube(this),
        m_hdrEnvMap(this),
        m_environmentCubemapFB(this, 1024, 1024, 1.f),
        m_diffuseIrradianceFB(this, 32, 32, 1.f),
        m_glossyIrradianceFB(this, 512, 512, 1.f),
        m_brdfLookupTexture(this),
        // TODO: for now the width and height are hardcoded. But it should be consistent with what's used in glintNoise.frag.glsl
        m_glintNoiseFB(this, 1024, 1024, 1.f),
        m_progPBR(this), m_progCubemapConversion(this),
        m_progCubemapDiffuseConvolution(this),
        m_progCubemapGlossyConvolution(this),
        m_progEnvMap(this),
        m_progGlintNoise(this),
        m_glCamera(), m_mousePosPrev(), m_albedo(0.5f, 0.5f, 0.5f),
        m_cubemapsNotGenerated(true),
        backgroundColour(0)
{
    setFocusPolicy(Qt::StrongFocus);
}

MyGL::~MyGL()
{
    makeCurrent();
    glDeleteVertexArrays(1, &vao);
    m_geomSquare.destroy();
}

void MyGL::initializeGL()
{
    // Create an OpenGL context using Qt's QOpenGLFunctions_3_2_Core class
    // If you were programming in a non-Qt context you might use GLEW (GL Extension Wrangler)instead
    initializeOpenGLFunctions();
    // Print out some information about the current OpenGL context
    debugContextVersion();

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    // Set the color with which the screen is filled at the start of each render call.
    glClearColor(0.5, 0.5, 0.5, 1);

    printGLErrorLog();

    // Create a Vertex Attribute Object
    glGenVertexArrays(1, &vao);

    m_geomSquare.create();
    QString path = getCurrentPath();
    path.append("/assignment_package/objs/");
    m_geomMesh.LoadOBJ("sphere.obj", path);
    m_geomMesh.create();
    m_geomCube.create();

    m_progPBR.create(":/glsl/pbr.vert.glsl", ":/glsl/pbr.frag.glsl");
    m_progCubemapConversion.create(":/glsl/cubemap.vert.glsl", ":/glsl/cubemap_uv_conversion.frag.glsl");
    m_progCubemapDiffuseConvolution.create(":/glsl/cubemap.vert.glsl", ":/glsl/diffuseConvolution.frag.glsl");
    m_progCubemapGlossyConvolution.create(":/glsl/cubemap.vert.glsl", ":/glsl/glossyConvolution.frag.glsl");
    m_progEnvMap.create(":/glsl/envMap.vert.glsl", ":/glsl/envMap.frag.glsl");
    m_progGlintNoise.create(":/glsl/glintNoise.vert.glsl", ":/glsl/glintNoise.frag.glsl");
    setupShaderHandles();

    path = getCurrentPath();
    path.append("/assignment_package/environment_maps/clarens.hdr");
    m_hdrEnvMap.create(path.toStdString().c_str(), false);

    m_environmentCubemapFB.create(true);
    m_diffuseIrradianceFB.create();
    m_glossyIrradianceFB.create(true);
    m_brdfLookupTexture.create(":/textures/brdfLUT.png", false);
    m_glintNoiseFB.create();

    // We have to have a VAO bound in OpenGL 3.2 Core. But if we're not
    // using multiple VAOs, we can just bind one once.
    glBindVertexArray(vao);

    initPBRunifs();
}

void MyGL::resizeGL(int w, int h)
{
    //This code sets the concatenated view and perspective projection matrices used for
    //our scene's camera view.
    m_glCamera = Camera(w, h);
    glm::mat4 viewproj = m_glCamera.getViewProj();

    // Upload the view-projection matrix to our shaders (i.e. onto the graphics card)

    m_progPBR.setUnifMat4("u_ViewProj", viewproj);

    printGLErrorLog();
}

//This function is called by Qt any time your GL window is supposed to update
//For example, when the function update() is called, paintGL is called implicitly.
void MyGL::paintGL() {

    // If this is the very first draw cycle, or
    // if we've loaded a new env map, do the following:
    // 1. Draw the 2D environment map to the cube map frame buffer
    // 2. Generate mip levels for the cubemapped environment map
    //    so they can be sampled in the glossy irradiance computation
    // 3. Compute the diffuse irradiance of the newly loaded env map
    // 4. Compute the glossy irradiance of the newly loaded env map
    if(m_cubemapsNotGenerated) {
        // Convert the 2D HDR environment map texture to a cube map
        renderCubeMapToTexture();
        printGLErrorLog();
        // Generate mipmap levels for the environment map so that the
        // glossy reflection convolution has reduced fireflies
        m_environmentCubemapFB.generateMipMaps();
        printGLErrorLog();
        // Generate a cubemap of the diffuse irradiance (light reflected by the
        // Lambertian BRDF)
        renderConvolvedDiffuseCubeMapToTexture();
        printGLErrorLog();
        // Generate a cubemap of the varying levels of glossy irradiance light
        // reflected by the Cook-Torrance
        renderConvolvedGlossyCubeMapToTexture();
        printGLErrorLog();
        // Generate the noise map for glint
        renderGlintNoiseToTexture(m_glintNoiseFB.width(), 66);
        printGLErrorLog();

        // So that we are sure all previous lines are executed
        m_cubemapsNotGenerated = false;
    }

    glViewport(0,0,this->width() * this->devicePixelRatio(),this->height() * this->devicePixelRatio());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderEnvironmentMap();

    m_progPBR.setUnifMat4("u_ViewProj", m_glCamera.getViewProj());
    m_progPBR.setUnifVec3("u_CamPos", m_glCamera.eye);

    //Send the geometry's transformation matrix to the shader
    m_progPBR.setUnifMat4("u_Model", modelMatrix);
    m_progPBR.setUnifMat4("u_ModelInvTr", glm::transpose(glm::inverse(modelMatrix)));
    // Set up the diffuse irradiance map on the GPU so our surface shader can read it
    m_diffuseIrradianceFB.bindToTextureSlot(DIFFUSE_IRRADIANCE_CUBE_TEX_SLOT);
    m_progPBR.setUnifInt("u_DiffuseIrradianceMap", DIFFUSE_IRRADIANCE_CUBE_TEX_SLOT);
    // Set up the glossy irradiance map on the GPU so our surface shader can read it
    m_glossyIrradianceFB.bindToTextureSlot(GLOSSY_IRRADIANCE_CUBE_TEX_SLOT);
    m_progPBR.setUnifInt("u_GlossyIrradianceMap", GLOSSY_IRRADIANCE_CUBE_TEX_SLOT);
    // Also load our BRDF lookup texture for split-sum approximation
    m_brdfLookupTexture.bind(BRDF_LUT_TEX_SLOT);
    printGLErrorLog();
    m_progPBR.setUnifInt("u_BRDFLookupTexture", BRDF_LUT_TEX_SLOT);
    // Load the glint noise texture
    m_glintNoiseFB.bindToTextureSlot(GLINT_NOISE_TEX_SLOT);
    m_progPBR.setUnifInt("u_GlintNoiseTexture", GLINT_NOISE_TEX_SLOT);
    m_progPBR.setUnifUint("_Glint2023NoiseMapSize", m_glintNoiseFB.width());

    if(m_textureAlbedo.m_isCreated) {
        m_textureAlbedo.bind(ALBEDO_TEX_SLOT);
        m_progPBR.setUnifInt("u_AlbedoMap", ALBEDO_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseAlbedoMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseAlbedoMap", 0);
    }
    if(m_textureMetallic.m_isCreated) {
        m_textureMetallic.bind(METALLIC_TEX_SLOT);
        m_progPBR.setUnifInt("u_MetallicMap", METALLIC_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseMetallicMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseMetallicMap", 0);
    }
    if(m_textureRoughness.m_isCreated) {
        m_textureRoughness.bind(ROUGHNESS_TEX_SLOT);
        m_progPBR.setUnifInt("u_RoughnessMap", ROUGHNESS_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseRoughnessMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseRoughnessMap", 0);
    }
    if(m_textureAO.m_isCreated) {
        m_textureAO.bind(AO_TEX_SLOT);
        m_progPBR.setUnifInt("u_AOMap", AO_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseAOMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseAOMap", 0);
    }
    if(m_textureNormals.m_isCreated) {
        m_textureNormals.bind(NORMALS_TEX_SLOT);
        m_progPBR.setUnifInt("u_NormalMap", NORMALS_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseNormalMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseNormalMap", 0);
    }
    if(m_textureDisplacement.m_isCreated) {
        m_textureDisplacement.bind(DISPLACEMENT_TEX_SLOT);
        m_progPBR.setUnifInt("u_DisplacementMap", DISPLACEMENT_TEX_SLOT);
        m_progPBR.setUnifInt("u_UseDisplacementMap", 1);
    }
    else {
        m_progPBR.setUnifInt("u_UseDisplacementMap", 0);
    }
    //Draw the example sphere using our lambert shader
    m_progPBR.draw(m_geomMesh);
}

void MyGL::renderCubeMapToTexture() {
    m_progCubemapConversion.useMe();
    // Set the cube map conversion shader's sampler to tex slot 0
    m_progCubemapConversion.setUnifInt("u_EquirectangularMap", ENV_MAP_FLAT_TEX_SLOT);
    // put the HDR environment map into texture slot 0
    m_hdrEnvMap.bind(ENV_MAP_FLAT_TEX_SLOT);
    // Set viewport dimensions equal to those of our cubemap faces
    glViewport(0, 0, m_environmentCubemapFB.width(), m_environmentCubemapFB.height());
    m_environmentCubemapFB.bindFrameBuffer();

    // Iterate over each face of the cube and apply the appropriate rotation
    // view matrix to the cube, then draw it w/ the HDR texture applied to it
    for(int i = 0; i < 6; ++i) {
        m_progCubemapConversion.setUnifMat4("u_ViewProj", views[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               m_environmentCubemapFB.getCubemapHandle(), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_progCubemapConversion.draw(m_geomCube);
    }
    // Make sure we reset our OpenGL output to the default frame buffer
    // so additional draw calls are visible on screen.
    glBindFramebuffer(GL_FRAMEBUFFER, this->defaultFramebufferObject());

}

void MyGL::renderConvolvedDiffuseCubeMapToTexture() {
    m_progCubemapDiffuseConvolution.useMe();
    // Set the cube map conversion shader's sampler to tex slot 0
    m_progCubemapDiffuseConvolution.setUnifInt("u_EnvironmentMap", ENV_MAP_CUBE_TEX_SLOT);
    // put the HDR environment map into texture slot 0
    m_environmentCubemapFB.bindToTextureSlot(ENV_MAP_CUBE_TEX_SLOT);
    // Set viewport dimensions equal to those of our cubemap faces
    glViewport(0, 0, m_diffuseIrradianceFB.width(), m_diffuseIrradianceFB.height());
    m_diffuseIrradianceFB.bindFrameBuffer();

    // Iterate over each face of the cube and apply the appropriate rotation
    // view matrix to the cube, then draw it w/ the HDR texture applied to it
    for(int i = 0; i < 6; ++i) {
        m_progCubemapDiffuseConvolution.setUnifMat4("u_ViewProj", views[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               m_diffuseIrradianceFB.getCubemapHandle(), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_progCubemapDiffuseConvolution.draw(m_geomCube);
    }
    // Make sure we reset our OpenGL output to the default frame buffer
    // so additional draw calls are visible on screen.
    glBindFramebuffer(GL_FRAMEBUFFER, this->defaultFramebufferObject());

}

void MyGL::renderConvolvedGlossyCubeMapToTexture() {
    m_progCubemapGlossyConvolution.useMe();
    // Set the cube map conversion shader's sampler to tex slot 0
    m_progCubemapGlossyConvolution.setUnifInt("u_EnvironmentMap", ENV_MAP_CUBE_TEX_SLOT);
    // put the HDR environment map into texture slot 0
    m_environmentCubemapFB.bindToTextureSlot(ENV_MAP_CUBE_TEX_SLOT);
    // Set viewport dimensions equal to those of our cubemap faces
    glViewport(0, 0, m_glossyIrradianceFB.width(), m_glossyIrradianceFB.height());
    m_glossyIrradianceFB.bindFrameBuffer();

    const unsigned int maxMipLevels = 5;
    for(unsigned int mipLevel = 0; mipLevel < maxMipLevels; ++mipLevel) {
        // Resize our frame buffer according to our mip level
        unsigned int mipWidth  = static_cast<unsigned int>(m_glossyIrradianceFB.width() * std::pow(0.5, mipLevel));
        unsigned int mipHeight = static_cast<unsigned int>(m_glossyIrradianceFB.height() * std::pow(0.5, mipLevel));
        m_glossyIrradianceFB.bindRenderBuffer(mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = static_cast<float>(mipLevel) / static_cast<float>(maxMipLevels - 1);
        m_progCubemapGlossyConvolution.setUnifFloat("u_Roughness", roughness);

        // Iterate over each face of the cube and apply the appropriate rotation
        // view matrix to the cube, then draw it w/ the HDR texture applied to it
        for(int i = 0; i < 6; ++i) {
            m_progCubemapGlossyConvolution.setUnifMat4("u_ViewProj", views[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   m_glossyIrradianceFB.getCubemapHandle(), mipLevel);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            m_progCubemapGlossyConvolution.draw(m_geomCube);
        }
    }
    // Make sure we reset our OpenGL output to the default frame buffer
    // so additional draw calls are visible on screen.
    glBindFramebuffer(GL_FRAMEBUFFER, this->defaultFramebufferObject());
}

void MyGL::renderGlintNoiseToTexture(unsigned int size, unsigned int seed) {
    m_progGlintNoise.useMe(); 

    // vertex shader
    m_progGlintNoise.setUnifMat4("u_Model", modelMatrix);
    m_progGlintNoise.setUnifMat4("u_ViewProj", m_glCamera.getViewProj());
    // fragment shader
    m_progGlintNoise.setUnifUint("_FrameSize", size);
    m_progGlintNoise.setUnifUint("_Seed", seed);

    // A framebuffer in OpenGL is a collection of memory buffers that can be used as a destination for rendering. 
    // It acts as an intermediate storage for the pixel data generated by your rendering operations. 
    // Typically, when you render something in OpenGL, the output is drawn directly to the screen, which is the default framebuffer managed by the system. 
    // However, you can also render to off-screen framebuffers, which are useful for a variety of advanced rendering techniques.
    m_glintNoiseFB.bindFrameBuffer();

    // Set viewport dimensions equal to those of our cubemap faces
    glViewport(0, 0, m_glintNoiseFB.width(), m_glintNoiseFB.height());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               m_glintNoiseFB.getTextureHandle(), 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_progGlintNoise.draw(m_geomSquare);

    // Make sure we reset our OpenGL output to the default frame buffer
    // so additional draw calls are visible on screen.
    glBindFramebuffer(GL_FRAMEBUFFER, this->defaultFramebufferObject());
}

// Change which block of code is enabled by #if statements
// in order to display one of your convoluted irradiance maps
// instead of the photographic environment map.
void MyGL::renderEnvironmentMap() {
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Standard alpha blending

    // Clear the background with a fully transparent color (RGBA: 0, 0, 0, 0)
    glClearColor(0.f, 0.f, 0.f, 0.f); // Set clear color to transparent
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the color and depth buffers

    // Render photographic environment map as background
    m_environmentCubemapFB.bindToTextureSlot(ENV_MAP_CUBE_TEX_SLOT);
    m_progEnvMap.setUnifInt("u_EnvironmentMap", ENV_MAP_CUBE_TEX_SLOT);

    // Set the environment map shader's cubemap sampler to the same tex slot
    m_progEnvMap.setUnifMat4("u_ViewProj", m_glCamera.getViewProj_OrientOnly());

    m_progEnvMap.setUnifUint("backgroundType", backgroundColour);

    // Draw the environment cube, which should output RGBA values in the fragment shader
    m_progEnvMap.draw(m_geomCube);

    // Disable blending after rendering the environment map (optional)
    glDisable(GL_BLEND);
}

void MyGL::setupShaderHandles() {
    m_progPBR.addUniform("u_Model");
    m_progPBR.addUniform("u_ModelInvTr");
    m_progPBR.addUniform("u_ViewProj");
    m_progPBR.addUniform("u_CamPos");
    m_progPBR.addUniform("u_Albedo");
    m_progPBR.addUniform("u_Metallic");
    m_progPBR.addUniform("u_Roughness");
    m_progPBR.addUniform("u_AmbientOcclusion");
    m_progPBR.addUniform("u_AlbedoMap");
    m_progPBR.addUniform("u_MetallicMap");
    m_progPBR.addUniform("u_RoughnessMap");
    m_progPBR.addUniform("u_AOMap");
    m_progPBR.addUniform("u_NormalMap");
    m_progPBR.addUniform("u_UseAlbedoMap");
    m_progPBR.addUniform("u_UseMetallicMap");
    m_progPBR.addUniform("u_UseRoughnessMap");
    m_progPBR.addUniform("u_UseAOMap");
    m_progPBR.addUniform("u_UseNormalMap");
    m_progPBR.addUniform("u_DiffuseIrradianceMap");
    m_progPBR.addUniform("u_GlossyIrradianceMap");
    m_progPBR.addUniform("u_BRDFLookupTexture");
    m_progPBR.addUniform("u_DisplacementMap");
    m_progPBR.addUniform("u_DisplacementMagnitude");
    m_progPBR.addUniform("u_UseDisplacementMap");
    // For glints
    m_progPBR.addUniform("u_GlintNoiseTexture");
    m_progPBR.addUniform("_Glint2023NoiseMapSize");
    m_progPBR.addUniform("u_UseGlint");
    m_progPBR.addUniform("_ScreenSpaceScale");
    m_progPBR.addUniform("_LogMicrofacetDensity");
    m_progPBR.addUniform("_DensityRandomization");

    m_progCubemapConversion.addUniform("u_EquirectangularMap");
    m_progCubemapConversion.addUniform("u_ViewProj");

    m_progEnvMap.addUniform("u_EnvironmentMap");
    m_progEnvMap.addUniform("u_ViewProj");
    m_progEnvMap.addUniform("backgroundType");

    m_progCubemapDiffuseConvolution.addUniform("u_EnvironmentMap");
    m_progCubemapDiffuseConvolution.addUniform("u_ViewProj");

    m_progCubemapGlossyConvolution.addUniform("u_EnvironmentMap");
    m_progCubemapGlossyConvolution.addUniform("u_Roughness");
    m_progCubemapGlossyConvolution.addUniform("u_ViewProj");

    m_progGlintNoise.addUniform("u_Model");
    m_progGlintNoise.addUniform("u_ViewProj");
    m_progGlintNoise.addUniform("_FrameSize");
    m_progGlintNoise.addUniform("_Seed");
}

void MyGL::initPBRunifs() {
    m_progPBR.setUnifVec3("u_Albedo", m_albedo);
    m_progPBR.setUnifFloat("u_AmbientOcclusion", 1.f);
    m_progPBR.setUnifFloat("u_Metallic", 0.5f);
    m_progPBR.setUnifFloat("u_Roughness", 0.5f);
    m_progPBR.setUnifFloat("u_DisplacementMagnitude", 0.2f);
    // Initailise glints
    m_progPBR.setUnifUint("u_UseGlint", 1);
    m_progPBR.setUnifFloat("_ScreenSpaceScale", 2.5f);
    m_progPBR.setUnifFloat("_LogMicrofacetDensity", 40.f);
    m_progPBR.setUnifFloat("_DensityRandomization", 10.f);
}

void MyGL::rotateModel() {
    modelMatrix = glm::rotate(glm::mat4(1.0f), rotationX, glm::vec3(1, 0, 0)) * modelMatrix_default;
    modelMatrix = glm::rotate(glm::mat4(1.0f), rotationY, glm::vec3(0, 1, 0)) * modelMatrix;
    modelMatrix = glm::rotate(glm::mat4(1.0f), rotationZ, glm::vec3(0, 0, 1)) * modelMatrix;
}

void MyGL::slot_setRed(int r) {
    m_albedo.r = r / 100.f;
    m_progPBR.setUnifVec3("u_Albedo", m_albedo);
    std::cout << "Red: " << m_albedo.r << std::endl;
    update();
}
void MyGL::slot_setGreen(int g) {
    m_albedo.g = g / 100.f;
    m_progPBR.setUnifVec3("u_Albedo", m_albedo);
    std::cout << "Green: " << m_albedo.g << std::endl;
    update();
}
void MyGL::slot_setBlue(int b) {
    m_albedo.b = b / 100.f;
    m_progPBR.setUnifVec3("u_Albedo", m_albedo);
    std::cout << "Blue: " << m_albedo.b << std::endl;
    update();
}

void MyGL::slot_setMetallic(int m) {
    m_progPBR.setUnifFloat("u_Metallic", m / 100.f);
    std::cout << "Metallic: " << m / 100.f << std::endl;
    update();
}
void MyGL::slot_setRoughness(int r) {
    m_progPBR.setUnifFloat("u_Roughness", r / 100.f);
    std::cout << "Roughness: " << r / 100.f << std::endl;
    update();
}
void MyGL::slot_setAO(int a) {
    m_progPBR.setUnifFloat("u_AmbientOcclusion", a / 100.f);
    std::cout << "Ambient Occlusion: " << a / 100.f << std::endl;
    update();
}
void MyGL::slot_setDisplacement(double d) {
    m_progPBR.setUnifFloat("u_DisplacementMagnitude", d);
    update();
}

void MyGL::slot_loadEnvMap() {
    QString path = getCurrentPath();
    path.append("/assignment_package/environment_maps/");
    QString filepath = QFileDialog::getOpenFileName(
                        0, QString("Load Environment Map"),
                        path, tr("*.hdr"));
    Texture2DHDR tex(this);
    try {
        tex.create(filepath.toStdString().c_str(), false);
    }
    catch(std::exception &e) {
        std::cout << "Error: Failed to load HDR image" << std::endl;
        return;
    }
    this->m_hdrEnvMap.destroy();
    this->m_hdrEnvMap = tex;
    this->m_cubemapsNotGenerated = true;
    update();
}

void MyGL::slot_loadScene() {
    QString path = getCurrentPath();
    path.append("/assignment_package/models/");
    QString filepath = QFileDialog::getOpenFileName(
                        0, QString("Load Environment Map"),
                        path, tr("*.json"));
    QFile file(filepath);
    if(file.open(QIODevice::ReadOnly)) {
        QByteArray rawData = file.readAll();
        // Parse document
        QJsonDocument doc(QJsonDocument::fromJson(rawData));
        // Get JSON object
        QJsonObject json = doc.object();

        QString obj = json["obj"].toString();
        QString albedo = json["albedo"].toString();
        QString metallic = json["metallic"].toString();
        QString normal = json["normal"].toString();
        QString roughness = json["roughness"].toString();
        QString ambientOcclusion = json["ambientOcclusion"].toString();
        QString displacement = json["displacement"].toString();

        Mesh mesh(this);
        mesh.LoadOBJ(obj, path);


        if(m_textureAlbedo.m_isCreated) {m_textureAlbedo.destroy();}
        if(m_textureMetallic.m_isCreated) {m_textureMetallic.destroy();}
        if(m_textureNormals.m_isCreated) {m_textureNormals.destroy();}
        if(m_textureRoughness.m_isCreated) {m_textureRoughness.destroy();}
        if(m_textureAO.m_isCreated) {m_textureAO.destroy();}
        if(m_textureDisplacement.m_isCreated) {m_textureDisplacement.destroy();}

        if(albedo != "") {
            m_textureAlbedo.create((path + albedo).toStdString().c_str(), true);
        }
        if(metallic != "") {
            m_textureMetallic.create((path + metallic).toStdString().c_str(), true);
        }
        if(normal != "") {
            m_textureNormals.create((path + normal).toStdString().c_str(), true);
        }
        if(roughness != "") {
            m_textureRoughness.create((path + roughness).toStdString().c_str(), true);
        }
        if(ambientOcclusion != "") {
            m_textureAO.create((path + ambientOcclusion).toStdString().c_str(), true);
        }
        if(displacement != "") {
            m_textureDisplacement.create((path + displacement).toStdString().c_str(), true);
        }


        m_geomMesh.destroy();
        m_geomMesh = mesh;
        m_geomMesh.create();
    }
    update();
}

void MyGL::slot_revertToSphere() {
    m_geomMesh.destroy();
    QString path = getCurrentPath();
    path.append("/assignment_package/objs/");
    m_geomMesh.LoadOBJ("sphere.obj", path);
    m_geomMesh.create();

    if(m_textureAlbedo.m_isCreated) {
        m_textureAlbedo.destroy();
    }
    if(m_textureMetallic.m_isCreated) {
        m_textureMetallic.destroy();
    }
    if(m_textureNormals.m_isCreated) {
        m_textureNormals.destroy();
    }
    if(m_textureRoughness.m_isCreated) {
        m_textureRoughness.destroy();
    }
    if(m_textureAO.m_isCreated) {
        m_textureAO.destroy();
    }    
    if(m_textureDisplacement.m_isCreated) {
        m_textureDisplacement.destroy();
    }
    update();
}

void MyGL::slot_loadOBJ() {
    QString path = getCurrentPath();
    path.append("/assignment_package/models/");
    QString filepath = QFileDialog::getOpenFileName(
                        0, QString("Load OBJ File"),
                        path, tr("*.obj"));
    m_geomMesh.destroy();
    m_geomMesh.LoadOBJ(filepath, "");
    m_geomMesh.create();

    if(m_textureAlbedo.m_isCreated) {
        m_textureAlbedo.destroy();
    }
    if(m_textureMetallic.m_isCreated) {
        m_textureMetallic.destroy();
    }
    if(m_textureNormals.m_isCreated) {
        m_textureNormals.destroy();
    }
    if(m_textureRoughness.m_isCreated) {
        m_textureRoughness.destroy();
    }
    if(m_textureAO.m_isCreated) {
        m_textureAO.destroy();
    }
    if(m_textureDisplacement.m_isCreated) {
        m_textureDisplacement.destroy();
    }
    update();
}

void MyGL::slot_saveImage() {
    // Get the filename using a QFileDialog to allow the user to choose where to save the image
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.png *.xpm *.jpg)"));

    // Check if the user provided a filename
    if (fileName.isEmpty()) {
        return; // If the user cancelled the dialog, do nothing
    }

    // Supported extensions
    QStringList supportedExtensions = {"png", "jpg", "xpm"};

    // Extract the extension from the file name
    QString extension = QFileInfo(fileName).suffix().toLower();

    // Check if the file has a valid extension
    if (!supportedExtensions.contains(extension)) {
        QMessageBox::warning(this, tr("Invalid File Extension"), 
                             tr("Please specify a valid file extension when you are entering the filename: .png, .jpg, or .xpm."));
        return;
    }

    // Bind the default framebuffer (typically framebuffer 0)
    makeCurrent();

    // Get the size of the viewport (usually the size of the window)
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int width = viewport[2];
    int height = viewport[3];

    // Allocate memory to store the pixel data
    QByteArray pixelData;
    pixelData.resize(width * height * 4); // 4 channels: RGBA

    // Read the pixel data from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixelData.data());

    // Create a QImage from the pixel data
    QImage image(reinterpret_cast<const uchar*>(pixelData.constData()), width, height, QImage::Format_RGBA8888);

    // OpenGL's origin is bottom-left, while QImage's origin is top-left, so we need to mirror the image vertically
    QImage flippedImage = image.mirrored();

    // Save the image to the specified file
    if (!flippedImage.save(fileName)) {
        qWarning("Failed to save image");
    }
}

void MyGL::slot_setUseGlint(bool b) {
    m_progPBR.setUnifInt("u_UseGlint", b ? 1 : 0);
    update();
}

void MyGL::slot_setScreenSpaceScale(double d) {
    m_progPBR.setUnifFloat("_ScreenSpaceScale", d);
    update();
}

void MyGL::slot_setLogMicrofacetDensity(double d) {
    m_progPBR.setUnifFloat("_LogMicrofacetDensity", d);
    update();
}

void MyGL::slot_setDensityRandomization(double d) {
    m_progPBR.setUnifFloat("_DensityRandomization", d);
    update();
}

void MyGL::slot_setRotationX(double d) {
    rotationX = glm::radians(static_cast<float>(d));
    rotateModel();
    update();
}

void MyGL::slot_setRotationY(double d) {
    rotationY = glm::radians(static_cast<float>(d));
    rotateModel();
    update();
}

void MyGL::slot_setRotationZ(double d) {
    rotationZ = glm::radians(static_cast<float>(d));
    rotateModel();
    update();
}

void MyGL::slot_changeBackgroundColour(int idx) {
    std::cout << "Background colour: " << idx << std::endl;
    backgroundColour = static_cast<unsigned int>(idx);

    update();
}

QString MyGL::getCurrentPath() const {
    QString path = QDir::currentPath();
    path = path.left(path.lastIndexOf("/"));
#ifdef __APPLE__
    path = path.left(path.lastIndexOf("/"));
    path = path.left(path.lastIndexOf("/"));
    path = path.left(path.lastIndexOf("/"));
#endif
    return path;
}
