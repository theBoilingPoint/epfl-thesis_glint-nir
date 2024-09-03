#pragma once
#include "openglcontext.h"
#include "glm_includes.h"
#include "texture.h"

#define mkU std::make_unique
#define uPtr std::unique_ptr

enum class GBufferOutputType : unsigned int {
    POSITION_WORLD,   // stores world-space position
    NORMAL,           // stores world-space normal
    ALBEDO,           // stores material albedo
    METAL_ROUGH_MASK, // stores metallic, roughness, and geom mask as RGB channels
    PBR,              // stores PBR shader LTE output
    SSR,              // stores the screen-space reflection of the scene
    SSR_BLUR0,        // first level of blurred glossy reflection
    SSR_BLUR1,        // second level of blurred glossy reflection
    SSR_BLUR2,        // third level of blurred glossy reflection
    SSR_BLUR3,        // fourth level of blurred glossy reflection
    NONE,          // Used for cube map, which doesn't output to G buffer
    DOF, // stores the depth of field params (i.e. distance to the focal plane)
    PREPROCESSED_PBR, // stores the preprocessed PBR texture
    BLURRED_PBR, // stores the blurred PBR texture
    GLINT_NOISE // stores the glint noise texture
};

// A class representing a frame buffer in the OpenGL pipeline.
// Stores three GPU handles: one to a frame buffer object, one to
// a texture object that will store the frame buffer's contents,
// and one to a depth buffer needed to properly render to the frame
// buffer.
// Redirect your render output to a FrameBuffer by invoking
// bindFrameBuffer() before ShaderProgram::draw, and read
// from the frame buffer's output texture by invoking
// bindToTextureSlot() and then associating a ShaderProgram's
// sampler2d with the appropriate texture slot.
class FrameBuffer {
protected:
    OpenGLContext *mp_context;
    GLuint m_frameBuffer;
    GLuint m_depthRenderBuffer;
    std::unordered_map<GBufferOutputType, uPtr<Texture>> m_outputTextures;

    unsigned int m_width, m_height, m_devicePixelRatio;
    bool m_created;

    unsigned int m_textureSlot;

public:
    FrameBuffer(OpenGLContext *context, unsigned int width, unsigned int height, unsigned int devicePixelRatio);
    // Make sure to call resize from MyGL::resizeGL to keep your frame buffer up to date with
    // your screen dimensions
    void resize(unsigned int width, unsigned int height, unsigned int devicePixelRatio);
    // Initialize all GPU-side data required
    virtual void create(bool mipmap = false);
    // Deallocate all GPU-side data
    virtual void destroy();
    void bindFrameBuffer();
    void bindRenderBuffer(unsigned int w, unsigned int h);
    // Associate our output texture with the indicated texture slot
    virtual void bindToTextureSlot(unsigned int slot, GBufferOutputType tex);
    unsigned int getTextureSlot() const;
    inline unsigned int width() const {
        return m_width;
    }
    inline unsigned int height() const {
        return m_height;
    }

    // Append a Texture to receive the next-lowest unused
    // GL_COLOR_ATTACHMENT as data.
    void addTexture(GBufferOutputType a);

    std::unordered_map<GBufferOutputType, uPtr<Texture>> &getOutputTextures();
};

class CubeMapFrameBuffer : public FrameBuffer {
protected:
    unsigned int m_outputCubeMap;

public:
    CubeMapFrameBuffer(OpenGLContext *context, unsigned int width, unsigned int height, unsigned int devicePixelRatio);

    void create(bool mipmap = false) override;
    void destroy() override;
    void bindToTextureSlot(unsigned int slot);

    unsigned int getCubemapHandle() const;

    void generateMipMaps();
};
