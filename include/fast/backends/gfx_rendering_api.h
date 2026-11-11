#pragma once

#include <stdint.h>

#include <unordered_map>
#include <set>
#include <utility>
#include <vector>
#include "imconfig.h"

namespace Fast {
struct ShaderProgram;

struct GfxClipParameters {
    bool z_is_from_0_to_1;
    bool invertY;
};

enum FilteringMode { FILTER_THREE_POINT, FILTER_LINEAR, FILTER_NONE };

// A hash function used to hash a: pair<float, float>
struct hash_pair_ff {
    size_t operator()(const std::pair<float, float>& p) const {
        const auto hash1 = std::hash<float>{}(p.first);
        const auto hash2 = std::hash<float>{}(p.second);

        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};

class GfxRenderingAPI {
  public:
    virtual ~GfxRenderingAPI() = default;
    virtual const char* GetName() = 0;
    virtual int GetMaxTextureSize() = 0;
    virtual GfxClipParameters GetClipParameters() = 0;
    virtual void UnloadShader(ShaderProgram* oldPrg) = 0;
    virtual void LoadShader(ShaderProgram* newPrg) = 0;
    virtual void ClearShaderCache() = 0;
    virtual ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) = 0;
    // PORT: shader warm-up (Interpreter::QueueShaderWarmup). A backend that can build programs away from the render
    // thread takes the list and returns true (Metal); otherwise the interpreter builds them a few per frame.
    virtual bool PrebuildShadersAsync(const std::vector<std::pair<uint64_t, uint64_t>>& ids) {
        return false;
    }
    virtual float ShaderPrebuildProgress() { return 1.0f; }
    virtual uint32_t NewTexture() = 0;
    virtual void SelectTexture(int tile, uint32_t textureId) = 0;
    virtual void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) = 0;
    virtual void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) = 0;
    virtual void SetDepthTestAndMask(bool depth_test, bool z_upd) = 0;
    virtual void SetZmodeDecal(bool decal) = 0;
    virtual void SetViewport(int x, int y, int width, int height) = 0;
    virtual void SetScissor(int x, int y, int width, int height) = 0;
    virtual void SetUseAlpha(bool useAlpha) = 0;
    virtual void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) = 0;
    virtual void Init() = 0;
    virtual void OnResize() = 0;
    virtual void StartFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void FinishRender() = 0;
    virtual int CreateFramebuffer() = 0;
    virtual void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                             bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                             bool can_extract_depth) = 0;
    virtual void StartDrawToFramebuffer(int fbId, float noiseScale) = 0;
    virtual void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                 int dstY0, int dstX1, int dstY1) = 0;
    // PORT: copies srcFb's colour into dstFb (the whole w x h), ordered in streamFb's command stream before what is drawn
    // into streamFb next (deferred backends record per-framebuffer command buffers; frame interpolation copies a game
    // frame buffer's picture away and back, see Interpreter::ReplayBegin). Immediate backends: a plain copy.
    virtual void CopyFramebufferInStream(int streamFb, int dstFb, int srcFb, int w, int h) {
        CopyFramebuffer(dstFb, srcFb, 0, 0, w, h, 0, 0, w, h);
    }
    virtual void ClearFramebuffer(bool color, bool depth) = 0;
    virtual void ClearDepthRegion(int x, int y, int w, int h) {
        // Default: full depth clear. Backends that support scissored depth clears
        // (e.g. OpenGL) should override for a more precise partial clear.
        ClearFramebuffer(false, true);
    }
    // PORT: clears the whole colour buffer of the current framebuffer to this colour, alpha included (an N64 fill
    // writes the alpha bit; opaque draws output alpha 1). false: not supported, draw the fill instead.
    virtual bool ClearFramebufferColor(float r, float g, float b, float a) {
        return false;
    }
    // Initialize a new single-sample target from top-to-bottom RGBA8 pixels before its first draw.
    virtual void InitializeFramebuffer(int fbId, uint32_t width, uint32_t height, const uint8_t* rgba) = 0;
    virtual void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) = 0;
    // PORT: this framebuffer is read back after each pass (render-to-texture): backends may set it up for that
    virtual void SetFramebufferReadback(int fbId) {
    }
    virtual void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) = 0;
    virtual std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) = 0;
    virtual void* GetFramebufferTextureId(int fbId) = 0;
    // PORT: the game picture as ImGui draws it into the window, sampled nearest (a sharp upscale of a low render
    // resolution, Fast3dGui::DrawGame). Backends whose ImGui renderer takes the sampler from the texture id (Vulkan)
    // return a second id; the others keep the id and switch the sampler in UseNearestSamplingForImGui, which an ImGui
    // draw callback calls before the picture (ImDrawCallback_ResetRenderState after it restores the default).
    virtual void* GetFramebufferTextureIdNearest(int fbId) {
        return GetFramebufferTextureId(fbId);
    }
    virtual void UseNearestSamplingForImGui() {
    }
    virtual void SelectTextureFb(int fbId) = 0;
    virtual void DeleteTexture(uint32_t texId) = 0;
    virtual void SetTextureFilter(FilteringMode mode) = 0;
    virtual FilteringMode GetTextureFilter() = 0;
    virtual void SetSrgbMode() = 0;
    virtual ImTextureID GetTextureById(int id) = 0;
    virtual void SetCurrentPrimDepth(float depth) = 0;

  protected:
    int8_t mCurrentDepthTest = 0;
    int8_t mCurrentDepthMask = 0;
    int8_t mCurrentZmodeDecal = 0;
    int8_t mLastDepthTest = -1;
    int8_t mLastDepthMask = -1;
    int8_t mLastZmodeDecal = -1;
    bool mSrgbMode = false;
    float mCurrentPrimDepth = 0.0f;
    bool mPrimDepthDirty = true;
};
} // namespace Fast
