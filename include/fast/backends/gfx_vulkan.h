//
//  gfx_vulkan.h
//  libultraship
//
//  PORT: Vulkan rendering backend for fast3d, adapted from Kenix3/libultraship#1265 (KiritoDv) to this tree's
//  rendering API (CPU-transformed vertices, per-combiner shaders). Mirrors the Metal backend's deferred
//  per-framebuffer command buffers: every framebuffer drawn during a frame records into its own primary command
//  buffer, and EndFrame submits them in framebuffer-id order with the screen (fb 0, the swapchain image) last,
//  followed by the present.
//
//  The Vulkan entry points come from volk, loaded at run time from the library SDL loads, so nothing links against
//  a Vulkan loader.
//
#pragma once
#ifdef ENABLE_VULKAN

#include "gfx_rendering_api.h"
#include "gfx_shader_cache.h"
#include "../interpreter.h"

#include <volk.h>
#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdexcept>

struct SDL_Window;
struct ImDrawData;

namespace Ship {
class ConsoleVariable;
class ResourceManager;
} // namespace Ship

namespace Fast {

// PORT: a Vulkan failure the renderer cannot continue from (the port selects OpenGL for the next start and exits,
// Port::ExitAfterVulkanFailure). It must leave the render loop before another driver call uses an invalid handle.
class VulkanFatalError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};
// PORT: the swapchain could not be created or recreated.
class VulkanSwapchainError : public VulkanFatalError {
  public:
    using VulkanFatalError::VulkanFatalError;
};

constexpr uint32_t VK_FRAMES_IN_FLIGHT = 3;

// cantor pairing for the shader pool key (same scheme as Metal)
static inline size_t vk_cantor(uint64_t a, uint64_t b) {
    return (a + b) * (a + b + 1) / 2 + b;
}

struct vk_hash_pair_shader_ids {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return vk_cantor(p.first, p.second);
    }
};

// Per-draw constants: the push-constant block of shaders/vulkan/default.shader.glsl
struct VulkanPushConstants {
    int32_t frameCount;
    float noiseScale;
    float primDepth;
    int32_t textureFiltering[2];
};

constexpr uint32_t VK_MAX_VERTEX_ATTRIBS = 16;



// A program's shader modules and vertex layout, with pipelines built ahead (warm-up workers) for render states
// earlier runs drew it with. Built on the render thread or on a worker; the render thread adopts it.
struct BuiltProgramVK {
    VkShaderModule vs = VK_NULL_HANDLE;
    VkShaderModule fs = VK_NULL_HANDLE;
    uint32_t attribCount = 0;
    VkVertexInputAttributeDescription attribs[VK_MAX_VERTEX_ATTRIBS];
    uint8_t numFloats = 0;
    bool useAlphaBlend = false;
    bool threePoint = false; // the settings the source was generated with
    bool srgb = false;
    std::map<uint32_t, VkPipeline> pipelines;
};

struct ShaderProgramVK {
    uint64_t shader_id0;
    uint64_t shader_id1;

    uint8_t numInputs;
    bool usedTextures[SHADER_MAX_TEXTURES];
    bool markedForDeletion = false;

    // Modules, vertex layout (the interpreter's packing order; fixed locations, see the shader template) and the
    // pipelines keyed by packed render state (msaa level, depth buffer, depth test/write, decal), created lazily
    BuiltProgramVK built;
};

struct TextureDataVK {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE; // owned by the sampler cache
    VkDescriptorSet imguiSet = VK_NULL_HANDLE;
    VkDescriptorSet imguiSetNearest = VK_NULL_HANDLE; // PORT: the same image with mImGuiSamplerNearest
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t filtering = 0;
    bool linear_filtering = false;
    bool isSwapchainAlias = false; // fb 0: the image belongs to the swapchain
};

struct FramebufferVK {
    uint32_t mTextureId = 0;

    VkImage mDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory mDepthMemory = VK_NULL_HANDLE;
    VkImageView mDepthView = VK_NULL_HANDLE;

    // MSAA attachments (resolve target is the main texture)
    VkImage mMsaaImage = VK_NULL_HANDLE;
    VkDeviceMemory mMsaaMemory = VK_NULL_HANDLE;
    VkImageView mMsaaView = VK_NULL_HANDLE;
    VkImage mMsaaDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory mMsaaDepthMemory = VK_NULL_HANDLE;
    VkImageView mMsaaDepthView = VK_NULL_HANDLE;

    VkFramebuffer mFramebuffer = VK_NULL_HANDLE;
    VkRenderPass mRenderPass = VK_NULL_HANDLE; // canonical, owned by the pass cache

    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    uint32_t mMsaaLevel = 1;
    bool mHasDepthBuffer = false;
    bool mRenderTarget = false;

    // Per-frame recording state
    VkCommandBuffer mCommandBuffer = VK_NULL_HANDLE;
    bool mPassActive = false;

    VkPipeline mLastPipeline = VK_NULL_HANDLE;
    VkImageView mLastBoundViews[SHADER_MAX_TEXTURES] = {};
    VkSampler mLastBoundSamplers[SHADER_MAX_TEXTURES] = {};
    float mLastDepthBias = 0.0f;
    VkViewport mViewport = {};
    VkRect2D mScissor = {};
};

class GfxRenderingAPIVK final : public GfxRenderingAPI {
  public:
    GfxRenderingAPIVK(std::shared_ptr<Ship::ConsoleVariable> consoleVariable = nullptr,
                      std::shared_ptr<Ship::ResourceManager> resourceManager = nullptr);
    ~GfxRenderingAPIVK() override;
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    bool PrebuildShadersAsync(const std::vector<std::pair<uint64_t, uint64_t>>& ids) override;
    float ShaderPrebuildProgress() override;
    void ClearShaderCache() override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetCurrentPrimDepth(float depth) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebufferInStream(int streamFb, int dstFb, int srcFb, int w, int h) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ClearDepthRegion(int x, int y, int w, int h) override;
    bool ClearFramebufferColor(float r, float g, float b, float a) override;
    void InitializeFramebuffer(int fbId, uint32_t width, uint32_t height, const uint8_t* rgba) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void* GetFramebufferTextureIdNearest(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;

    // Called from Fast3dGui (mirrors the Metal backend's custom ImGui hooks). VulkanInit creates the instance and
    // device; the window backend calls it before Init().
    bool VulkanInit(SDL_Window* window);
    void NewFrame();
    void RenderDrawData(ImDrawData* drawData);
    void ShutdownImGui();

  private:
    struct FrameSlot {
        VkFence fence = VK_NULL_HANDLE;
        bool fenceInFlight = false;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        // PORT: the command buffers allocated from commandPool. vkResetCommandPool (NewFrame) only resets them, so
        // they are handed out again frame after frame; allocating new ones every frame grew the pool without end
        // (VK_ERROR_OUT_OF_HOST_MEMORY in vkAllocateCommandBuffers on an Intel HD driver after a while in the GB Tower)
        std::vector<VkCommandBuffer> commandBuffers;
        size_t commandBuffersUsed = 0;
        // texture descriptor sets, reset every frame; more pools are added when a frame uses up the first
        std::vector<VkDescriptorPool> descriptorPools;
        size_t descriptorPoolIndex = 0;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;

        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
        uint8_t* vertexMapped = nullptr;
        size_t vertexCapacity = 0;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        uint8_t* stagingMapped = nullptr;
        size_t stagingCapacity = 0;

        std::vector<std::pair<VkImage, VkDeviceMemory>> garbageImages;
        std::vector<VkImageView> garbageViews;
        std::vector<VkBuffer> garbageBuffers;
        std::vector<VkDeviceMemory> garbageMemory;
        std::vector<VkPipeline> garbagePipelines;
        std::vector<VkShaderModule> garbageModules;
    };

    struct ReadbackBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint8_t* mapped = nullptr;
        size_t capacity = 0;
        bool coherent = true;
    };

    uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buffer,
                      VkDeviceMemory& memory, uint8_t** mapped);
    void EnsureReadbackBuffer(ReadbackBuffer& rb, size_t size);
    const uint8_t* ReadbackData(ReadbackBuffer& rb);
    void CreateImage(uint32_t width, uint32_t height, VkSampleCountFlagBits samples, VkFormat format,
                     VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory);
    VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect);
    void TransitionToGeneral(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect);
    void FullBarrier(VkCommandBuffer cmd);
    VkCommandBuffer AcquireCommandBuffer(); // PORT: a reset buffer of this frame's pool (FrameSlot::commandBuffers)
    VkCommandBuffer BeginOneShot();
    void EndOneShot(VkCommandBuffer cmd);
    VkRenderPass GetRenderPass(uint32_t msaaLevel, bool hasDepth);
    VkSampler GetSampler(bool linear, uint32_t cms, uint32_t cmt);
    VkDescriptorPool CreateDescriptorPool();
    VkDescriptorSet AllocateTextureSet();
    std::string BuildShaderSource(const CCFeatures& ccFeatures, bool vertex, bool threePoint, bool srgb);
    // thread-safe (warm-up workers): GLSL -> SPIR-V -> modules, and pipelines for a packed render state
    BuiltProgramVK BuildProgram(uint64_t shaderId0, uint64_t shaderId1, bool threePoint, bool srgb);
    VkPipeline CreatePipeline(const BuiltProgramVK& prg, uint32_t key);
    void DestroyBuiltProgram(BuiltProgramVK& built);
    VkPipeline GetPipelineForState(ShaderProgramVK* prg, FramebufferVK& fb);
    void PrebuildWorker();
    void LoadPipelineList();
    void RecordPipeline(uint64_t shaderId0, uint64_t shaderId1, uint32_t key);
    void SavePipelineCache();
    void DestroyShaderProgram(ShaderProgramVK& prg);
    void EnsureUploadCmd();
    void GrowStaging(FrameSlot& slot, size_t needed);
    void CreateSwapchain(uint32_t width, uint32_t height);
    bool SurfaceHasNoArea() const; // PORT: minimised (see NewFrame)
    void DestroySwapchainViews();
    void RebuildScreenFramebuffer();
    void EnsurePassActive(int fbId);
    void EndPass(FramebufferVK& fb);
    void SubmitFramebufferAndWait(int fbId);
    void CollectGarbage(FrameSlot& slot);
    void DestroyTextureData(TextureDataVK& tex, bool deferred);

    SDL_Window* mWindow = nullptr;

    std::shared_ptr<Ship::ConsoleVariable> mConsoleVariable;
    VkInstance mInstance = VK_NULL_HANDLE;
    VkSurfaceKHR mSurface = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    uint32_t mQueueFamily = 0;
    VkQueue mQueue = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties mDeviceProps = {};
    bool mHasMirrorClampToEdge = false;
    bool mHasDepthClamp = false;
    bool mHasDualSrcBlend = false;

    VkSwapchainKHR mSwapchain = VK_NULL_HANDLE;
    VkFormat mSwapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D mSwapchainExtent = {};
    std::vector<VkImage> mSwapchainImages;
    std::vector<VkImageView> mSwapchainViews;
    std::vector<VkFramebuffer> mSwapchainFramebuffers;
    std::vector<bool> mSwapchainImageUsed; // first-use layout tracking
    uint32_t mSwapchainImageIndex = 0;
    bool mSwapchainNeedsRecreate = false;
    bool mVsync = true;
    bool mFrameActive = false;

    FrameSlot mFrames[VK_FRAMES_IN_FLIGHT];
    uint32_t mFrameSlot = 0;

    size_t mVertexRingOffset = 0;
    size_t mVertexRingCapacity = 0;
    size_t mStagingOffset = 0;

    VkCommandBuffer mUploadCmd = VK_NULL_HANDLE;
    VkFence mReadbackFence = VK_NULL_HANDLE; // SubmitFramebufferAndWait

    VkDescriptorSetLayout mTextureSetLayout = VK_NULL_HANDLE; // 6 combined image samplers (fragment)
    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;         // + the push constants
    ShaderDiskCache mShaderCache;
    VkPipelineCache mPipelineCache = VK_NULL_HANDLE;

    // PORT: shader warm-up and the on-disk caches. Workers build the queued programs (and the pipelines listed for
    // them) into mPrebuilt; CreateAndLoadNewShader takes a finished one, or waits for one being built. Only the
    // render thread touches mShaderProgramPool. vulkan_pipelines.txt lists the (program, render state) pipelines drawn
    // so far, vulkan_pipeline_cache.bin is the driver's VkPipelineCache data.
    std::mutex mRenderPassMutex;
    std::mutex mPrebuildMutex;
    std::condition_variable mPrebuildCond;
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint32_t>>> mPrebuildQueue;
    size_t mPrebuildNext = 0;
    bool mPrebuildThreePoint = false;
    bool mPrebuildSrgb = false;
    std::map<std::pair<uint64_t, uint64_t>, BuiltProgramVK> mPrebuilt;  // finished, not yet drawn
    std::set<std::pair<uint64_t, uint64_t>> mPrebuilding;               // being built by a worker
    std::set<std::pair<uint64_t, uint64_t>> mPrebuildSkip;              // built by the render thread itself
    std::vector<std::thread> mPrebuildThreads;
    std::atomic<bool> mPrebuildStop{ false };
    int mPrebuildRunning = 0;
    size_t mPrebuildBuilt = 0;
    std::chrono::steady_clock::time_point mPrebuildStart;
    std::map<std::pair<uint64_t, uint64_t>, std::vector<uint32_t>> mPipelineList;
    std::set<std::tuple<uint64_t, uint64_t, uint32_t>> mKnownPipelines;
    std::string mPipelineListPath;
    std::string mPipelineCachePath;
    std::atomic<bool> mPipelineCacheDirty{ false }; // set by workers too
    std::chrono::steady_clock::time_point mPipelineCacheSaved;

    std::map<uint32_t, VkRenderPass> mRenderPassCache; // key: msaa | (hasDepth << 8)
    std::map<uint32_t, VkSampler> mSamplerCache;       // key: linear | cms<<1 | cmt<<9 | filter mode<<17

    std::unordered_map<std::pair<uint64_t, uint64_t>, ShaderProgramVK, vk_hash_pair_shader_ids> mShaderProgramPool;

    std::vector<TextureDataVK> mTextures;
    std::vector<FramebufferVK> mFramebuffers;
    std::vector<int> mDrawnFramebuffers;

    TextureDataVK mDummyTexture; // bound to unused sampler slots
    VkSampler mImGuiSampler = VK_NULL_HANDLE;
    VkSampler mImGuiSamplerNearest = VK_NULL_HANDLE; // PORT: the game picture's sharp upscale (DrawGame)

    ShaderProgramVK* mShaderProgram = nullptr;

    int mCurrentTile = 0;
    uint32_t mCurrentTextureIds[SHADER_MAX_TEXTURES] = {};

    int32_t mRenderTargetHeight = 0;
    int mCurrentFramebuffer = 0;
    FilteringMode mCurrentFilterMode = FILTER_THREE_POINT;
    float mNoiseScale = 1.0f;
    int32_t mFrameCount = 0;

    // Offscreen readbacks (synchronous) and the deferred screen readback (mirrors Metal: the copy is recorded at the
    // end of a frame, the CPU reads it in the next one once that frame's fence has signalled)
    ReadbackBuffer mReadback;
    ReadbackBuffer mScreenReadback;
    uint32_t mScreenReadbackWidth = 0;
    uint32_t mScreenReadbackHeight = 0;
    bool mScreenReadbackRequested = false;
    bool mScreenReadbackDataReady = false;
    uint32_t mScreenReadbackSlot = 0;

    bool mImGuiInitialized = false;
};

} // namespace Fast

// Loads the Vulkan library for SDL (with the Homebrew MoltenVK locations as a fallback on macOS); before a window with
// SDL_WINDOW_VULKAN is created.
bool Vulkan_LoadLibrary();
// A Vulkan driver with at least one device is installed.
bool Vulkan_IsSupported();

#endif
