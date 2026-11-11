//
//  gfx_metal.cpp
//  libultraship
//
//  Created by David Chavez on 16.08.22.
//

#ifdef __APPLE__

#include <chrono>
#include "fast/backends/gfx_metal.h"

#include <unordered_set>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <time.h>
#include <math.h>
#include <cmath>
#include <stddef.h>

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <SDL3/SDL_render.h>
#include <imgui_impl_metal.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include "fast/backends/gfx_metal_shader.h"
#include "fast/backends/resize_keep.h"

#include "libultraship/libultra/abi.h"
#include "ship/config/ConsoleVariable.h"
#include "ship/resource/ResourceManager.h"

#define ARRAY_COUNT(arr) (int32_t)(sizeof(arr) / sizeof(arr[0]))

// MARK: - Helpers
namespace Fast {
extern bool gPortReplayFrame; // interpreter.cpp (frame interpolation)

GfxRenderingAPIMetal::GfxRenderingAPIMetal(std::shared_ptr<Ship::ConsoleVariable> consoleVariable,
                                           std::shared_ptr<Ship::ResourceManager> resourceManager)
    : mConsoleVariable(std::move(consoleVariable)), mResourceManager(std::move(resourceManager)) {
}

static MTL::SamplerAddressMode gfx_cm_to_metal(uint32_t val) {
    switch (val) {
        case G_TX_NOMIRROR | G_TX_CLAMP:
            return MTL::SamplerAddressModeClampToEdge;
        case G_TX_MIRROR | G_TX_WRAP:
            return MTL::SamplerAddressModeMirrorRepeat;
        case G_TX_MIRROR | G_TX_CLAMP:
            return MTL::SamplerAddressModeMirrorClampToEdge;
        case G_TX_NOMIRROR | G_TX_WRAP:
            return MTL::SamplerAddressModeRepeat;
    }

    return MTL::SamplerAddressModeClampToEdge;
}

// MARK: - ImGui & SDL Wrappers

bool GfxRenderingAPIMetal::NonUniformThreadGroupSupported() {
#ifdef __IOS__
    // iOS devices with A11 or later support dispatch threads
    return mDevice->supportsFamily(MTL::GPUFamilyApple4);
#else
    // macOS devices with Metal 2 support dispatch threads
    return mDevice->supportsFamily(MTL::GPUFamilyMac2);
#endif
}

bool GfxRenderingAPIMetal::MetalInit(SDL_Renderer* renderer) {
    mRenderer = renderer;
    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    mLayer = (CA::MetalLayer*)SDL_GetRenderMetalLayer(renderer);
    mLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    mDevice = mLayer->device();
    mCommandQueue = mDevice->newCommandQueue();
    mReadbackQueue = mDevice->newCommandQueue();

    for (size_t i = 0; i < kMaxVertexBufferPoolSize; i++) {
        MTL::Buffer* new_buffer = mDevice->newBuffer(256 * 32 * 3 * sizeof(float) * 50, MTL::ResourceStorageModeShared);
        mVertexBufferPool[i] = new_buffer;
    }

    autorelease_pool->release();
    mNonUniformThreadgroupSupported = NonUniformThreadGroupSupported();

    return ImGui_ImplMetal_Init(mDevice);
}

static void SetupScreenFramebuffer(uint32_t width, uint32_t height);

void GfxRenderingAPIMetal::NewFrame() {
    int width, height;
    SDL_GetCurrentRenderOutputSize(mRenderer, &width, &height);
    SetupScreenFramebuffer(width, height);

    MTL::RenderPassDescriptor* current_render_pass = mFramebuffers[0].mRenderPassDescriptor;
    ImGui_ImplMetal_NewFrame(current_render_pass);
}

void GfxRenderingAPIMetal::SetupFloatingFrame() {
    // We need the descriptor for the main framebuffer and to clear the existing depth attachment
    // so that we can set ImGui up again for our floating windows. Helps avoid Metal API validation issues.
    MTL::RenderPassDescriptor* current_render_pass = mFramebuffers[0].mRenderPassDescriptor;
    current_render_pass->setDepthAttachment(nullptr);
    ImGui_ImplMetal_NewFrame(current_render_pass);
}

void GfxRenderingAPIMetal::RenderDrawData(ImDrawData* drawData) {
    auto framebuffer = mFramebuffers[0];

    // Workaround for detecting when transitioning to/from full screen mode.
    MTL::Texture* screen_texture = mTextures[framebuffer.mTextureId].texture;
    int fb_width = (int)(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    int fb_height = (int)(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (screen_texture->width() != fb_width || screen_texture->height() != fb_height)
        return;

    ImGui_ImplMetal_RenderDrawData(drawData, framebuffer.mCommandBuffer, framebuffer.mCommandEncoder);
}

// PORT: imgui_impl_metal's pipeline (same shaders, vertex layout and blending) with a nearest-magnifying sampler, set
// on ImGui's encoder by a draw callback before the game picture (Fast3dGui::DrawGame); ImDrawCallback_ResetRenderState
// after the picture puts ImGui's own pipeline back. Made again when the screen's pixel formats or sample count change.
void GfxRenderingAPIMetal::UseNearestSamplingForImGui() {
    FramebufferMetal& framebuffer = mFramebuffers[0];
    if (framebuffer.mCommandEncoder == nullptr || framebuffer.mRenderPassDescriptor == nullptr) {
        return;
    }
    MTL::RenderPassDescriptor* pass = framebuffer.mRenderPassDescriptor;
    MTL::Texture* color = pass->colorAttachments()->object(0)->texture();
    if (color == nullptr) {
        return;
    }
    MTL::Texture* depth = pass->depthAttachment() != nullptr ? pass->depthAttachment()->texture() : nullptr;
    MTL::Texture* stencil = pass->stencilAttachment() != nullptr ? pass->stencilAttachment()->texture() : nullptr;
    const MTL::PixelFormat depthFormat = depth != nullptr ? depth->pixelFormat() : MTL::PixelFormatInvalid;
    const MTL::PixelFormat stencilFormat = stencil != nullptr ? stencil->pixelFormat() : MTL::PixelFormatInvalid;
    const uint64_t key = ((uint64_t)color->pixelFormat() << 40) | ((uint64_t)depthFormat << 24) |
                         ((uint64_t)stencilFormat << 8) | (uint64_t)color->sampleCount();
    if (mImGuiNearestPipeline == nullptr || mImGuiNearestKey != key) {
        static MTL::Function* sVertex = nullptr;
        static MTL::Function* sFragment = nullptr;
        if (sVertex == nullptr) {
            static const char* source = R"(
                #include <metal_stdlib>
                using namespace metal;
                struct Uniforms { float4x4 projectionMatrix; };
                struct VertexIn {
                    float2 position  [[attribute(0)]];
                    float2 texCoords [[attribute(1)]];
                    uchar4 color     [[attribute(2)]];
                };
                struct VertexOut { float4 position [[position]]; float2 texCoords; float4 color; };
                vertex VertexOut imguiNearestVertex(VertexIn in [[stage_in]], constant Uniforms& uniforms [[buffer(1)]]) {
                    VertexOut out;
                    out.position = uniforms.projectionMatrix * float4(in.position, 0, 1);
                    out.texCoords = in.texCoords;
                    out.color = float4(in.color) / float4(255.0);
                    return out;
                }
                fragment half4 imguiNearestFragment(VertexOut in [[stage_in]],
                                                    texture2d<half, access::sample> texture [[texture(0)]]) {
                    constexpr sampler nearestSampler(coord::normalized, min_filter::linear, mag_filter::nearest,
                                                     mip_filter::linear);
                    return half4(in.color) * texture.sample(nearestSampler, in.texCoords);
                }
            )";
            NS::Error* error = nullptr;
            MTL::Library* library =
                mDevice->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), nullptr, &error);
            if (library == nullptr) {
                SPDLOG_ERROR("Failed to compile the ImGui nearest shader: {}",
                             error ? error->localizedDescription()->cString(NS::UTF8StringEncoding) : "?");
                return;
            }
            sVertex = library->newFunction(NS::String::string("imguiNearestVertex", NS::UTF8StringEncoding));
            sFragment = library->newFunction(NS::String::string("imguiNearestFragment", NS::UTF8StringEncoding));
            library->release();
        }
        MTL::VertexDescriptor* vd = MTL::VertexDescriptor::alloc()->init();
        vd->attributes()->object(0)->setOffset(offsetof(ImDrawVert, pos));
        vd->attributes()->object(0)->setFormat(MTL::VertexFormatFloat2);
        vd->attributes()->object(0)->setBufferIndex(0);
        vd->attributes()->object(1)->setOffset(offsetof(ImDrawVert, uv));
        vd->attributes()->object(1)->setFormat(MTL::VertexFormatFloat2);
        vd->attributes()->object(1)->setBufferIndex(0);
        vd->attributes()->object(2)->setOffset(offsetof(ImDrawVert, col));
        vd->attributes()->object(2)->setFormat(MTL::VertexFormatUChar4);
        vd->attributes()->object(2)->setBufferIndex(0);
        vd->layouts()->object(0)->setStepRate(1);
        vd->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
        vd->layouts()->object(0)->setStride(sizeof(ImDrawVert));
        MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(sVertex);
        desc->setFragmentFunction(sFragment);
        desc->setVertexDescriptor(vd);
        desc->setRasterSampleCount(color->sampleCount());
        MTL::RenderPipelineColorAttachmentDescriptor* c0 = desc->colorAttachments()->object(0);
        c0->setPixelFormat(color->pixelFormat());
        c0->setBlendingEnabled(true);
        c0->setRgbBlendOperation(MTL::BlendOperationAdd);
        c0->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        c0->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        c0->setAlphaBlendOperation(MTL::BlendOperationAdd);
        c0->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        c0->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        desc->setDepthAttachmentPixelFormat(depthFormat);
        desc->setStencilAttachmentPixelFormat(stencilFormat);
        NS::Error* error = nullptr;
        MTL::RenderPipelineState* pipeline = mDevice->newRenderPipelineState(desc, &error);
        desc->release();
        vd->release();
        if (pipeline == nullptr) {
            SPDLOG_ERROR("Failed to create the ImGui nearest pipeline: {}",
                         error ? error->localizedDescription()->cString(NS::UTF8StringEncoding) : "?");
            return;
        }
        if (mImGuiNearestPipeline != nullptr) {
            mImGuiNearestPipeline->release();
        }
        mImGuiNearestPipeline = pipeline;
        mImGuiNearestKey = key;
    }
    framebuffer.mCommandEncoder->setRenderPipelineState(mImGuiNearestPipeline);
}

// MARK: - Metal Graphics Rendering API

const char* GfxRenderingAPIMetal::GetName() {
    return "Metal";
}

int GfxRenderingAPIMetal::GetMaxTextureSize() {
    return mDevice->supportsFamily(MTL::GPUFamilyApple3) ? 16384 : 8192;
}

void GfxRenderingAPIMetal::Init() {
    // Create the default framebuffer which represents the window
    FramebufferMetal& fb = mFramebuffers[CreateFramebuffer()];
    fb.mMsaaLevel = 1;

    // Check device for supported msaa levels
    for (uint32_t sample_count = 1; sample_count <= METAL_MAX_MULTISAMPLE_SAMPLE_COUNT; sample_count++) {
        if (mDevice->supportsTextureSampleCount(sample_count)) {
            mMsaaNumQualityLevels[sample_count - 1] = 1;
        } else {
            mMsaaNumQualityLevels[sample_count - 1] = 0;
        }
    }

    // Compute shader for retrieving depth values
    const char* depth_shader = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct CoordUniforms {
            uint2 coords[1024];
        };

        kernel void depthKernel(depth2d<float, access::read> depth_texture [[ texture(0) ]],
                                     constant CoordUniforms& query_coords [[ buffer(0) ]],
                                     device float* output_values [[ buffer(1) ]],
                                     ushort2 thread_position [[ thread_position_in_grid ]]) {
            uint2 coord = query_coords.coords[thread_position.x];
            output_values[thread_position.x] = depth_texture.read(coord);
        }

        kernel void convertToRGB5A1(texture2d<half, access::read> inTexture [[ texture(0) ]],
                                    device short* outputBuffer [[ buffer(0) ]],
                                    uint2 gid [[ thread_position_in_grid ]]) {
            uint index = gid.x + (inTexture.get_width() * gid.y);
            half4 pixel = inTexture.read(gid);
            // Recover the stored UNORM8 bits, then take the high five bits as the RDP does.
            uint r = uint(round(float(pixel.r) * 255.0f)) >> 3;
            uint g = uint(round(float(pixel.g) * 255.0f)) >> 3;
            uint b = uint(round(float(pixel.b) * 255.0f)) >> 3;
            uint a = pixel.a > 0;
            outputBuffer[index] = (r << 11) | (g << 6) | (b << 1) | a;
        }
    )";

    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    NS::Error* error = nullptr;
    MTL::Library* library =
        mDevice->newLibrary(NS::String::string(depth_shader, NS::UTF8StringEncoding), nullptr, &error);

    if (error != nullptr)
        SPDLOG_ERROR("Failed to compile shader library: {}",
                     error->localizedDescription()->cString(NS::UTF8StringEncoding));

    mDepthComputeFunction = library->newFunction(NS::String::string("depthKernel", NS::UTF8StringEncoding));
    mConvertToRgb5a1Function = library->newFunction(NS::String::string("convertToRGB5A1", NS::UTF8StringEncoding));

    library->release();
    autorelease_pool->release();

    gfx_metal_shader_set_resource_manager(mResourceManager);
}

struct GfxClipParameters GfxRenderingAPIMetal::GetClipParameters() {
    return { true, false };
}

void GfxRenderingAPIMetal::UnloadShader(struct ShaderProgram* old_prg) {
}

void GfxRenderingAPIMetal::LoadShader(struct ShaderProgram* new_prg) {
    mShaderProgram = (struct ShaderProgramMetal*)new_prg;
}

void GfxRenderingAPIMetal::ClearShaderCache() {
    for (auto& [key, prg] : mShaderProgramPool) {
        prg.markedForDeletion = true;
    }
}

// PORT: one program's library, descriptor and pipeline, built on any thread (the render thread, or a warm-up worker)
GfxRenderingAPIMetal::BuiltShader GfxRenderingAPIMetal::BuildShader(uint64_t shader_id0, uint64_t shader_id1,
                                                                    bool three_point, bool srgb, uint32_t msaa_level) {
    // gfx_metal_build_shader keeps its state in statics
    static std::mutex sSourceMutex;
    CCFeatures cc_features;
    gfx_cc_get_features(shader_id0, shader_id1, &cc_features);

    BuiltShader built;
    built.threePoint = three_point;
    built.srgb = srgb;
    built.msaaLevel = msaa_level;
    std::string buf;
    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    MTL::VertexDescriptor* vertex_descriptor;
    {
        std::lock_guard<std::mutex> lock(sSourceMutex);
        vertex_descriptor = gfx_metal_build_shader(buf, built.numFloats, cc_features, three_point);
    }
    // debug: PORT_SHADER_SALT=<text> makes every source new to the Metal cache (measures cold builds)
    if (const char* salt = getenv("PORT_SHADER_SALT")) {
        buf += std::string("\n// ") + salt + "\n";
    }

    NS::Error* error = nullptr;
    MTL::Library* library =
        mDevice->newLibrary(NS::String::string(buf.data(), NS::UTF8StringEncoding), nullptr, &error);

    if (error != nullptr)
        SPDLOG_ERROR("Failed to compile shader library, error {}",
                     error->localizedDescription()->cString(NS::UTF8StringEncoding));

    MTL::RenderPipelineDescriptor* pipeline_descriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    MTL::Function* vertexFunc = library->newFunction(NS::String::string("vertexShader", NS::UTF8StringEncoding));
    MTL::Function* fragmentFunc = library->newFunction(NS::String::string("fragmentShader", NS::UTF8StringEncoding));

    pipeline_descriptor->setVertexFunction(vertexFunc);
    pipeline_descriptor->setFragmentFunction(fragmentFunc);
    pipeline_descriptor->setVertexDescriptor(vertex_descriptor);

    pipeline_descriptor->colorAttachments()->object(0)->setPixelFormat(srgb ? MTL::PixelFormatBGRA8Unorm_sRGB
                                                                                 : MTL::PixelFormatBGRA8Unorm);
    pipeline_descriptor->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    if (cc_features.opt_alpha) {
        pipeline_descriptor->colorAttachments()->object(0)->setBlendingEnabled(true);
        // PORT: dual-source (default.shader.metal): RGB = src * a + dst * (1 - a) with a = color1.a; alpha = coverage
        // (color.a: 1, or 0 for o_invisible draws, which keep the old alpha). Was alpha = dst: in a render-to-texture
        // image cleared to transparent, translucent or alpha-0 texels stayed invisible where the N64 shows them.
        static const bool sOldAlphaBlend = getenv("PORT_OLD_ALPHA_BLEND") != nullptr; // debug: alpha = dst
        pipeline_descriptor->colorAttachments()->object(0)->setSourceRGBBlendFactor(MTL::BlendFactorSource1Alpha);
        pipeline_descriptor->colorAttachments()->object(0)->setDestinationRGBBlendFactor(
            MTL::BlendFactorOneMinusSource1Alpha);
        pipeline_descriptor->colorAttachments()->object(0)->setRgbBlendOperation(MTL::BlendOperationAdd);
        pipeline_descriptor->colorAttachments()->object(0)->setSourceAlphaBlendFactor(
            sOldAlphaBlend ? MTL::BlendFactorZero : MTL::BlendFactorOne);
        pipeline_descriptor->colorAttachments()->object(0)->setDestinationAlphaBlendFactor(
            sOldAlphaBlend ? MTL::BlendFactorOne : MTL::BlendFactorOneMinusSourceAlpha);
        pipeline_descriptor->colorAttachments()->object(0)->setAlphaBlendOperation(MTL::BlendOperationAdd);
        pipeline_descriptor->colorAttachments()->object(0)->setWriteMask(MTL::ColorWriteMaskAll);
    } else {
        pipeline_descriptor->colorAttachments()->object(0)->setBlendingEnabled(false);
        pipeline_descriptor->colorAttachments()->object(0)->setWriteMask(MTL::ColorWriteMaskAll);
    }

    // PORT: only the window's msaa level now; other levels when first used (PipelineFor)
    pipeline_descriptor->setSampleCount(msaa_level);
    error = nullptr;
    built.pipeline = mDevice->newRenderPipelineState(pipeline_descriptor, &error);
    if (!built.pipeline || error != nullptr) {
        SPDLOG_ERROR("Failed to create pipeline state, error {}",
                     error->localizedDescription()->cString(NS::UTF8StringEncoding));
    }
    built.descriptor = pipeline_descriptor; // keeps the alloc's reference

    vertexFunc->release();
    fragmentFunc->release();
    library->release();
    autorelease_pool->release();
    return built;
}

struct ShaderProgram* GfxRenderingAPIMetal::CreateAndLoadNewShader(uint64_t shader_id0, uint64_t shader_id1) {
    // PORT_TRACE_PACING=1 or PORT_TRACE_SHADERS=1: how long each new shader took (a draw waits for it)
    static const bool sTraceShaders = getenv("PORT_TRACE_PACING") != nullptr || getenv("PORT_TRACE_SHADERS") != nullptr;
    auto buildStart = std::chrono::steady_clock::now();
    bool three_point = mCurrentFilterMode == FILTER_THREE_POINT;
    uint32_t msaa_level = std::clamp<uint32_t>(mFramebuffers[0].mMsaaLevel, 1, 8);
    auto key = std::make_pair(shader_id0, shader_id1);

    // a program the warm-up built (or is building: wait for it, it is further along than a new build here)
    BuiltShader built;
    const char* how = "built";
    {
        std::unique_lock<std::mutex> lock(mPrebuildMutex);
        mPrebuildCond.wait(lock, [&] { return mPrebuilding.count(key) == 0; });
        auto it = mPrebuilt.find(key);
        if (it != mPrebuilt.end()) {
            built = it->second;
            mPrebuilt.erase(it);
            how = "prebuilt";
        } else {
            mPrebuildSkip.insert(key);
        }
    }
    if (built.descriptor != nullptr && (built.threePoint != three_point || built.srgb != mSrgbMode)) {
        // built for other settings (the filter changed since)
        built.descriptor->release();
        if (built.pipeline != nullptr) {
            built.pipeline->release();
        }
        built = {};
    }
    if (built.descriptor == nullptr) {
        built = BuildShader(shader_id0, shader_id1, three_point, mSrgbMode, msaa_level);
        how = "built";
    }

    CCFeatures cc_features;
    gfx_cc_get_features(shader_id0, shader_id1, &cc_features);
    struct ShaderProgramMetal* prg = &mShaderProgramPool[std::make_pair(shader_id0, shader_id1)];
    prg->shader_id0 = shader_id0;
    prg->shader_id1 = shader_id1;
    prg->usedTextures[0] = cc_features.usedTextures[0];
    prg->usedTextures[1] = cc_features.usedTextures[1];
    prg->usedTextures[2] = cc_features.used_masks[0];
    prg->usedTextures[3] = cc_features.used_masks[1];
    prg->usedTextures[4] = cc_features.used_blend[0];
    prg->usedTextures[5] = cc_features.used_blend[1];
    prg->numInputs = cc_features.numInputs;
    prg->numFloats = built.numFloats;
    prg->pipeline_descriptor = built.descriptor;
    prg->pipeline_state_variants[built.msaaLevel] = built.pipeline;
    PipelineFor(prg, mFramebuffers[0].mMsaaLevel);

    LoadShader((struct ShaderProgram*)prg);

    if (sTraceShaders) {
        fprintf(stderr, "[shader] %016llx/%016llx %s in %.1f ms\n", (unsigned long long)shader_id0,
                (unsigned long long)shader_id1, how,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count());
    }
    return (struct ShaderProgram*)prg;
}

// PORT: shader warm-up: builds the programs on worker threads; CreateAndLoadNewShader takes them
bool GfxRenderingAPIMetal::PrebuildShadersAsync(const std::vector<std::pair<uint64_t, uint64_t>>& ids) {
    std::lock_guard<std::mutex> lock(mPrebuildMutex);
    mPrebuildQueue.insert(mPrebuildQueue.end(), ids.begin(), ids.end());
    mPrebuildThreePoint = mCurrentFilterMode == FILTER_THREE_POINT;
    mPrebuildSrgb = mSrgbMode;
    mPrebuildMsaa = std::clamp<uint32_t>(mFramebuffers.empty() ? 1 : mFramebuffers[0].mMsaaLevel, 1, 8);
    if (mPrebuildThreads.empty() && !ids.empty()) {
        // the Metal compiler runs in its own processes: a few builds at a time finish sooner (cold, on 10 cores: 111
        // programs in 35 s with one thread, 13 s with 4, 8.5 s with 8)
        unsigned n = std::clamp(std::thread::hardware_concurrency() / 2, 2u, 6u);
        if (const char* threads = getenv("PORT_SHADER_THREADS")) { // debug: the number of builder threads
            n = std::clamp(atoi(threads), 1, 32);
        }
        mPrebuildStart = std::chrono::steady_clock::now();
        mPrebuildRunning = n;
        for (unsigned i = 0; i < n; i++) {
            mPrebuildThreads.emplace_back([this] { PrebuildWorker(); });
        }
    }
    return true;
}

float GfxRenderingAPIMetal::ShaderPrebuildProgress() {
    std::lock_guard<std::mutex> lock(mPrebuildMutex);
    if (mPrebuildQueue.empty()) return 1.0f;
    return float(mPrebuildNext - mPrebuilding.size()) / float(mPrebuildQueue.size());
}

void GfxRenderingAPIMetal::PrebuildWorker() {
    static const bool sTrace = getenv("PORT_TRACE_PACING") != nullptr || getenv("PORT_TRACE_SHADERS") != nullptr;
    std::unique_lock<std::mutex> lock(mPrebuildMutex);
    while (!mPrebuildStop && mPrebuildNext < mPrebuildQueue.size()) {
        auto key = mPrebuildQueue[mPrebuildNext++];
        if (mPrebuildSkip.count(key) || mPrebuilt.count(key) || mPrebuilding.count(key)) {
            continue;
        }
        mPrebuilding.insert(key);
        bool three_point = mPrebuildThreePoint, srgb = mPrebuildSrgb;
        uint32_t msaa = mPrebuildMsaa;
        lock.unlock();
        BuiltShader built = BuildShader(key.first, key.second, three_point, srgb, msaa);
        lock.lock();
        mPrebuilding.erase(key);
        mPrebuilt[key] = built;
        mPrebuildBuilt++;
        mPrebuildCond.notify_all();
    }
    if (--mPrebuildRunning == 0 && sTrace) {
        fprintf(stderr, "[shader] warm-up done: %zu programs queued, %zu built ahead, in %.0f ms\n",
                mPrebuildQueue.size(), mPrebuildBuilt,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mPrebuildStart).count());
    }
}

GfxRenderingAPIMetal::~GfxRenderingAPIMetal() {
    mPrebuildStop = true;
    for (auto& t : mPrebuildThreads) {
        t.join();
    }
}

// PORT: the program's pipeline for an msaa level, built on first use
MTL::RenderPipelineState* GfxRenderingAPIMetal::PipelineFor(ShaderProgramMetal* prg, int msaa_level) {
    if (msaa_level < 1 || msaa_level > 8) {
        msaa_level = 1;
    }
    if (prg->pipeline_state_variants[msaa_level] == nullptr && prg->pipeline_descriptor != nullptr) {
        NS::Error* error = nullptr;
        prg->pipeline_descriptor->setSampleCount(msaa_level);
        MTL::RenderPipelineState* pipeline_state = mDevice->newRenderPipelineState(prg->pipeline_descriptor, &error);
        if (!pipeline_state || error != nullptr) {
            SPDLOG_ERROR("Failed to create pipeline state, error {}",
                         error->localizedDescription()->cString(NS::UTF8StringEncoding));
        }
        prg->pipeline_state_variants[msaa_level] = pipeline_state;
    }
    return prg->pipeline_state_variants[msaa_level];
}

struct ShaderProgram* GfxRenderingAPIMetal::LookupShader(uint64_t shader_id0, uint64_t shader_id1) {
    auto it = mShaderProgramPool.find(std::make_pair(shader_id0, shader_id1));

    if (it == mShaderProgramPool.end()) {
        return nullptr;
    }

    struct ShaderProgramMetal* prg = &it->second;

    if (prg->markedForDeletion) {
        mShaderProgramPool.erase(it);
        return nullptr;
    }

    return it == mShaderProgramPool.end() ? nullptr : (struct ShaderProgram*)&it->second;
}

void GfxRenderingAPIMetal::ShaderGetInfo(struct ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    struct ShaderProgramMetal* p = (struct ShaderProgramMetal*)prg;

    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

uint32_t GfxRenderingAPIMetal::NewTexture() {
    mTextures.resize(mTextures.size() + 1);
    return (uint32_t)(mTextures.size() - 1);
}

void GfxRenderingAPIMetal::DeleteTexture(uint32_t texID) {
}

void GfxRenderingAPIMetal::SelectTexture(int tile, uint32_t texture_id) {
    mCurrentTile = tile;
    mCurrentTextureIds[tile] = texture_id;
}

void GfxRenderingAPIMetal::UploadTexture(const uint8_t* rgba32_buf, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }

    TextureDataMetal* texture_data = &mTextures[mCurrentTextureIds[mCurrentTile]];

    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    MTL::TextureDescriptor* texture_descriptor =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, width, height, true);
    texture_descriptor->setArrayLength(1);
    texture_descriptor->setMipmapLevelCount(1);
    texture_descriptor->setSampleCount(1);
    texture_descriptor->setStorageMode(MTL::StorageModeShared);

    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);

    MTL::Texture* texture = texture_data->texture;
    if (texture_data->texture == nullptr || texture_data->texture->width() != width ||
        texture_data->texture->height() != height) {
        if (texture_data->texture != nullptr)
            texture_data->texture->release();

        texture = mDevice->newTexture(texture_descriptor);
    }

    NS::UInteger bytes_per_pixel = 4;
    NS::UInteger bytes_per_row = bytes_per_pixel * width;
    texture->replaceRegion(region, 0, rgba32_buf, bytes_per_row);
    texture_data->texture = texture;

    autorelease_pool->release();
}

void GfxRenderingAPIMetal::SetSamplerParameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    TextureDataMetal* texture_data = &mTextures[mCurrentTextureIds[tile]];
    texture_data->linear_filtering = linear_filter;
    texture_data->filtering = !linear_filter ? FILTER_LINEAR : FILTER_THREE_POINT;

    // This function is called twice per texture, the first one only to set default values.
    // Maybe that could be skipped? Anyway, make sure to release the first default sampler
    // state before setting the actual one.
    if (texture_data->sampler != nullptr) {
        texture_data->sampler->release();
    }

    MTL::SamplerDescriptor* sampler_descriptor = MTL::SamplerDescriptor::alloc()->init();
    MTL::SamplerMinMagFilter filter = linear_filter && mCurrentFilterMode == FILTER_LINEAR
                                          ? MTL::SamplerMinMagFilterLinear
                                          : MTL::SamplerMinMagFilterNearest;
    sampler_descriptor->setMinFilter(filter);
    sampler_descriptor->setMagFilter(filter);
    sampler_descriptor->setSAddressMode(gfx_cm_to_metal(cms));
    sampler_descriptor->setTAddressMode(gfx_cm_to_metal(cmt));
    sampler_descriptor->setRAddressMode(MTL::SamplerAddressModeRepeat);

    texture_data->sampler = mDevice->newSamplerState(sampler_descriptor);
    sampler_descriptor->release();
}

void GfxRenderingAPIMetal::SetDepthTestAndMask(bool depth_test, bool depth_mask) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = depth_mask;
}

void GfxRenderingAPIMetal::SetCurrentPrimDepth(float depth) {
    if (depth != mCurrentPrimDepth) {
        mCurrentPrimDepth = depth;
        mPrimDepthDirty = true;
    }
}

void GfxRenderingAPIMetal::SetZmodeDecal(bool zmode_decal) {
    mCurrentZmodeDecal = zmode_decal;
}

void GfxRenderingAPIMetal::SetViewport(int x, int y, int width, int height) {
    FramebufferMetal& fb = mFramebuffers[mCurrentFramebuffer];

    fb.mViewport->originX = x;
    fb.mViewport->originY = mRenderTargetHeight - y - height;
    fb.mViewport->width = width;
    fb.mViewport->height = height;
    fb.mViewport->znear = 0;
    fb.mViewport->zfar = 1;

    fb.mCommandEncoder->setViewport(*fb.mViewport);
}

void GfxRenderingAPIMetal::SetScissor(int x, int y, int width, int height) {
    FramebufferMetal& fb = mFramebuffers[mCurrentFramebuffer];
    TextureDataMetal tex = mTextures[fb.mTextureId];

    // clamp to viewport size as metal does not support larger values than viewport size
    fb.mScissorRect->x = std::max(0, std::min<int>(x, tex.width));
    fb.mScissorRect->y = std::max(0, std::min<int>(mRenderTargetHeight - y - height, tex.height));
    fb.mScissorRect->width = std::max(0, std::min<int>(width, tex.width));
    fb.mScissorRect->height = std::max(0, std::min<int>(height, tex.height));

    fb.mCommandEncoder->setScissorRect(*fb.mScissorRect);
}

void GfxRenderingAPIMetal::SetUseAlpha(bool use_alpha) {
    // Already part of the pipeline state from shader info
}

void GfxRenderingAPIMetal::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();
    bool textures_changed = false;

    auto& current_framebuffer = mFramebuffers[mCurrentFramebuffer];

    if (current_framebuffer.mLastDepthTest != mCurrentDepthTest ||
        current_framebuffer.mLastDepthMask != mCurrentDepthMask) {
        current_framebuffer.mLastDepthTest = mCurrentDepthTest;
        current_framebuffer.mLastDepthMask = mCurrentDepthMask;

        MTL::DepthStencilDescriptor* depth_descriptor = MTL::DepthStencilDescriptor::alloc()->init();
        depth_descriptor->setDepthWriteEnabled(mCurrentDepthMask);
        depth_descriptor->setDepthCompareFunction(
            mCurrentDepthTest ? (mCurrentZmodeDecal ? MTL::CompareFunctionLessEqual : MTL::CompareFunctionLess)
                              : MTL::CompareFunctionAlways);

        MTL::DepthStencilState* depth_stencil_state = mDevice->newDepthStencilState(depth_descriptor);
        current_framebuffer.mCommandEncoder->setDepthStencilState(depth_stencil_state);

        depth_descriptor->release();
    }

    if (current_framebuffer.mLastZmodeDecal != mCurrentZmodeDecal) {
        current_framebuffer.mLastZmodeDecal = mCurrentZmodeDecal;

        current_framebuffer.mCommandEncoder->setTriangleFillMode(MTL::TriangleFillModeFill);
        current_framebuffer.mCommandEncoder->setCullMode(MTL::CullModeNone);
        current_framebuffer.mCommandEncoder->setFrontFacingWinding(MTL::WindingCounterClockwise);

        // SSDB = SlopeScaledDepthBias 120 leads to -2 at 240p which is the same as N64 mode which has very little
        // fighting
        const int n64modeFactor = 120;
        const int noVanishFactor = 100;
        float SSDB = -2;
        switch (mConsoleVariable->GetInteger(CVAR_Z_FIGHTING_MODE, 0)) {
            case 1: // scaled z-fighting (N64 mode like)
                SSDB = -1.0f * (float)mRenderTargetHeight / n64modeFactor;
                break;
            case 2: // no vanishing paths
                SSDB = -1.0f * (float)mRenderTargetHeight / noVanishFactor;
                break;
            case 0: // disabled
            default:
                SSDB = -2;
        }
        current_framebuffer.mCommandEncoder->setDepthBias(0, mCurrentZmodeDecal ? SSDB : 0, 0);
    }

    MTL::Buffer* vertex_buffer = mVertexBufferPool[mCurrentVertexBufferPoolIndex];
    memcpy((char*)vertex_buffer->contents() + mCurrentVertexBufferOffset, buf_vbo, sizeof(float) * buf_vbo_len);

    if (!current_framebuffer.mHasBoundVertexShader) {
        current_framebuffer.mCommandEncoder->setVertexBuffer(vertex_buffer, 0, 0);
        current_framebuffer.mHasBoundVertexShader = true;
    }

    current_framebuffer.mCommandEncoder->setVertexBufferOffset(mCurrentVertexBufferOffset, 0);

    if (!current_framebuffer.mHasBoundFragShader) {
        current_framebuffer.mCommandEncoder->setFragmentBuffer(mFrameUniformBuffer, 0, 0);
        current_framebuffer.mHasBoundFragShader = true;
    }

    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        if (mShaderProgram->usedTextures[i]) {
            if (current_framebuffer.mLastBoundTextures[i] != mTextures[mCurrentTextureIds[i]].texture) {
                current_framebuffer.mLastBoundTextures[i] = mTextures[mCurrentTextureIds[i]].texture;
                current_framebuffer.mCommandEncoder->setFragmentTexture(mTextures[mCurrentTextureIds[i]].texture, i);

                if (current_framebuffer.mLastBoundSamplers[i] != mTextures[mCurrentTextureIds[i]].sampler) {
                    current_framebuffer.mLastBoundSamplers[i] = mTextures[mCurrentTextureIds[i]].sampler;
                    current_framebuffer.mCommandEncoder->setFragmentSamplerState(
                        mTextures[mCurrentTextureIds[i]].sampler, i);
                }
            }
        }

        if (mCurrentFilterMode == FILTER_THREE_POINT) {
            mDrawUniforms.textureFiltering[i] = mTextures[mCurrentTextureIds[i]].filtering;
            mDrawUniforms.textureFiltering[i] = mTextures[mCurrentTextureIds[i]].filtering;
            textures_changed = true;
        }
    }

    if (textures_changed || mPrimDepthDirty) {
        mDrawUniforms.prim_depth = mCurrentPrimDepth;
        current_framebuffer.mCommandEncoder->setFragmentBytes(&mDrawUniforms, sizeof(DrawUniforms), 1);
        mPrimDepthDirty = false;
    }

    if (current_framebuffer.mLastShaderProgram != mShaderProgram) {
        current_framebuffer.mLastShaderProgram = mShaderProgram;

        MTL::RenderPipelineState* pipeline_state = PipelineFor(mShaderProgram, current_framebuffer.mMsaaLevel);
        current_framebuffer.mCommandEncoder->setRenderPipelineState(pipeline_state);
    }

    current_framebuffer.mCommandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0.f, buf_vbo_num_tris * 3);
    mCurrentVertexBufferOffset += sizeof(float) * buf_vbo_len;

    autorelease_pool->release();
}

void GfxRenderingAPIMetal::OnResize() {
}

void GfxRenderingAPIMetal::StartFrame() {
    mFrameUniforms.frameCount++;
    if (mFrameUniforms.frameCount > 150) {
        // No high values, as noise starts to look ugly
        mFrameUniforms.frameCount = 0;
    }

    if (!mFrameUniformBuffer) {
        mFrameUniformBuffer = mDevice->newBuffer(sizeof(FrameUniforms), MTL::ResourceCPUCacheModeDefaultCache);
    }
    if (!mCoordUniformBuffer) {
        mCoordUniformBuffer = mDevice->newBuffer(sizeof(CoordUniforms), MTL::ResourceCPUCacheModeDefaultCache);
    }

    mCurrentVertexBufferOffset = 0;

    mFrameAutoreleasePool = NS::AutoreleasePool::alloc()->init();
}

void GfxRenderingAPIMetal::EndFrame() {
    std::set<int>::iterator it = mDrawnFramebuffers.begin();
    it++;

    while (it != mDrawnFramebuffers.end()) {
        auto& framebuffer = mFramebuffers[*it];

        if (!framebuffer.mHasEndedEncoding)
            framebuffer.mCommandEncoder->endEncoding();

        framebuffer.mCommandBuffer->commit();
        it++;
    }

    auto& screen_framebuffer = mFramebuffers[0];
    if (!screen_framebuffer.mHasEndedEncoding)
        screen_framebuffer.mCommandEncoder->endEncoding();

    // Deferred screen FB readback: blit the rendered texture into a shared
    // MTLBuffer using a blit encoder (cheap DMA — no render pass boundary,
    // no TBDR tile flush/reload).  The CPU reads this buffer next frame in
    // ReadFramebufferToCPU and converts BGRA8 → RGBA5551 on the CPU side.
    if (mScreenReadbackRequested && !gPortReplayFrame) {
        MTL::Texture* screenTex = mTextures[screen_framebuffer.mTextureId].texture;
        uint32_t w = mScreenReadbackWidth;
        uint32_t h = mScreenReadbackHeight;
        size_t bytesPerRow = w * 4; // BGRA8 = 4 bytes per pixel
        size_t bufSize = bytesPerRow * h;

        // Allocate / reuse a persistent shared buffer for CPU readback.
        if (!mScreenReadbackBuffer || mScreenReadbackBuffer->length() < bufSize) {
            if (mScreenReadbackBuffer)
                mScreenReadbackBuffer->release();
            mScreenReadbackBuffer = mDevice->newBuffer(bufSize, MTL::ResourceStorageModeShared);
        }

        MTL::BlitCommandEncoder* blit = screen_framebuffer.mCommandBuffer->blitCommandEncoder();
        blit->copyFromTexture(screenTex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(w, h, 1), mScreenReadbackBuffer, 0,
                              bytesPerRow, bufSize);
        blit->endEncoding();

        // Don't set mScreenReadbackDataReady yet — the blit hasn't executed.
        // Retain the command buffer so ReadFramebufferToCPU can wait on it.
        mScreenReadbackRequested = false;
    }

    screen_framebuffer.mCommandBuffer->presentDrawable(mCurrentDrawable);
    mCurrentVertexBufferPoolIndex = (mCurrentVertexBufferPoolIndex + 1) % kMaxVertexBufferPoolSize;
    screen_framebuffer.mCommandBuffer->commit();

    // Now that commit has been called, retain the command buffer for GPU sync
    // in ReadFramebufferToCPU next frame (waitUntilCompleted before reading).
    if (gPortReplayFrame) {
        // PORT: an in-between frame (frame interpolation): readback state left for the next real frame
    } else if (mScreenReadbackBuffer && !mScreenReadbackDataReady) {
        mScreenReadbackCmdBuf = screen_framebuffer.mCommandBuffer->retain();
        mScreenReadbackDataReady = true;
    } else if (mScreenReadbackDataReady && mScreenReadbackCmdBuf) {
        // Nobody consumed the readback this frame — data is going stale.
        // Release the retained command buffer to avoid leaking it, and reset
        // so the next ReadFramebufferToCPU requests a fresh blit instead of
        // serving outdated pixels.
        mScreenReadbackCmdBuf->release();
        mScreenReadbackCmdBuf = nullptr;
        mScreenReadbackDataReady = false;
    }

    mDrawnFramebuffers.clear();

    // Cleanup states
    for (int fb_id = 0; fb_id < (int)mFramebuffers.size(); fb_id++) {
        FramebufferMetal& fb = mFramebuffers[fb_id];

        fb.mLastShaderProgram = nullptr;
        fb.mCommandBuffer = nullptr;
        fb.mCommandEncoder = nullptr;
        fb.mHasEndedEncoding = false;
        fb.mHasBoundVertexShader = false;
        fb.mHasBoundFragShader = false;
        for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
            fb.mLastBoundTextures[i] = nullptr;
            fb.mLastBoundSamplers[i] = nullptr;
        }
        memset(fb.mViewport, 0, sizeof(MTL::Viewport));
        memset(fb.mScissorRect, 0, sizeof(MTL::ScissorRect));
        fb.mLastDepthTest = -1;
        fb.mLastDepthMask = -1;
        fb.mLastZmodeDecal = -1;
    }

    mFrameAutoreleasePool->release();
}

void GfxRenderingAPIMetal::FinishRender() {
}

int GfxRenderingAPIMetal::CreateFramebuffer() {
    uint32_t texture_id = NewTexture();
    TextureDataMetal& t = mTextures[texture_id];

    size_t index = mFramebuffers.size();
    mFramebuffers.resize(mFramebuffers.size() + 1);
    FramebufferMetal& data = mFramebuffers.back();
    data.mScissorRect = new MTL::ScissorRect();
    data.mViewport = new MTL::Viewport();
    data.mTextureId = texture_id;

    uint32_t tile = 0;
    uint32_t saved = mCurrentTextureIds[tile];
    mCurrentTextureIds[tile] = texture_id;
    SetSamplerParameters(0, true, G_TX_WRAP, G_TX_WRAP);
    mCurrentTextureIds[tile] = saved;

    return (int)index;
}

void GfxRenderingAPIMetal::SetupScreenFramebuffer(uint32_t width, uint32_t height) {
    mCurrentDrawable = nullptr;
    mCurrentDrawable = mLayer->nextDrawable();

    bool msaa_enabled = mConsoleVariable->GetInteger("gMSAAValue", 1) > 1;

    FramebufferMetal& fb = mFramebuffers[0];
    TextureDataMetal& tex = mTextures[fb.mTextureId];

    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    tex.texture = mCurrentDrawable->texture();

    MTL::RenderPassDescriptor* render_pass_descriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    render_pass_descriptor->colorAttachments()->object(0)->setTexture(tex.texture);
    render_pass_descriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
    render_pass_descriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    tex.width = width;
    tex.height = height;

    // recreate depth texture only if necessary (size changed)
    if (fb.mDepthTexture == nullptr || (fb.mDepthTexture->width() != width || fb.mDepthTexture->height() != height)) {
        if (fb.mDepthTexture != nullptr)
            fb.mDepthTexture->release();

        // If possible, we eventually we want to disable this when msaa is enabled since we don't need this depth
        // texture However, problem is if the user switches to msaa during game, we need a way to then generate it
        // before drawing.
        MTL::TextureDescriptor* depth_tex_desc =
            MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, width, height, true);

        depth_tex_desc->setTextureType(MTL::TextureType2D);
        depth_tex_desc->setStorageMode(MTL::StorageModePrivate);
        depth_tex_desc->setSampleCount(1);
        depth_tex_desc->setArrayLength(1);
        depth_tex_desc->setMipmapLevelCount(1);
        depth_tex_desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);

        fb.mDepthTexture = mDevice->newTexture(depth_tex_desc);
    }

    render_pass_descriptor->depthAttachment()->setTexture(fb.mDepthTexture);
    render_pass_descriptor->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
    render_pass_descriptor->depthAttachment()->setStoreAction(MTL::StoreActionStore);
    render_pass_descriptor->depthAttachment()->setClearDepth(1);

    if (fb.mRenderPassDescriptor != nullptr)
        fb.mRenderPassDescriptor->release();

    fb.mRenderPassDescriptor = render_pass_descriptor;
    fb.mRenderPassDescriptor->retain();
    fb.mRenderTarget = true;
    fb.mHasDepthBuffer = true;

    autorelease_pool->release();
}

// PORT: framebuffers whose colour texture was just (re)created: their contents are undefined (often leftover GPU
// memory). The game draws some frames incrementally (fades only add a translucent layer over the previous picture), so
// the first pass into such a texture clears it to black, as an untouched N64 frame buffer would read.
static std::unordered_set<int> sFreshColorFbs;

// PORT: the old picture drawn into a resized framebuffer's new texture (fast/backends/resize_keep.h)
static MTL::RenderPipelineState* ResizeCopyPipeline(MTL::Device* device, MTL::PixelFormat format, uint32_t samples) {
    static std::unordered_map<uint64_t, MTL::RenderPipelineState*> sPipelines;
    static MTL::Function* sVertex = nullptr;
    static MTL::Function* sFragment = nullptr;
    uint64_t key = ((uint64_t)format << 8) | samples;
    if (auto it = sPipelines.find(key); it != sPipelines.end()) {
        return it->second;
    }
    if (sVertex == nullptr) {
        static const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;
            struct CopyOut { float4 pos [[position]]; float2 uv; };
            vertex CopyOut resizeCopyVertex(uint vid [[vertex_id]], constant float4& rect [[buffer(0)]]) {
                float2 corner = float2(vid & 1, vid >> 1); // triangle strip over the destination rectangle
                CopyOut out;
                out.pos = float4(mix(rect.xy, rect.zw, corner), 0.0, 1.0);
                out.uv = corner;
                return out;
            }
            fragment float4 resizeCopyFragment(CopyOut in [[stage_in]], texture2d<float> src [[texture(0)]]) {
                constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
                return src.sample(linearSampler, in.uv);
            }
        )";
        NS::Error* error = nullptr;
        MTL::Library* library = device->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), nullptr, &error);
        if (library == nullptr) {
            SPDLOG_ERROR("Failed to compile the resize copy shader: {}",
                         error ? error->localizedDescription()->cString(NS::UTF8StringEncoding) : "?");
            return nullptr;
        }
        sVertex = library->newFunction(NS::String::string("resizeCopyVertex", NS::UTF8StringEncoding));
        sFragment = library->newFunction(NS::String::string("resizeCopyFragment", NS::UTF8StringEncoding));
        library->release();
    }
    MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(sVertex);
    desc->setFragmentFunction(sFragment);
    desc->colorAttachments()->object(0)->setPixelFormat(format);
    desc->setSampleCount(samples);
    NS::Error* error = nullptr;
    MTL::RenderPipelineState* pipeline = device->newRenderPipelineState(desc, &error);
    desc->release();
    if (pipeline == nullptr) {
        SPDLOG_ERROR("Failed to create the resize copy pipeline: {}",
                     error ? error->localizedDescription()->cString(NS::UTF8StringEncoding) : "?");
    }
    sPipelines[key] = pipeline;
    return pipeline;
}

static bool DrawOldPictureScaled(MTL::Device* device, MTL::CommandQueue* queue, MTL::Texture* oldTex,
                                 MTL::Texture* newTex, MTL::Texture* newMsaaTex, uint32_t msaa) {
    MTL::RenderPipelineState* pipeline = ResizeCopyPipeline(device, newTex->pixelFormat(), msaa > 1 ? msaa : 1);
    if (pipeline == nullptr) {
        return false;
    }
    const float nw = (float)newTex->width(), nh = (float)newTex->height();
    PortResizeKeepRect r = PortResizeKeepDest((float)oldTex->width(), (float)oldTex->height(), nw, nh);
    // destination rectangle in clip space (y up; texture rows run down)
    float rect[4] = { r.x0 / nw * 2.0f - 1.0f, 1.0f - r.y0 / nh * 2.0f, r.x1 / nw * 2.0f - 1.0f, 1.0f - r.y1 / nh * 2.0f };

    MTL::RenderPassDescriptor* pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    auto* color = pass->colorAttachments()->object(0);
    if (msaa > 1 && newMsaaTex != nullptr) {
        color->setTexture(newMsaaTex);
        color->setResolveTexture(newTex);
        color->setStoreAction(MTL::StoreActionStoreAndMultisampleResolve);
    } else {
        color->setTexture(newTex);
        color->setStoreAction(MTL::StoreActionStore);
    }
    color->setLoadAction(MTL::LoadActionClear);
    color->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));

    MTL::CommandBuffer* cb = queue->commandBuffer();
    MTL::RenderCommandEncoder* enc = cb->renderCommandEncoder(pass);
    enc->setRenderPipelineState(pipeline);
    enc->setVertexBytes(rect, sizeof(rect), 0);
    enc->setFragmentTexture(oldTex, 0);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
    enc->endEncoding();
    cb->commit();
    // resizes are rare: wait, so the old texture can go and no later pass (on either queue) runs before the copy
    cb->waitUntilCompleted();
    return true;
}

void GfxRenderingAPIMetal::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                                       bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                                       bool can_extract_depth) {
    // Screen framebuffer is handled separately on a frame by frame basis
    // see `SetupScreenFramebuffer`.
    if (fb_id == 0) {
        int width, height;
        SDL_GetCurrentRenderOutputSize(mRenderer, &width, &height);
        mLayer->setDrawableSize({ CGFloat(width), CGFloat(height) });

        return;
    }

    FramebufferMetal& fb = mFramebuffers[fb_id];
    TextureDataMetal& tex = mTextures[fb.mTextureId];

    width = std::max(width, 1U);
    height = std::max(height, 1U);
    while (msaa_level > 1 && mMsaaNumQualityLevels[msaa_level - 1] == 0) {
        --msaa_level;
    }

    bool diff = tex.width != width || tex.height != height || fb.mMsaaLevel != msaa_level;

    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    if (diff || (fb.mRenderPassDescriptor != nullptr) != render_target) {
        MTL::TextureDescriptor* tex_descriptor = MTL::TextureDescriptor::alloc()->init();
        tex_descriptor->setTextureType(MTL::TextureType2D);
        tex_descriptor->setWidth(width);
        tex_descriptor->setHeight(height);
        tex_descriptor->setSampleCount(1);
        tex_descriptor->setMipmapLevelCount(1);
        tex_descriptor->setPixelFormat(mSrgbMode ? MTL::PixelFormatBGRA8Unorm_sRGB : MTL::PixelFormatBGRA8Unorm);
        tex_descriptor->setUsage((render_target ? MTL::TextureUsageRenderTarget : 0) | MTL::TextureUsageShaderRead);

        // PORT: a rendered picture of another size is drawn into the new texture (DrawOldPictureScaled)
        MTL::Texture* old_texture = nullptr;
        if (render_target && fb.mRenderPassDescriptor != nullptr && tex.texture != nullptr && PortResizeKeepEnabled() &&
            (tex.width != width || tex.height != height) && tex.width > 1 && tex.height > 1 &&
            tex.texture->pixelFormat() == tex_descriptor->pixelFormat()) {
            old_texture = tex.texture;
        } else if (tex.texture != nullptr) {
            tex.texture->release();
        }

        tex.texture = mDevice->newTexture(tex_descriptor);

        if (msaa_level > 1) {
            tex_descriptor->setTextureType(MTL::TextureType2DMultisample);
            tex_descriptor->setSampleCount(msaa_level);
            tex_descriptor->setStorageMode(MTL::StorageModePrivate);
            tex_descriptor->setUsage(render_target ? MTL::TextureUsageRenderTarget : 0);

            if (tex.msaaTexture != nullptr)
                tex.msaaTexture->release();
            tex.msaaTexture = mDevice->newTexture(tex_descriptor);
        }

        bool kept = old_texture != nullptr && DrawOldPictureScaled(mDevice, mCommandQueue, old_texture, tex.texture,
                                                                   msaa_level > 1 ? tex.msaaTexture : nullptr,
                                                                   msaa_level);
        if (old_texture != nullptr) {
            old_texture->release();
        }
        if (render_target && !kept) {
            sFreshColorFbs.insert(fb_id);
        } else if (kept) {
            sFreshColorFbs.erase(fb_id);
        }

        if (render_target) {
            MTL::RenderPassDescriptor* render_pass_descriptor = MTL::RenderPassDescriptor::renderPassDescriptor();

            bool fb_msaa_enabled = (msaa_level > 1);
            bool game_msaa_enabled = mConsoleVariable->GetInteger("gMSAAValue", 1) > 1;

            if (fb_msaa_enabled) {
                render_pass_descriptor->colorAttachments()->object(0)->setTexture(tex.msaaTexture);
                render_pass_descriptor->colorAttachments()->object(0)->setResolveTexture(tex.texture);
                render_pass_descriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
                render_pass_descriptor->colorAttachments()->object(0)->setStoreAction(
                    MTL::StoreActionStoreAndMultisampleResolve);
            } else {
                render_pass_descriptor->colorAttachments()->object(0)->setTexture(tex.texture);
                render_pass_descriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
                render_pass_descriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
            }

            if (fb.mRenderPassDescriptor != nullptr)
                fb.mRenderPassDescriptor->release();

            fb.mRenderPassDescriptor = render_pass_descriptor;
            fb.mRenderPassDescriptor->retain();
        }

        tex.width = width;
        tex.height = height;

        tex_descriptor->release();
    }

    if (has_depth_buffer && (diff || !fb.mHasDepthBuffer || (fb.mDepthTexture != nullptr) != can_extract_depth)) {
        MTL::TextureDescriptor* depth_tex_desc =
            MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, width, height, true);
        depth_tex_desc->setTextureType(MTL::TextureType2D);
        depth_tex_desc->setStorageMode(MTL::StorageModePrivate);
        depth_tex_desc->setSampleCount(1);
        depth_tex_desc->setArrayLength(1);
        depth_tex_desc->setMipmapLevelCount(1);
        depth_tex_desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);

        if (fb.mDepthTexture != nullptr)
            fb.mDepthTexture->release();

        fb.mDepthTexture = mDevice->newTexture(depth_tex_desc);

        if (msaa_level > 1) {
            depth_tex_desc->setTextureType(MTL::TextureType2DMultisample);
            depth_tex_desc->setSampleCount(msaa_level);

            if (fb.mMsaaDepthTexture != nullptr)
                fb.mMsaaDepthTexture->release();

            fb.mMsaaDepthTexture = mDevice->newTexture(depth_tex_desc);
        }
    }

    if (has_depth_buffer && fb.mRenderPassDescriptor != nullptr) {
        if (msaa_level > 1) {
            fb.mRenderPassDescriptor->depthAttachment()->setTexture(fb.mMsaaDepthTexture);
            fb.mRenderPassDescriptor->depthAttachment()->setResolveTexture(fb.mDepthTexture);
            fb.mRenderPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
            fb.mRenderPassDescriptor->depthAttachment()->setStoreAction(MTL::StoreActionMultisampleResolve);
            fb.mRenderPassDescriptor->depthAttachment()->setClearDepth(1);
        } else {
            fb.mRenderPassDescriptor->depthAttachment()->setTexture(fb.mDepthTexture);
            fb.mRenderPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
            fb.mRenderPassDescriptor->depthAttachment()->setStoreAction(MTL::StoreActionStore);
            fb.mRenderPassDescriptor->depthAttachment()->setClearDepth(1);
        }
    } else if (fb.mRenderPassDescriptor != nullptr) {
        fb.mRenderPassDescriptor->setDepthAttachment(nullptr);
    }

    fb.mRenderTarget = render_target;
    fb.mHasDepthBuffer = has_depth_buffer;
    fb.mMsaaLevel = msaa_level;

    autorelease_pool->release();
}

void GfxRenderingAPIMetal::StartDrawToFramebuffer(int fb_id, float noise_scale) {
    FramebufferMetal& fb = mFramebuffers[fb_id];
    mRenderTargetHeight = mTextures[fb.mTextureId].height;

    mCurrentFramebuffer = fb_id;
    mDrawnFramebuffers.insert(fb_id);

    if (fb.mRenderTarget && fb.mCommandBuffer == nullptr && fb.mCommandEncoder == nullptr &&
        fb.mRenderPassDescriptor != nullptr) {
        // FBs that are read back to CPU use a separate queue so
        // commit+waitUntilCompleted doesn't deadlock on enqueue ordering.
        MTL::CommandQueue* queue = fb.mUseReadbackQueue ? mReadbackQueue : mCommandQueue;
        fb.mCommandBuffer = queue->commandBuffer();
        std::string fbcb_label = fmt::format("FrameBuffer {} Command Buffer", fb_id);
        fb.mCommandBuffer->setLabel(NS::String::string(fbcb_label.c_str(), NS::UTF8StringEncoding));

        if (!fb.mUseReadbackQueue) {
            // Queue the command buffers in order of start draw
            fb.mCommandBuffer->enqueue();
        }

        MTL::RenderPassColorAttachmentDescriptor* color0 = fb.mRenderPassDescriptor->colorAttachments()->object(0);
        static const bool sNoFreshClear = getenv("PORT_NO_FRESH_CLEAR") != nullptr; // debug: old behaviour
        bool fresh = sFreshColorFbs.erase(fb_id) != 0 && !sNoFreshClear;
        MTL::LoadAction loadAction = color0->loadAction();
        if (fresh) {
            color0->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));
            color0->setLoadAction(MTL::LoadActionClear);
        }
        fb.mCommandEncoder = fb.mCommandBuffer->renderCommandEncoder(fb.mRenderPassDescriptor);
        if (fresh) {
            color0->setLoadAction(loadAction);
        }
        std::string fbce_label = fmt::format("FrameBuffer {} Command Encoder", fb_id);
        fb.mCommandEncoder->setLabel(NS::String::string(fbce_label.c_str(), NS::UTF8StringEncoding));
        fb.mCommandEncoder->setDepthClipMode(MTL::DepthClipModeClamp);
    }

    if (noise_scale != 0.0f) {
        mFrameUniforms.noiseScale = 1.0f / noise_scale;
    }

    memcpy(mFrameUniformBuffer->contents(), &mFrameUniforms, sizeof(FrameUniforms));
}

void GfxRenderingAPIMetal::ClearFramebuffer(bool color, bool depth) {
    if (!color && !depth) {
        return;
    }

    auto& framebuffer = mFramebuffers[mCurrentFramebuffer];

    // End the current render encoder
    framebuffer.mCommandEncoder->endEncoding();

    // Track the original load action and set the next load actions to Load to leverage the blit results
    MTL::RenderPassColorAttachmentDescriptor* srcColorAttachment =
        framebuffer.mRenderPassDescriptor->colorAttachments()->object(0);
    MTL::LoadAction origLoadAction = srcColorAttachment->loadAction();
    if (color) {
        srcColorAttachment->setLoadAction(MTL::LoadActionClear);
    }

    MTL::RenderPassDepthAttachmentDescriptor* srcDepthAttachment = framebuffer.mRenderPassDescriptor->depthAttachment();
    MTL::LoadAction origDepthLoadAction = MTL::LoadActionDontCare;
    if (depth && framebuffer.mHasDepthBuffer) {
        origDepthLoadAction = srcDepthAttachment->loadAction();
        srcDepthAttachment->setLoadAction(MTL::LoadActionClear);
    }

    // Create a new render encoder back onto the framebuffer
    framebuffer.mCommandEncoder = framebuffer.mCommandBuffer->renderCommandEncoder(framebuffer.mRenderPassDescriptor);

    std::string fbce_label = fmt::format("FrameBuffer {} Command Encoder After Clear", mCurrentFramebuffer);
    framebuffer.mCommandEncoder->setLabel(NS::String::string(fbce_label.c_str(), NS::UTF8StringEncoding));
    framebuffer.mCommandEncoder->setDepthClipMode(MTL::DepthClipModeClamp);
    framebuffer.mCommandEncoder->setViewport(*framebuffer.mViewport);
    framebuffer.mCommandEncoder->setScissorRect(*framebuffer.mScissorRect);

    // Now that the command encoder is started, we set the original load actions back for the next frame's use
    srcColorAttachment->setLoadAction(origLoadAction);
    if (depth && framebuffer.mHasDepthBuffer) {
        srcDepthAttachment->setLoadAction(origDepthLoadAction);
    }

    // Reset the framebuffer so the encoder is setup again when rendering triangles
    framebuffer.mHasBoundVertexShader = false;
    framebuffer.mHasBoundFragShader = false;
    framebuffer.mLastShaderProgram = nullptr;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        framebuffer.mLastBoundTextures[i] = nullptr;
        framebuffer.mLastBoundSamplers[i] = nullptr;
    }
    framebuffer.mLastDepthTest = -1;
    framebuffer.mLastDepthMask = -1;
    framebuffer.mLastZmodeDecal = -1;
}

bool GfxRenderingAPIMetal::ClearFramebufferColor(float r, float g, float b, float a) {
    auto& framebuffer = mFramebuffers[mCurrentFramebuffer];
    if (mCurrentFramebuffer == 0 || framebuffer.mCommandEncoder == nullptr || framebuffer.mRenderPassDescriptor == nullptr) {
        return false;
    }
    MTL::RenderPassColorAttachmentDescriptor* color0 = framebuffer.mRenderPassDescriptor->colorAttachments()->object(0);
    MTL::ClearColor saved = color0->clearColor();
    color0->setClearColor(MTL::ClearColor::Make(r, g, b, a));
    ClearFramebuffer(true, false);
    color0->setClearColor(saved);
    return true;
}

void GfxRenderingAPIMetal::ResolveMSAAColorBuffer(int fb_id_target, int fb_id_source) {
    int source_texture_id = mFramebuffers[fb_id_source].mTextureId;
    MTL::Texture* source_texture = mTextures[source_texture_id].texture;

    int target_texture_id = mFramebuffers[fb_id_target].mTextureId;
    MTL::Texture* target_texture =
        target_texture_id == 0 ? mCurrentDrawable->texture() : mTextures[target_texture_id].texture;

    // Workaround for detecting when transitioning to/from full screen mode.
    if (source_texture->width() != target_texture->width() || source_texture->height() != target_texture->height()) {
        return;
    }

    // When the target buffer is our main window buffer, we need to perform the blit operation on the target
    // buffer instead of the source buffer
    if (fb_id_target != 0) {
        // Copy over the source framebuffer's texture to the target
        auto& source_framebuffer = mFramebuffers[fb_id_source];
        source_framebuffer.mCommandEncoder->endEncoding();
        source_framebuffer.mHasEndedEncoding = true;

        MTL::BlitCommandEncoder* blit_encoder = source_framebuffer.mCommandBuffer->blitCommandEncoder();
        blit_encoder->setLabel(NS::String::string("MSAA Copy Encoder", NS::UTF8StringEncoding));
        blit_encoder->copyFromTexture(source_texture, target_texture);
        blit_encoder->endEncoding();
    } else {
        // End the current render encoder
        auto& target_framebuffer = mFramebuffers[fb_id_target];
        target_framebuffer.mCommandEncoder->endEncoding();

        // Create a blit encoder
        MTL::BlitCommandEncoder* blit_encoder = target_framebuffer.mCommandBuffer->blitCommandEncoder();
        blit_encoder->setLabel(NS::String::string("MSAA Copy Encoder", NS::UTF8StringEncoding));

        // Copy the texture over using the origins and size
        blit_encoder->copyFromTexture(source_texture, target_texture);
        blit_encoder->endEncoding();

        // Update the load action to Load to leverage the blit results
        // The original load action will be set back on the next frame by SetupScreenFramebuffer
        MTL::RenderPassColorAttachmentDescriptor* targetColorAttachment =
            target_framebuffer.mRenderPassDescriptor->colorAttachments()->object(0);
        targetColorAttachment->setLoadAction(MTL::LoadActionLoad);

        // Create a new render encoder back onto the framebuffer
        target_framebuffer.mCommandEncoder =
            target_framebuffer.mCommandBuffer->renderCommandEncoder(target_framebuffer.mRenderPassDescriptor);

        std::string fbce_label = fmt::format("FrameBuffer {} Command Encoder After MSAA Resolve", fb_id_target);
        target_framebuffer.mCommandEncoder->setLabel(NS::String::string(fbce_label.c_str(), NS::UTF8StringEncoding));
    }
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIMetal::GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    auto framebuffer = mFramebuffers[fb_id];

    if (coordinates.size() > mCoordBufferSize) {
        if (mDepthValueOutputBuffer != nullptr)
            mDepthValueOutputBuffer->release();

        mDepthValueOutputBuffer =
            mDevice->newBuffer(sizeof(float) * coordinates.size(), MTL::ResourceOptionCPUCacheModeDefault);
        mDepthValueOutputBuffer->setLabel(NS::String::string("Depth output buffer", NS::UTF8StringEncoding));

        mCoordBufferSize = coordinates.size();
    }

    // zero out the buffer
    memset(mCoordUniformBuffer->contents(), 0, sizeof(CoordUniforms));
    memset(mDepthValueOutputBuffer->contents(), 0, sizeof(float) * coordinates.size());

    // map coordinates to right y axis
    size_t i = 0;
    for (const auto& coord : coordinates) {
        mCoordUniforms.coords[i].x = (uint32_t)(int32_t)coord.first;
        mCoordUniforms.coords[i].y = (uint32_t)(int32_t)(framebuffer.mDepthTexture->height() - 1 - coord.second);
        ++i;
    }

    // set uniform values
    memcpy(mCoordUniformBuffer->contents(), &mCoordUniforms, sizeof(CoordUniforms));

    NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

    auto command_buffer = mCommandQueue->commandBuffer();
    command_buffer->setLabel(NS::String::string("Depth Shader Command Buffer", NS::UTF8StringEncoding));

    NS::Error* error = nullptr;
    MTL::ComputePipelineState* compute_pipeline_state = mDevice->newComputePipelineState(mDepthComputeFunction, &error);

    MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
    compute_encoder->setComputePipelineState(compute_pipeline_state);
    compute_encoder->setTexture(framebuffer.mDepthTexture, 0);
    compute_encoder->setBuffer(mCoordUniformBuffer, 0, 0);
    compute_encoder->setBuffer(mDepthValueOutputBuffer, 0, 1);

    MTL::Size thread_group_size = MTL::Size::Make(1, 1, 1);
    MTL::Size thread_group_count = MTL::Size::Make(coordinates.size(), 1, 1);

    // We validate if the device supports non-uniform threadgroup sizes
    if (mNonUniformThreadgroupSupported) {
        compute_encoder->dispatchThreads(thread_group_count, thread_group_size);
    } else {
        compute_encoder->dispatchThreadgroups(thread_group_count, thread_group_size);
    }

    compute_encoder->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    // Now the depth values can be accessed in the buffer.
    float* depth_values = (float*)mDepthValueOutputBuffer->contents();

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    {
        size_t i = 0;
        for (const auto& coord : coordinates) {
            res.emplace(coord, depth_values[i++] * 65532.0f);
        }
    }

    compute_pipeline_state->release();
    autorelease_pool->release();

    return res;
}

void* GfxRenderingAPIMetal::GetFramebufferTextureId(int fb_id) {
    return (void*)mTextures[mFramebuffers[fb_id].mTextureId].texture;
}

void GfxRenderingAPIMetal::SelectTextureFb(int fb_id) {
    int tile = 0;
    SelectTexture(tile, mFramebuffers[fb_id].mTextureId);
}

// PORT: the copy goes into streamFb's command buffer, between what was drawn into it so far and what comes next
void GfxRenderingAPIMetal::CopyFramebufferInStream(int streamFb, int dstFb, int srcFb, int w, int h) {
    if (streamFb >= (int)mFramebuffers.size() || dstFb >= (int)mFramebuffers.size() ||
        srcFb >= (int)mFramebuffers.size()) {
        return;
    }
    FramebufferMetal& stream = mFramebuffers[streamFb];
    if (stream.mCommandBuffer == nullptr || stream.mCommandEncoder == nullptr) {
        StartDrawToFramebuffer(streamFb, 0.0f);
    }
    if (stream.mCommandBuffer == nullptr || stream.mCommandEncoder == nullptr) {
        return;
    }
    const TextureDataMetal& srcTex = mTextures[mFramebuffers[srcFb].mTextureId];
    const TextureDataMetal& dstTex = mTextures[mFramebuffers[dstFb].mTextureId];
    MTL::Texture* src = srcTex.texture;
    MTL::Texture* dst = dstTex.texture;
    if (src == nullptr || dst == nullptr) {
        return;
    }
    const NS::UInteger cw = std::min<NS::UInteger>({ (NS::UInteger)w, src->width(), dst->width() });
    const NS::UInteger ch = std::min<NS::UInteger>({ (NS::UInteger)h, src->height(), dst->height() });
    // with MSAA the picture lives in the multisampled texture (loaded by every pass, resolved into the plain one): copied
    // whole when both framebuffers have one of the same size and sample count
    MTL::Texture* srcMs = mFramebuffers[srcFb].mMsaaLevel > 1 ? srcTex.msaaTexture : nullptr;
    MTL::Texture* dstMs = mFramebuffers[dstFb].mMsaaLevel > 1 ? dstTex.msaaTexture : nullptr;
    const bool copyMs = srcMs != nullptr && dstMs != nullptr && srcMs->sampleCount() == dstMs->sampleCount() &&
                        srcMs->width() == dstMs->width() && srcMs->height() == dstMs->height();
    stream.mCommandEncoder->endEncoding();
    MTL::BlitCommandEncoder* blit = stream.mCommandBuffer->blitCommandEncoder();
    blit->copyFromTexture(src, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(cw, ch, 1), dst, 0, 0, MTL::Origin(0, 0, 0));
    if (copyMs) {
        blit->copyFromTexture(srcMs, dstMs);
    }
    blit->endEncoding();
    // a new render encoder that keeps what the framebuffer holds (as CopyFramebuffer)
    MTL::RenderPassColorAttachmentDescriptor* color0 = stream.mRenderPassDescriptor->colorAttachments()->object(0);
    MTL::LoadAction colorLoad = color0->loadAction();
    color0->setLoadAction(MTL::LoadActionLoad);
    MTL::RenderPassDepthAttachmentDescriptor* depth = stream.mRenderPassDescriptor->depthAttachment();
    MTL::LoadAction depthLoad = MTL::LoadActionDontCare;
    if (stream.mHasDepthBuffer) {
        depthLoad = depth->loadAction();
        depth->setLoadAction(MTL::LoadActionLoad);
    }
    stream.mCommandEncoder = stream.mCommandBuffer->renderCommandEncoder(stream.mRenderPassDescriptor);
    stream.mCommandEncoder->setDepthClipMode(MTL::DepthClipModeClamp);
    stream.mCommandEncoder->setViewport(*stream.mViewport);
    stream.mCommandEncoder->setScissorRect(*stream.mScissorRect);
    color0->setLoadAction(colorLoad);
    if (stream.mHasDepthBuffer) {
        depth->setLoadAction(depthLoad);
    }
    stream.mHasBoundVertexShader = false;
    stream.mHasBoundFragShader = false;
    stream.mLastShaderProgram = nullptr;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        stream.mLastBoundTextures[i] = nullptr;
        stream.mLastBoundSamplers[i] = nullptr;
    }
}

void GfxRenderingAPIMetal::CopyFramebuffer(int fb_dst_id, int fb_src_id, int srcX0, int srcY0, int srcX1, int srcY1,
                                           int dstX0, int dstY0, int dstX1, int dstY1) {
    if (fb_src_id >= (int)mFramebuffers.size() || fb_dst_id >= (int)mFramebuffers.size()) {
        return;
    }

    FramebufferMetal& source_framebuffer = mFramebuffers[fb_src_id];

    int source_texture_id = source_framebuffer.mTextureId;
    MTL::Texture* source_texture = mTextures[source_texture_id].texture;

    int target_texture_id = mFramebuffers[fb_dst_id].mTextureId;
    MTL::Texture* target_texture = mTextures[target_texture_id].texture;

    // End the current render encoder
    source_framebuffer.mCommandEncoder->endEncoding();

    // Create a blit encoder
    MTL::BlitCommandEncoder* blit_encoder = source_framebuffer.mCommandBuffer->blitCommandEncoder();
    blit_encoder->setLabel(NS::String::string("Copy Framebuffer Encoder", NS::UTF8StringEncoding));

    MTL::Origin source_origin = MTL::Origin(srcX0, srcY0, 0);
    MTL::Origin target_origin = MTL::Origin(dstX0, dstY0, 0);
    MTL::Size source_size = MTL::Size(srcX1 - srcX0, srcY1 - srcY0, 1);

    // Copy the texture over using the origins and size
    blit_encoder->copyFromTexture(source_texture, 0, 0, source_origin, source_size, target_texture, 0, 0,
                                  target_origin);
    blit_encoder->endEncoding();

    // Track the original load action and set the next load actions to Load to leverage the blit results
    MTL::RenderPassColorAttachmentDescriptor* srcColorAttachment =
        source_framebuffer.mRenderPassDescriptor->colorAttachments()->object(0);
    MTL::LoadAction origLoadAction = srcColorAttachment->loadAction();
    srcColorAttachment->setLoadAction(MTL::LoadActionLoad);

    MTL::RenderPassDepthAttachmentDescriptor* srcDepthAttachment =
        source_framebuffer.mRenderPassDescriptor->depthAttachment();
    MTL::LoadAction origDepthLoadAction = MTL::LoadActionDontCare;
    if (source_framebuffer.mHasDepthBuffer) {
        origDepthLoadAction = srcDepthAttachment->loadAction();
        srcDepthAttachment->setLoadAction(MTL::LoadActionLoad);
    }

    // Create a new render encoder back onto the framebuffer
    source_framebuffer.mCommandEncoder =
        source_framebuffer.mCommandBuffer->renderCommandEncoder(source_framebuffer.mRenderPassDescriptor);

    std::string fbce_label = fmt::format("FrameBuffer {} Command Encoder After Copy", fb_src_id);
    source_framebuffer.mCommandEncoder->setLabel(NS::String::string(fbce_label.c_str(), NS::UTF8StringEncoding));
    source_framebuffer.mCommandEncoder->setDepthClipMode(MTL::DepthClipModeClamp);
    source_framebuffer.mCommandEncoder->setViewport(*source_framebuffer.mViewport);
    source_framebuffer.mCommandEncoder->setScissorRect(*source_framebuffer.mScissorRect);

    // Now that the command encoder is started, we set the original load actions back for the next frame's use
    srcColorAttachment->setLoadAction(origLoadAction);
    if (source_framebuffer.mHasDepthBuffer) {
        srcDepthAttachment->setLoadAction(origDepthLoadAction);
    }

    // Reset the framebuffer so the encoder is setup again when rendering triangles
    source_framebuffer.mHasBoundVertexShader = false;
    source_framebuffer.mHasBoundFragShader = false;
    source_framebuffer.mLastShaderProgram = nullptr;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        source_framebuffer.mLastBoundTextures[i] = nullptr;
        source_framebuffer.mLastBoundSamplers[i] = nullptr;
    }
    source_framebuffer.mLastDepthTest = -1;
    source_framebuffer.mLastDepthMask = -1;
    source_framebuffer.mLastZmodeDecal = -1;
}

extern std::vector<uint8_t> gPortScreenBgra; // interpreter.cpp
extern uint32_t gPortScreenW, gPortScreenH;

void GfxRenderingAPIMetal::InitializeFramebuffer(int fbId, uint32_t width, uint32_t height, const uint8_t* rgba) {
    auto& fb = mFramebuffers.at(fbId);
    auto* texture = mTextures[fb.mTextureId].texture;
    const size_t rowBytes = ((size_t)width * 4 + 255) & ~size_t(255);
    auto* staging = mDevice->newBuffer(rowBytes * height, MTL::ResourceStorageModeShared);
    auto* bytes = static_cast<uint8_t*>(staging->contents());
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto* src = rgba + ((size_t)y * width + x) * 4;
            auto* dst = bytes + y * rowBytes + x * 4;
            dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = src[3];
        }
    }
    auto* command = mReadbackQueue->commandBuffer();
    auto* blit = command->blitCommandEncoder();
    blit->copyFromBuffer(staging, 0, rowBytes, rowBytes * height, MTL::Size(width, height, 1),
                         texture, 0, 0, MTL::Origin(0, 0, 0));
    blit->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    staging->release();
    sFreshColorFbs.erase(fbId);
    fb.mRenderPassDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
}

void GfxRenderingAPIMetal::ReadFramebufferToCPU(int fb_id, uint32_t width, uint32_t height, uint16_t* rgba16_buf) {
    if (fb_id >= (int)mFramebuffers.size()) {
        return;
    }

    FramebufferMetal& framebuffer = mFramebuffers[fb_id];
    MTL::Texture* texture = mTextures[framebuffer.mTextureId].texture;

    if (fb_id == 0) {
        // Screen Framebuffer: deferred blit (zero encoder overhead)
        // Return previous frame's data.  Wait on the retained command buffer
        // to guarantee the GPU blit has finished before reading the shared buffer.
        if (mScreenReadbackDataReady && mScreenReadbackBuffer && mScreenReadbackCmdBuf) {
            mScreenReadbackCmdBuf->waitUntilCompleted();
            mScreenReadbackCmdBuf->release();
            mScreenReadbackCmdBuf = nullptr;
            mScreenReadbackDataReady = false;

            uint8_t* bgra = (uint8_t*)mScreenReadbackBuffer->contents();
            uint32_t count = mScreenReadbackWidth * mScreenReadbackHeight;
            // PORT: keep the full 8-bit pixels too, for screenshots (the RGBA5551 below quantises gradients)
            gPortScreenBgra.assign(bgra, bgra + (size_t)count * 4);
            gPortScreenW = mScreenReadbackWidth;
            gPortScreenH = mScreenReadbackHeight;
            for (uint32_t i = 0; i < count; i++) {
                uint8_t b = bgra[i * 4 + 0];
                uint8_t g = bgra[i * 4 + 1];
                uint8_t r = bgra[i * 4 + 2];
                uint8_t a = bgra[i * 4 + 3];
                rgba16_buf[i] = ((r >> 3) << 11) | ((g >> 3) << 6) | ((b >> 3) << 1) | (a >> 7);
            }
        }

        // Request a blit in EndFrame and we don't touch any encoders here
        mScreenReadbackRequested = true;
        mScreenReadbackWidth = width;
        mScreenReadbackHeight = height;
    } else {
        // Custom Framebuffer: synchronous readback via readback queue
        if (framebuffer.mCommandBuffer == nullptr) { // PORT: nothing drawn into it this frame: leave the buffer as is
            return;
        }
        if (framebuffer.mCommandEncoder && !framebuffer.mHasEndedEncoding) {
            framebuffer.mCommandEncoder->endEncoding();
        }

        if (!mConvertToRgb5a1PipelineState) {
            NS::Error* pso_error = nullptr;
            mConvertToRgb5a1PipelineState = mDevice->newComputePipelineState(mConvertToRgb5a1Function, &pso_error);
        }

        MTL::Buffer* output_buffer =
            mDevice->newBuffer(sizeof(uint16_t) * width * height, MTL::ResourceStorageModeShared);

        MTL::ComputeCommandEncoder* compute_encoder = framebuffer.mCommandBuffer->computeCommandEncoder();
        compute_encoder->setComputePipelineState(mConvertToRgb5a1PipelineState);
        compute_encoder->setTexture(texture, 0);
        compute_encoder->setBuffer(output_buffer, 0, 0);

        MTL::Size tg = MTL::Size::Make(8, 8, 1);
        MTL::Size total = MTL::Size::Make(width, height, 1);
        if (mNonUniformThreadgroupSupported) {
            compute_encoder->dispatchThreads(total, tg);
        } else {
            compute_encoder->dispatchThreadgroups(MTL::Size::Make((width + 7) / 8, (height + 7) / 8, 1), tg);
        }
        compute_encoder->endEncoding();

        framebuffer.mCommandBuffer->commit();

        if (framebuffer.mUseReadbackQueue) {
            framebuffer.mCommandBuffer->waitUntilCompleted();
            uint16_t* values = (uint16_t*)output_buffer->contents();
            memcpy(rgba16_buf, values, sizeof(uint16_t) * width * height);
        } else {
            // First frame on main queue — can't block without deadlock.
            // Return zeros so the caller gets black instead of garbage.
            memset(rgba16_buf, 0, sizeof(uint16_t) * width * height);
            framebuffer.mUseReadbackQueue = true;
        }

        output_buffer->release();

        // Reset state so StartDrawToFramebuffer can create a fresh command buffer.
        // Example case: the FB can be reused for another sprite in the same frame.
        framebuffer.mCommandBuffer = nullptr;
        framebuffer.mCommandEncoder = nullptr;
        framebuffer.mHasEndedEncoding = false;
        mDrawnFramebuffers.erase(fb_id);
    }
}

// PORT: a render-to-texture target is read back whenever the game stops drawing into it. Its passes go to the readback
// queue from the start: the first pass on the main queue would only run after the frame's own command buffers (enqueued
// earlier, committed at the end of the frame), so a later pass on the readback queue (drawn and read back right away)
// ran before it, and the first pass's clear then wiped what the later one drew (the Pokémon pictures on the VS screen).
void GfxRenderingAPIMetal::SetFramebufferReadback(int fb_id) {
    if (fb_id > 0 && fb_id < (int)mFramebuffers.size() && mFramebuffers[fb_id].mCommandBuffer == nullptr) {
        mFramebuffers[fb_id].mUseReadbackQueue = true;
    }
}

void GfxRenderingAPIMetal::SetTextureFilter(FilteringMode mode) {
    mCurrentFilterMode = mode;
    gfx_texture_cache_clear();
}

FilteringMode GfxRenderingAPIMetal::GetTextureFilter() {
    return mCurrentFilterMode;
}

ImTextureID GfxRenderingAPIMetal::GetTextureById(int fb_id) {
    return (void*)mTextures[fb_id].texture;
}

void GfxRenderingAPIMetal::SetSrgbMode() {
    mSrgbMode = true;
}
} // namespace Fast

bool Metal_IsSupported() {
#ifdef __IOS__
    // iOS always supports Metal and MTLCopyAllDevices is not available
    return true;
#else
    NS::Array* devices = MTLCopyAllDevices();
    NS::UInteger count = devices->count();

    devices->release();

    return count > 0;
#endif
}

#endif
