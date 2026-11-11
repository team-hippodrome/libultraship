//
//  gfx_vulkan.cpp
//  libultraship
//
//  PORT: Vulkan rendering backend for fast3d, adapted from Kenix3/libultraship#1265 (KiritoDv) to this tree's
//  rendering API. Architecturally it mirrors the Metal backend: per-framebuffer command buffers recorded during the
//  frame and submitted in framebuffer order at EndFrame, with the swapchain (fb 0) last.
//
//  Simplifications kept from #1265, chosen for correctness and portability (MoltenVK included):
//   - every image lives in VK_IMAGE_LAYOUT_GENERAL (except the swapchain image, which transitions to PRESENT_SRC at
//     the end of the screen command buffer),
//   - render passes always LOAD/STORE; clears go through vkCmdClearAttachments,
//   - pipelines are created lazily per (shader, depth/msaa state) combination.
//
//  Shaders: shaders/vulkan/default.shader.glsl (prism template, combiner as default.shader.metal), compiled to SPIR-V
//  with glslang when a combiner is first used.
//
#ifdef ENABLE_VULKAN

#include "fast/backends/gfx_vulkan.h"
#include "fast/backends/resize_keep.h"

#include <vector>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <chrono>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>
#include <prism/processor.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include <imgui_impl_vulkan.h>

#include "libultraship/libultra/gbi.h"
#include "ship/config/ConsoleVariable.h"
#include "ship/resource/ResourceManager.h"
#include "ship/resource/archive/ArchiveManager.h"
#include "ship/resource/factory/ShaderFactory.h"
#include "ship/core/Context.h"

// PORT: errors that leave the device unusable (out of host or device memory, device lost) end the render loop on the
// window thread (Fast::VulkanFatalError: the port selects OpenGL for the next start and exits). Logging and going on
// passed a null command buffer to the next call: an Intel HD driver (igvk64.dll) crashed in vkBeginCommandBuffer
// after vkAllocateCommandBuffers had failed with VK_ERROR_OUT_OF_HOST_MEMORY. Other threads (pipeline prebuild
// workers) only log: an exception there would end in std::terminate.
static std::thread::id sVkRenderThread;
static void VkCheckFailed(VkResult res, const char* expr, int line) {
    SPDLOG_ERROR("Vulkan error {} at {}: {}", (int)res, line, expr);
    const bool fatal = res == VK_ERROR_OUT_OF_HOST_MEMORY || res == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                       res == VK_ERROR_DEVICE_LOST || res == VK_ERROR_INITIALIZATION_FAILED;
    if (fatal && std::this_thread::get_id() == sVkRenderThread) {
        // for the dialog: the Vulkan call's name and what the error means
        std::string call(expr);
        const size_t at = call.find("vk");
        size_t end = at;
        while (end < call.size() && (isalnum((unsigned char)call[end]) || call[end] == '_')) end++;
        call = at != std::string::npos ? call.substr(at, end - at) : "A Vulkan call";
        const char* what = res == VK_ERROR_OUT_OF_HOST_MEMORY     ? "the graphics driver ran out of memory (VK_ERROR_OUT_OF_HOST_MEMORY)"
                           : res == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "the graphics chip ran out of memory (VK_ERROR_OUT_OF_DEVICE_MEMORY)"
                           : res == VK_ERROR_DEVICE_LOST          ? "the graphics driver stopped responding (VK_ERROR_DEVICE_LOST)"
                                                                  : "the graphics driver failed (VK_ERROR_INITIALIZATION_FAILED)";
        throw Fast::VulkanFatalError(call + " failed: " + what + ".");
    }
}
#define VK_CHECK(expr)                            \
    do {                                          \
        VkResult _res = (expr);                   \
        if (_res != VK_SUCCESS) {                 \
            VkCheckFailed(_res, #expr, __LINE__); \
        }                                         \
    } while (0)

namespace Fast {

extern std::vector<uint8_t> gPortScreenBgra; // interpreter.cpp
extern uint32_t gPortScreenW, gPortScreenH;
extern bool gPortReplayFrame; // interpreter.cpp (frame interpolation)

static std::shared_ptr<Ship::ResourceManager> sVKResourceManager;

// debug: PORT_VK_STATS=1 prints where the backend spent its time, every 60 drawn frames
static const bool sStats = getenv("PORT_VK_STATS") != nullptr;
static struct {
    int programs = 0, prebuiltPrograms = 0, pipelines = 0, syncSubmits = 0, commandBuffers = 0, commandBuffersAllocated = 0;
    double programMs = 0, pipelineMs = 0, syncMs = 0, fenceMs = 0, acquireMs = 0;
} sStat;
static double MsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

GfxRenderingAPIVK::GfxRenderingAPIVK(std::shared_ptr<Ship::ConsoleVariable> consoleVariable,
                                     std::shared_ptr<Ship::ResourceManager> resourceManager)
    : mConsoleVariable(std::move(consoleVariable)) {
    sVKResourceManager = std::move(resourceManager);
}

// MARK: - Shader compilation (GLSL -> SPIR-V via glslang)

// Returns an empty vector (and logs the error with the numbered source) when the source does not compile. Thread-safe
// (glslang_initialize_process ran in VulkanInit).
static std::vector<uint32_t> CompileGlslToSpirv(const ShaderDiskCache& cache, const std::string& source, bool vertex) {
    glslang_version_t version{};
    glslang_get_version(&version);
    const std::string compiler = std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
                                 std::to_string(version.patch) + (version.flavor ? version.flavor : "");
    const uint64_t key = ShaderDiskCache::Hash({source, vertex ? "vertex" : "fragment", compiler,
                                               "vulkan1.1-spv1.3-default-v1"});
    const auto cached = cache.Load(key);
    if (cached.size() >= 5 * sizeof(uint32_t) && cached.size() % sizeof(uint32_t) == 0) {
        std::vector<uint32_t> code(cached.size() / sizeof(uint32_t));
        memcpy(code.data(), cached.data(), cached.size());
        if (code[0] == 0x07230203 && code[3] != 0 && code[4] == 0) return code;
    }
    glslang_input_t input = {};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = vertex ? GLSLANG_STAGE_VERTEX : GLSLANG_STAGE_FRAGMENT;
    input.client = GLSLANG_CLIENT_VULKAN;
    input.client_version = GLSLANG_TARGET_VULKAN_1_1;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_3;
    input.code = source.c_str();
    input.default_version = 450;
    input.default_profile = GLSLANG_NO_PROFILE;
    input.messages = GLSLANG_MSG_DEFAULT_BIT;
    input.resource = glslang_default_resource();

    std::vector<uint32_t> spirv;
    glslang_shader_t* shader = glslang_shader_create(&input);
    glslang_program_t* program = nullptr;
    const char* stage = vertex ? "vertex" : "fragment";
    if (!glslang_shader_preprocess(shader, &input) || !glslang_shader_parse(shader, &input)) {
        SPDLOG_ERROR("Vulkan: {} shader compilation failed: {}", stage, glslang_shader_get_info_log(shader));
    } else {
        program = glslang_program_create();
        glslang_program_add_shader(program, shader);
        if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
            SPDLOG_ERROR("Vulkan: {} shader link failed: {}", stage, glslang_program_get_info_log(program));
        } else {
            glslang_program_SPIRV_generate(program, input.stage);
            spirv.resize(glslang_program_SPIRV_get_size(program));
            glslang_program_SPIRV_get(program, spirv.data());
        }
    }
    if (spirv.empty()) {
        int line = 1;
        std::stringstream ss(source);
        std::string l;
        while (std::getline(ss, l)) {
            fprintf(stderr, "%4d| %s\n", line++, l.c_str());
        }
    }
    if (program != nullptr) {
        glslang_program_delete(program);
    }
    glslang_shader_delete(shader);
    if (!spirv.empty()) cache.Store(key, spirv.data(), spirv.size() * sizeof(uint32_t));
    return spirv;
}

// The combiner formula text (gfx_opengl.cpp): the template uses the OpenGL template's names (texVal0, vInput1, texel,
// frame_count, noise_scale)
prism::ContextTypes* append_formula(prism::ContextTypes* _, prism::ContextTypes* a_arg, prism::ContextTypes* a_single,
                                    prism::ContextTypes* a_mult, prism::ContextTypes* a_mix,
                                    prism::ContextTypes* a_with_alpha, prism::ContextTypes* a_only_alpha,
                                    prism::ContextTypes* a_alpha, prism::ContextTypes* a_first_cycle);

std::string GfxRenderingAPIVK::BuildShaderSource(const CCFeatures& ccFeatures, bool vertex, bool threePoint, bool srgb) {
    const bool dualOut = mHasDualSrcBlend && ccFeatures.opt_alpha;
    // the template's fixed attribute/varying locations of the textures' coordinates and clamp values
    static int sLocTex[2] = { 1, 4 };
    static int sLocClamp[2][2] = { { 2, 3 }, { 5, 6 } };
    prism::Processor processor;
    prism::ContextItems context = {
        { "VERTEX_SHADER", vertex },
        { "o_c", M_ARRAY(ccFeatures.c, int, 2, 2, 4) },
        { "o_alpha", ccFeatures.opt_alpha },
        { "o_fog", ccFeatures.opt_fog },
        { "o_texture_edge", ccFeatures.opt_texture_edge },
        { "o_noise", ccFeatures.opt_noise },
        { "o_2cyc", ccFeatures.opt_2cyc },
        { "o_alpha_threshold", ccFeatures.opt_alpha_threshold },
        { "o_invisible", ccFeatures.opt_invisible },
        { "o_grayscale", ccFeatures.opt_grayscale },
        { "o_prim_depth", ccFeatures.opt_prim_depth },
        { "o_textures", M_ARRAY(ccFeatures.usedTextures, bool, 2) },
        { "o_masks", M_ARRAY(ccFeatures.used_masks, bool, 2) },
        { "o_blend", M_ARRAY(ccFeatures.used_blend, bool, 2) },
        { "o_clamp", M_ARRAY(ccFeatures.clamp, bool, 2, 2) },
        { "o_inputs", ccFeatures.numInputs },
        { "loc_tex", M_ARRAY(sLocTex, int, 2) },
        { "loc_clamp", M_ARRAY(sLocClamp, int, 2, 2) },
        { "o_do_mix", M_ARRAY(ccFeatures.do_mix, bool, 2, 2) },
        { "o_do_single", M_ARRAY(ccFeatures.do_single, bool, 2, 2) },
        { "o_do_multiply", M_ARRAY(ccFeatures.do_multiply, bool, 2, 2) },
        { "o_color_alpha_same", M_ARRAY(ccFeatures.color_alpha_same, bool, 2) },
        { "o_three_point_filtering", threePoint },
        { "FILTER_THREE_POINT", FILTER_THREE_POINT },
        { "FILTER_LINEAR", FILTER_LINEAR },
        { "FILTER_NONE", FILTER_NONE },
        { "srgb_mode", srgb },
        // the fragment outputs as flat flags (prism drops the rest of a shader after a nested or @elseif branch that
        // is taken; see gfx_opengl.cpp)
        { "dual_out", dualOut },
        { "single_out", !dualOut },
        { "SHADER_0", SHADER_0 },
        { "SHADER_INPUT_1", SHADER_INPUT_1 },
        { "SHADER_INPUT_2", SHADER_INPUT_2 },
        { "SHADER_INPUT_3", SHADER_INPUT_3 },
        { "SHADER_INPUT_4", SHADER_INPUT_4 },
        { "SHADER_INPUT_5", SHADER_INPUT_5 },
        { "SHADER_INPUT_6", SHADER_INPUT_6 },
        { "SHADER_INPUT_7", SHADER_INPUT_7 },
        { "SHADER_TEXEL0", SHADER_TEXEL0 },
        { "SHADER_TEXEL0A", SHADER_TEXEL0A },
        { "SHADER_TEXEL1", SHADER_TEXEL1 },
        { "SHADER_TEXEL1A", SHADER_TEXEL1A },
        { "SHADER_1", SHADER_1 },
        { "SHADER_COMBINED", SHADER_COMBINED },
        { "SHADER_NOISE", SHADER_NOISE },
        { "append_formula", (InvokeFunc)append_formula },
    };
    processor.populate(context);

    if (gfx_get_shader(ccFeatures.shader_id) != nullptr) {
        // Custom shaders are written for the other backends' languages
        static bool sWarned = false;
        if (!sWarned) {
            SPDLOG_WARN("Vulkan: custom shaders are not supported, drawing with the default shader");
            sWarned = true;
        }
    }
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    auto res = std::static_pointer_cast<Ship::Shader>(
        sVKResourceManager->LoadResource("shaders/vulkan/default.shader.glsl", true, init));
    if (res == nullptr) {
        SPDLOG_CRITICAL("Vulkan: shaders/vulkan/default.shader.glsl is missing from the o2r");
        abort();
    }
    processor.load(*static_cast<std::string*>(res->GetRawPointer()));
    return processor.process();
}

// MARK: - Helpers

static VkSamplerAddressMode gfx_cm_to_vk(uint32_t val, bool hasMirrorClamp) {
    switch (val) {
        case G_TX_NOMIRROR | G_TX_CLAMP:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case G_TX_MIRROR | G_TX_WRAP:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case G_TX_MIRROR | G_TX_CLAMP:
            return hasMirrorClamp ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
                                  : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case G_TX_NOMIRROR | G_TX_WRAP:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

uint32_t GfxRenderingAPIVK::FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    SPDLOG_ERROR("Vulkan: no suitable memory type (bits {:x}, props {:x})", typeBits, (uint32_t)props);
    return 0;
}

void GfxRenderingAPIVK::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                                     VkBuffer& buffer, VkDeviceMemory& memory, uint8_t** mapped) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(mDevice, &bi, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(mDevice, buffer, &req);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(mDevice, &ai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(mDevice, buffer, memory, 0));

    if (mapped != nullptr) {
        VK_CHECK(vkMapMemory(mDevice, memory, 0, VK_WHOLE_SIZE, 0, (void**)mapped));
    }
}

void GfxRenderingAPIVK::EnsureReadbackBuffer(ReadbackBuffer& rb, size_t size) {
    if (rb.capacity >= size) {
        return;
    }
    if (rb.buffer != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(mDevice);
        vkDestroyBuffer(mDevice, rb.buffer, nullptr);
        vkFreeMemory(mDevice, rb.memory, nullptr);
    }
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(mDevice, &bi, nullptr, &rb.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(mDevice, rb.buffer, &req);

    // The CPU reads every byte back: host-cached memory where the device has it (uncached reads are slow)
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProps);
    const VkMemoryPropertyFlags wanted[3] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    uint32_t type = UINT32_MAX;
    for (VkMemoryPropertyFlags flags : wanted) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount && type == UINT32_MAX; i++) {
            if ((req.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags) {
                type = i;
            }
        }
        if (type != UINT32_MAX) {
            break;
        }
    }
    rb.coherent = (memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(mDevice, &ai, nullptr, &rb.memory));
    VK_CHECK(vkBindBufferMemory(mDevice, rb.buffer, rb.memory, 0));
    VK_CHECK(vkMapMemory(mDevice, rb.memory, 0, VK_WHOLE_SIZE, 0, (void**)&rb.mapped));
    rb.capacity = size;
}

// The buffer's bytes after the GPU wrote them (non-coherent memory is invalidated first)
const uint8_t* GfxRenderingAPIVK::ReadbackData(ReadbackBuffer& rb) {
    if (!rb.coherent) {
        VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
        range.memory = rb.memory;
        range.size = VK_WHOLE_SIZE;
        VK_CHECK(vkInvalidateMappedMemoryRanges(mDevice, 1, &range));
    }
    return rb.mapped;
}

void GfxRenderingAPIVK::CreateImage(uint32_t width, uint32_t height, VkSampleCountFlagBits samples, VkFormat format,
                                    VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.extent = { width, height, 1 };
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.format = format;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.usage = usage;
    ii.samples = samples;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(mDevice, &ii, nullptr, &image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(mDevice, image, &req);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(mDevice, &ai, nullptr, &memory));
    VK_CHECK(vkBindImageMemory(mDevice, image, memory, 0));
}

VkImageView GfxRenderingAPIVK::CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { aspect, 0, 1, 0, 1 };
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(mDevice, &vi, nullptr, &view));
    return view;
}

void GfxRenderingAPIVK::TransitionToGeneral(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { aspect, 0, 1, 0, 1 };
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

// Conservative everything-to-everything barrier used around blits/copies.
void GfxRenderingAPIVK::FullBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

// PORT: hands out the frame slot's command buffers in turn and allocates only when a frame needs more than any frame
// before it (each framebuffer drawn, each synchronous readback and the upload buffer take one). The slot's pool is
// reset in NewFrame after its fence, which puts all of them back in the initial state.
VkCommandBuffer GfxRenderingAPIVK::AcquireCommandBuffer() {
    FrameSlot& slot = mFrames[mFrameSlot];
    sStat.commandBuffers++;
    // PORT debug: PORT_DEBUG_VK_OOM=<n> makes the n-th acquire fail as a driver out of memory does
    static long sFailAt = getenv("PORT_DEBUG_VK_OOM") ? strtol(getenv("PORT_DEBUG_VK_OOM"), nullptr, 10) : 0;
    static long sAcquires = 0;
    if (++sAcquires == sFailAt) {
        VkCheckFailed(VK_ERROR_OUT_OF_HOST_MEMORY, "vkAllocateCommandBuffers (PORT_DEBUG_VK_OOM)", __LINE__);
    }
    if (slot.commandBuffersUsed < slot.commandBuffers.size()) {
        return slot.commandBuffers[slot.commandBuffersUsed++];
    }
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = slot.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &ai, &cmd));
    sStat.commandBuffersAllocated++;
    slot.commandBuffers.push_back(cmd);
    slot.commandBuffersUsed++;
    return cmd;
}

VkCommandBuffer GfxRenderingAPIVK::BeginOneShot() {
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = mFrames[mFrameSlot].commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &ai, &cmd));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void GfxRenderingAPIVK::EndOneShot(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(mQueue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(mQueue));
    vkFreeCommandBuffers(mDevice, mFrames[mFrameSlot].commandPool, 1, &cmd);
}

// Canonical render pass per (msaa, hasDepth). All attachments LOAD/STORE in LAYOUT_GENERAL; MSAA passes add a resolve
// attachment (the sampled texture).
VkRenderPass GfxRenderingAPIVK::GetRenderPass(uint32_t msaaLevel, bool hasDepth) {
    std::lock_guard<std::mutex> lock(mRenderPassMutex); // warm-up workers create pipelines too
    uint32_t key = msaaLevel | (hasDepth ? 1u << 8 : 0);
    auto it = mRenderPassCache.find(key);
    if (it != mRenderPassCache.end()) {
        return it->second;
    }

    bool msaa = msaaLevel > 1;
    std::vector<VkAttachmentDescription> attachments;

    VkAttachmentDescription color = {};
    color.format = mSwapchainFormat;
    color.samples = (VkSampleCountFlagBits)msaaLevel;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    color.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    attachments.push_back(color);

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference resolveRef = { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference depthRef = { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL };

    if (msaa) {
        VkAttachmentDescription resolve = color;
        resolve.samples = VK_SAMPLE_COUNT_1_BIT;
        resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments.push_back(resolve);
        resolveRef.attachment = 1;
    }

    if (hasDepth) {
        VkAttachmentDescription depth = {};
        depth.format = VK_FORMAT_D32_SFLOAT;
        depth.samples = (VkSampleCountFlagBits)msaaLevel;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        depth.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthRef.attachment = (uint32_t)attachments.size();
        attachments.push_back(depth);
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pResolveAttachments = msaa ? &resolveRef : nullptr;
    subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

    // Conservative full external dependencies: framebuffer textures get sampled by later framebuffers' draws within
    // the same submission.
    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    VkRenderPassCreateInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rp.attachmentCount = (uint32_t)attachments.size();
    rp.pAttachments = attachments.data();
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;

    VkRenderPass pass = VK_NULL_HANDLE;
    VK_CHECK(vkCreateRenderPass(mDevice, &rp, nullptr, &pass));
    mRenderPassCache[key] = pass;
    return pass;
}

VkSampler GfxRenderingAPIVK::GetSampler(bool linear, uint32_t cms, uint32_t cmt) {
    uint32_t key = (linear ? 1u : 0) | (cms << 1) | (cmt << 9) | ((uint32_t)mCurrentFilterMode << 17);
    auto it = mSamplerCache.find(key);
    if (it != mSamplerCache.end()) {
        return it->second;
    }

    VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkFilter filter = linear && mCurrentFilterMode == FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.magFilter = filter;
    si.minFilter = filter;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = gfx_cm_to_vk(cms, mHasMirrorClampToEdge);
    si.addressModeV = gfx_cm_to_vk(cmt, mHasMirrorClampToEdge);
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;

    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(mDevice, &si, nullptr, &sampler));
    mSamplerCache[key] = sampler;
    return sampler;
}

VkDescriptorPool GfxRenderingAPIVK::CreateDescriptorPool() {
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 * SHADER_MAX_TEXTURES };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 4096;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &dpi, nullptr, &pool));
    return pool;
}

VkDescriptorSet GfxRenderingAPIVK::AllocateTextureSet() {
    FrameSlot& slot = mFrames[mFrameSlot];
    VkDescriptorSetAllocateInfo dsa = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &mTextureSetLayout;
    for (;;) {
        if (slot.descriptorPoolIndex == slot.descriptorPools.size()) {
            slot.descriptorPools.push_back(CreateDescriptorPool());
        }
        dsa.descriptorPool = slot.descriptorPools[slot.descriptorPoolIndex];
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult res = vkAllocateDescriptorSets(mDevice, &dsa, &set);
        if (res == VK_SUCCESS) {
            return set;
        }
        if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) {
            SPDLOG_ERROR("Vulkan: vkAllocateDescriptorSets failed ({})", (int)res);
            return VK_NULL_HANDLE;
        }
        slot.descriptorPoolIndex++; // this pool is full: the next one (created if needed)
    }
}

void GfxRenderingAPIVK::EnsureUploadCmd() {
    if (mUploadCmd != VK_NULL_HANDLE) {
        return;
    }
    mUploadCmd = AcquireCommandBuffer();
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(mUploadCmd, &bi));
}

void GfxRenderingAPIVK::GrowStaging(FrameSlot& slot, size_t needed) {
    // Defer-destroy the old buffer: copies already recorded this frame still reference it, and it's freed in
    // CollectGarbage after this slot's fence, so a bigger buffer swaps in without a queue stall. Bytes already
    // staged in [0, mStagingOffset) stay valid in the (deferred) old one.
    size_t newCap = needed + (needed >> 2); // +25% headroom
    slot.garbageBuffers.push_back(slot.stagingBuffer);
    slot.garbageMemory.push_back(slot.stagingMemory);
    slot.stagingCapacity = newCap;
    CreateBuffer(newCap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, slot.stagingBuffer,
                 slot.stagingMemory, &slot.stagingMapped);
    SPDLOG_INFO("Vulkan: grew staging buffer to {} MB", newCap / (1024 * 1024));
}

// MARK: - Swapchain

void GfxRenderingAPIVK::DestroySwapchainViews() {
    for (VkFramebuffer fb : mSwapchainFramebuffers) {
        vkDestroyFramebuffer(mDevice, fb, nullptr);
    }
    mSwapchainFramebuffers.clear();
    for (VkImageView view : mSwapchainViews) {
        vkDestroyImageView(mDevice, view, nullptr);
    }
    mSwapchainViews.clear();
}

static void CheckSwapchainResult(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        const std::string message = std::string(operation) + " failed (Vulkan error " + std::to_string(result) + ").";
        SPDLOG_ERROR("{}", message);
        throw VulkanSwapchainError(message);
    }
}

void GfxRenderingAPIVK::CreateSwapchain(uint32_t width, uint32_t height) {
    CheckSwapchainResult(vkDeviceWaitIdle(mDevice), "vkDeviceWaitIdle");
    DestroySwapchainViews();

    VkSurfaceCapabilitiesKHR caps = {};
    CheckSwapchainResult(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &caps),
                         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    uint32_t formatCount = 0;
    CheckSwapchainResult(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, nullptr),
                         "vkGetPhysicalDeviceSurfaceFormatsKHR");
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    CheckSwapchainResult(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, formats.data()),
                         "vkGetPhysicalDeviceSurfaceFormatsKHR");

    if (formats.empty()) {
        throw VulkanSwapchainError("The Vulkan driver reported no surface formats.");
    }
    VkSurfaceFormatKHR chosen = formats[0];
    if (chosen.format == VK_FORMAT_UNDEFINED) {
        chosen.format = VK_FORMAT_B8G8R8A8_UNORM;
    }
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    if (mSwapchain != VK_NULL_HANDLE && chosen.format != mSwapchainFormat) {
        // The render passes and offscreen targets use the swapchain format; it does not change in practice
        SPDLOG_WARN("Vulkan: swapchain format changed ({} -> {})", (int)mSwapchainFormat, (int)chosen.format);
    }
    mSwapchainFormat = chosen.format;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent = { std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width),
                   std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height) };
    }
    mSwapchainExtent = extent;

    uint32_t imageCount = std::max(caps.minImageCount + 1, 3u);
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }

    // PORT: vsync off (the console variable) presents with MAILBOX or IMMEDIATE where the surface offers them
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!mVsync) {
        uint32_t modeCount = 0;
        CheckSwapchainResult(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &modeCount, nullptr),
                             "vkGetPhysicalDeviceSurfacePresentModesKHR");
        std::vector<VkPresentModeKHR> modes(modeCount);
        CheckSwapchainResult(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &modeCount, modes.data()),
                             "vkGetPhysicalDeviceSurfacePresentModesKHR");
        for (VkPresentModeKHR want : { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR }) {
            if (std::find(modes.begin(), modes.end(), want) != modes.end()) {
                presentMode = want;
                break;
            }
        }
    }

    VkSwapchainCreateInfoKHR sci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = mSurface;
    sci.minImageCount = imageCount;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    const VkImageUsageFlags requiredUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((caps.supportedUsageFlags & requiredUsage) != requiredUsage) {
        throw VulkanSwapchainError("The Vulkan surface does not support the required render/copy image usage.");
    }
    // Sampling a swapchain image is optional; do not request a usage the surface does not support.
    sci.imageUsage = requiredUsage | (caps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT);
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    if (caps.supportedCompositeAlpha == 0) {
        throw VulkanSwapchainError("The Vulkan surface reported no supported composite alpha mode.");
    }
    // Pick a mode actually advertised by this surface (OPAQUE first when available).
    sci.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(caps.supportedCompositeAlpha &
                                                                 (~caps.supportedCompositeAlpha + 1u));
    sci.presentMode = presentMode;
    if (sStats) {
        fprintf(stderr, "[vkstats] swapchain %ux%u, present mode %d (vsync %s)\n", extent.width, extent.height,
                (int)presentMode, mVsync ? "on" : "off");
    }
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = mSwapchain;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    // PORT debug: fail a chosen creation (1 = startup, 2 = first resize) to test the driver's error path.
    static unsigned int creations = 0;
    const char* failAt = getenv("PORT_VK_FAIL_SWAPCHAIN_AT");
    const bool injectFailure = failAt != nullptr && ++creations == strtoul(failAt, nullptr, 10);
    CheckSwapchainResult(injectFailure ? VK_ERROR_OUT_OF_HOST_MEMORY
                                      : vkCreateSwapchainKHR(mDevice, &sci, nullptr, &newSwapchain),
                         "vkCreateSwapchainKHR");
    if (mSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr);
    }
    mSwapchain = newSwapchain;

    uint32_t count = 0;
    CheckSwapchainResult(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &count, nullptr),
                         "vkGetSwapchainImagesKHR");
    mSwapchainImages.resize(count);
    CheckSwapchainResult(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &count, mSwapchainImages.data()),
                         "vkGetSwapchainImagesKHR");

    if (count == 0) {
        throw VulkanSwapchainError("The Vulkan driver returned an empty swapchain.");
    }
    mSwapchainViews.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        VkImageViewCreateInfo view = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        view.image = mSwapchainImages[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = mSwapchainFormat;
        view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        CheckSwapchainResult(vkCreateImageView(mDevice, &view, nullptr, &mSwapchainViews[i]), "vkCreateImageView");
    }
    mSwapchainImageUsed.assign(count, false);

    mSwapchainNeedsRecreate = false;
    RebuildScreenFramebuffer();
}

// (Re)creates the screen depth buffer and the per-swapchain-image framebuffers.
void GfxRenderingAPIVK::RebuildScreenFramebuffer() {
    if (mFramebuffers.empty()) {
        return; // Init() hasn't created fb 0 yet; called again afterwards
    }

    FramebufferVK& fb = mFramebuffers[0];
    TextureDataVK& tex = mTextures[fb.mTextureId];

    uint32_t width = mSwapchainExtent.width;
    uint32_t height = mSwapchainExtent.height;

    if (fb.mDepthImage == VK_NULL_HANDLE || fb.mWidth != width || fb.mHeight != height) {
        if (fb.mDepthView != VK_NULL_HANDLE) {
            vkDestroyImageView(mDevice, fb.mDepthView, nullptr);
            vkDestroyImage(mDevice, fb.mDepthImage, nullptr);
            vkFreeMemory(mDevice, fb.mDepthMemory, nullptr);
        }
        CreateImage(width, height, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, fb.mDepthImage,
                    fb.mDepthMemory);
        fb.mDepthView = CreateImageView(fb.mDepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkCommandBuffer cmd = BeginOneShot();
        TransitionToGeneral(cmd, fb.mDepthImage, VK_IMAGE_ASPECT_DEPTH_BIT);
        EndOneShot(cmd);
    }

    fb.mWidth = width;
    fb.mHeight = height;
    fb.mMsaaLevel = 1;
    fb.mHasDepthBuffer = true;
    fb.mRenderTarget = true;
    fb.mRenderPass = GetRenderPass(1, true);
    tex.width = width;
    tex.height = height;
    tex.isSwapchainAlias = true;

    for (VkFramebuffer f : mSwapchainFramebuffers) {
        vkDestroyFramebuffer(mDevice, f, nullptr);
    }
    mSwapchainFramebuffers.resize(mSwapchainViews.size());
    for (size_t i = 0; i < mSwapchainViews.size(); i++) {
        VkImageView attachments[2] = { mSwapchainViews[i], fb.mDepthView };
        VkFramebufferCreateInfo fci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fci.renderPass = fb.mRenderPass;
        fci.attachmentCount = 2;
        fci.pAttachments = attachments;
        fci.width = width;
        fci.height = height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(mDevice, &fci, nullptr, &mSwapchainFramebuffers[i]));
    }
}

// MARK: - ImGui & SDL hooks (called from Fast3dGui)

bool GfxRenderingAPIVK::VulkanInit(SDL_Window* window) {
    mShaderCache.Open("vulkan-spirv");
    mWindow = window;
    glslang_initialize_process();

    // The library SDL loaded for the window (Vulkan_LoadLibrary) provides the entry points
    auto getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
    if (getInstanceProcAddr == nullptr) {
        SPDLOG_ERROR("Vulkan: no vkGetInstanceProcAddr: {}", SDL_GetError());
        return false;
    }
    volkInitializeCustom(getInstanceProcAddr);

    // Instance
    Uint32 sdlExtCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    std::vector<const char*> instanceExts(sdlExts, sdlExts + sdlExtCount);

    uint32_t availExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, nullptr);
    std::vector<VkExtensionProperties> availExts(availExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, availExts.data());
    bool hasPortabilityEnum = false;
    for (const auto& e : availExts) {
        if (strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            hasPortabilityEnum = true;
        }
    }
    if (hasPortabilityEnum) {
        instanceExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME); // MoltenVK
    }

    std::vector<const char*> layers;
    if (getenv("PORT_VK_VALIDATION") != nullptr) { // debug: the Khronos validation layer (Vulkan SDK)
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "libultraship";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instanceExts.size();
    ici.ppEnabledExtensionNames = instanceExts.data();
    ici.enabledLayerCount = (uint32_t)layers.size();
    ici.ppEnabledLayerNames = layers.data();
    if (hasPortabilityEnum) {
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    VkResult res = vkCreateInstance(&ici, nullptr, &mInstance);
    if (res != VK_SUCCESS) {
        SPDLOG_ERROR("Vulkan: vkCreateInstance failed ({})", (int)res);
        return false;
    }
    volkLoadInstance(mInstance);

    if (!SDL_Vulkan_CreateSurface(window, mInstance, nullptr, &mSurface)) {
        SPDLOG_ERROR("Vulkan: SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        return false;
    }

    // Physical device + queue family with graphics & present; discrete GPUs first
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &devCount, nullptr);
    if (devCount == 0) {
        SPDLOG_ERROR("Vulkan: no physical devices");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(devCount);
    vkEnumeratePhysicalDevices(mInstance, &devCount, devices.data());
    std::stable_sort(devices.begin(), devices.end(), [](VkPhysicalDevice a, VkPhysicalDevice b) {
        VkPhysicalDeviceProperties pa, pb;
        vkGetPhysicalDeviceProperties(a, &pa);
        vkGetPhysicalDeviceProperties(b, &pb);
        return pa.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
               pb.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    });

    for (VkPhysicalDevice dev : devices) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, queues.data());
        for (uint32_t i = 0; i < qCount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, mSurface, &present);
            if ((queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                mPhysicalDevice = dev;
                mQueueFamily = i;
                break;
            }
        }
        if (mPhysicalDevice != VK_NULL_HANDLE) {
            break;
        }
    }
    if (mPhysicalDevice == VK_NULL_HANDLE) {
        SPDLOG_ERROR("Vulkan: no device with graphics+present support");
        return false;
    }
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &mDeviceProps);
    SPDLOG_INFO("Vulkan device: {} (API {}.{}.{})", mDeviceProps.deviceName, VK_VERSION_MAJOR(mDeviceProps.apiVersion),
                VK_VERSION_MINOR(mDeviceProps.apiVersion), VK_VERSION_PATCH(mDeviceProps.apiVersion));

    // Device
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &devExtCount, devExts.data());
    std::vector<const char*> enabledDevExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    for (const auto& e : devExts) {
        if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
            enabledDevExts.push_back("VK_KHR_portability_subset");
        }
        if (strcmp(e.extensionName, VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME) == 0) {
            enabledDevExts.push_back(VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);
            mHasMirrorClampToEdge = true;
        }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = mQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures supported = {};
    vkGetPhysicalDeviceFeatures(mPhysicalDevice, &supported);
    VkPhysicalDeviceFeatures features = {};
    features.sampleRateShading = supported.sampleRateShading;
    features.depthClamp = supported.depthClamp;
    // PORT: the colour blends by a second shader output's alpha (as Metal/DX11/GL; default.shader.* templates)
    features.dualSrcBlend = supported.dualSrcBlend;
    mHasDepthClamp = supported.depthClamp;
    mHasDualSrcBlend = supported.dualSrcBlend;

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabledDevExts.size();
    dci.ppEnabledExtensionNames = enabledDevExts.data();
    dci.pEnabledFeatures = &features;
    res = vkCreateDevice(mPhysicalDevice, &dci, nullptr, &mDevice);
    if (res != VK_SUCCESS) {
        SPDLOG_ERROR("Vulkan: vkCreateDevice failed ({})", (int)res);
        return false;
    }
    volkLoadDevice(mDevice);
    vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);

    // Texture set (bindings 0-5 as the template), push constants, pipeline cache
    {
        VkDescriptorSetLayoutBinding bindings[SHADER_MAX_TEXTURES] = {};
        for (uint32_t i = 0; i < SHADER_MAX_TEXTURES; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        li.bindingCount = SHADER_MAX_TEXTURES;
        li.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &li, nullptr, &mTextureSetLayout));

        VkPushConstantRange range = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VulkanPushConstants) };
        VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &mTextureSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(mDevice, &pli, nullptr, &mPipelineLayout));

        // PORT: the driver's pipeline cache from earlier runs (it ignores data from another driver or device)
        mPipelineCachePath = Ship::Context::GetPathRelativeToAppDirectory("vulkan_pipeline_cache.bin");
        mPipelineListPath = Ship::Context::GetPathRelativeToAppDirectory("vulkan_pipelines.txt");
        std::vector<uint8_t> cacheData;
        if (FILE* f = fopen(mPipelineCachePath.c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (size > 0) {
                cacheData.resize((size_t)size);
                if (fread(cacheData.data(), 1, cacheData.size(), f) != cacheData.size()) {
                    cacheData.clear();
                }
            }
            fclose(f);
        }
        VkPipelineCacheCreateInfo pci = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        pci.initialDataSize = cacheData.size();
        pci.pInitialData = cacheData.empty() ? nullptr : cacheData.data();
        if (vkCreatePipelineCache(mDevice, &pci, nullptr, &mPipelineCache) != VK_SUCCESS) {
            pci.initialDataSize = 0; // unusable data: start empty
            pci.pInitialData = nullptr;
            VK_CHECK(vkCreatePipelineCache(mDevice, &pci, nullptr, &mPipelineCache));
        }
        mPipelineCacheSaved = std::chrono::steady_clock::now();
        LoadPipelineList();
    }

    VkFenceCreateInfo readbackFence = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(mDevice, &readbackFence, nullptr, &mReadbackFence));

    // Per-frame slots
    mVertexRingCapacity = 256 * 32 * 3 * sizeof(float) * 50;
    for (uint32_t i = 0; i < VK_FRAMES_IN_FLIGHT; i++) {
        FrameSlot& slot = mFrames[i];

        VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpi.queueFamilyIndex = mQueueFamily;
        cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VK_CHECK(vkCreateCommandPool(mDevice, &cpi, nullptr, &slot.commandPool));

        slot.descriptorPools.push_back(CreateDescriptorPool());

        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(mDevice, &fci, nullptr, &slot.fence));
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VK_CHECK(vkCreateSemaphore(mDevice, &sci, nullptr, &slot.imageAvailable));
        VK_CHECK(vkCreateSemaphore(mDevice, &sci, nullptr, &slot.renderFinished));

        CreateBuffer(mVertexRingCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, slot.vertexBuffer,
                     slot.vertexMemory, &slot.vertexMapped);
        slot.vertexCapacity = mVertexRingCapacity;
        slot.stagingCapacity = 32 * 1024 * 1024;
        CreateBuffer(slot.stagingCapacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, slot.stagingBuffer,
                     slot.stagingMemory, &slot.stagingMapped);
    }

    // 1x1 white dummy texture for unused sampler slots
    {
        CreateImage(1, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mDummyTexture.image,
                    mDummyTexture.memory);
        mDummyTexture.view = CreateImageView(mDummyTexture.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
        mDummyTexture.sampler = GetSampler(false, G_TX_WRAP, G_TX_WRAP);

        VkCommandBuffer cmd = BeginOneShot();
        TransitionToGeneral(cmd, mDummyTexture.image, VK_IMAGE_ASPECT_COLOR_BIT);
        VkClearColorValue white = { { 1.0f, 1.0f, 1.0f, 1.0f } };
        VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, mDummyTexture.image, VK_IMAGE_LAYOUT_GENERAL, &white, 1, &range);
        FullBarrier(cmd);
        EndOneShot(cmd);
    }

    // ImGui samples its textures (the game picture among them) linearly, as imgui_impl_metal does
    {
        VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(mDevice, &si, nullptr, &mImGuiSampler));
        // PORT: and nearest, for a sharp upscale of the game picture (GetFramebufferTextureIdNearest)
        si.magFilter = VK_FILTER_NEAREST;
        VK_CHECK(vkCreateSampler(mDevice, &si, nullptr, &mImGuiSamplerNearest));
    }

    // Initial swapchain
    mVsync = mConsoleVariable == nullptr || mConsoleVariable->GetInteger(CVAR_VSYNC_ENABLED, 1) != 0;
    int dw = 0, dh = 0;
    SDL_GetWindowSizeInPixels(window, &dw, &dh);
    CreateSwapchain((uint32_t)std::max(dw, 1), (uint32_t)std::max(dh, 1));

    // debug: PORT_VK_SELFTEST=1 compiles a trivial shader pair (checks the glslang wiring)
    if (getenv("PORT_VK_SELFTEST") != nullptr) {
        auto vs = CompileGlslToSpirv(mShaderCache, "#version 450\nlayout(location = 0) in vec4 aPos;\n"
                                     "void main() { gl_Position = aPos; }\n",
                                     true);
        auto fs = CompileGlslToSpirv(mShaderCache, "#version 450\nlayout(location = 0, index = 0) out vec4 o0;\n"
                                     "layout(location = 0, index = 1) out vec4 o1;\n"
                                     "void main() { o0 = vec4(1.0); o1 = vec4(0.5); }\n",
                                     false);
        VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        VkShaderModule modules[2] = {};
        bool ok = !vs.empty() && !fs.empty();
        if (ok) {
            smci.codeSize = vs.size() * sizeof(uint32_t);
            smci.pCode = vs.data();
            ok = vkCreateShaderModule(mDevice, &smci, nullptr, &modules[0]) == VK_SUCCESS;
            smci.codeSize = fs.size() * sizeof(uint32_t);
            smci.pCode = fs.data();
            ok = ok && vkCreateShaderModule(mDevice, &smci, nullptr, &modules[1]) == VK_SUCCESS;
        }
        fprintf(stderr, "[vulkan] selftest on %s: SPIR-V %zu + %zu words, shader modules %s\n", mDeviceProps.deviceName,
                vs.size(), fs.size(), ok ? "ok" : "FAILED");
        for (VkShaderModule m : modules) {
            if (m != VK_NULL_HANDLE) {
                vkDestroyShaderModule(mDevice, m, nullptr);
            }
        }
    }

    // ImGui
    ImGui_ImplVulkan_InitInfo info = {};
    info.ApiVersion = VK_API_VERSION_1_1;
    info.Instance = mInstance;
    info.PhysicalDevice = mPhysicalDevice;
    info.Device = mDevice;
    info.QueueFamily = mQueueFamily;
    info.Queue = mQueue;
    info.RenderPass = GetRenderPass(1, true);
    info.MinImageCount = 2;
    info.ImageCount = (uint32_t)mSwapchainImages.size();
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.DescriptorPoolSize = 512;
    if (!ImGui_ImplVulkan_Init(&info)) {
        SPDLOG_ERROR("Vulkan: ImGui_ImplVulkan_Init failed");
        return false;
    }
    mImGuiInitialized = true;
    return true;
}

GfxRenderingAPIVK::~GfxRenderingAPIVK() {
    mPrebuildStop = true;
    for (auto& t : mPrebuildThreads) {
        t.join();
    }
    SavePipelineCache();
}

// MARK: - Shader warm-up and the on-disk caches

// vulkan_pipelines.txt: one "<shader id0> <shader id1> <render state key>" line (hex) per pipeline drawn so far
void GfxRenderingAPIVK::LoadPipelineList() {
    FILE* f = fopen(mPipelineListPath.c_str(), "r");
    if (f == nullptr) {
        return;
    }
    unsigned long long id0, id1;
    unsigned key;
    while (fscanf(f, "%llx %llx %x", &id0, &id1, &key) == 3) {
        if (mKnownPipelines.insert({ id0, id1, key }).second) {
            mPipelineList[{ id0, id1 }].push_back(key);
        }
    }
    fclose(f);
}

void GfxRenderingAPIVK::RecordPipeline(uint64_t shaderId0, uint64_t shaderId1, uint32_t key) {
    mPipelineCacheDirty = true;
    if (!mKnownPipelines.insert({ shaderId0, shaderId1, key }).second) {
        return;
    }
    if (FILE* f = fopen(mPipelineListPath.c_str(), "a")) {
        fprintf(f, "%016llx %016llx %x\n", (unsigned long long)shaderId0, (unsigned long long)shaderId1, key);
        fclose(f);
    }
}

void GfxRenderingAPIVK::SavePipelineCache() {
    if (mPipelineCache == VK_NULL_HANDLE || !mPipelineCacheDirty) {
        return;
    }
    size_t size = 0;
    if (vkGetPipelineCacheData(mDevice, mPipelineCache, &size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<uint8_t> data(size);
    if (vkGetPipelineCacheData(mDevice, mPipelineCache, &size, data.data()) != VK_SUCCESS) {
        return;
    }
    // written beside and renamed, so a crash mid-write leaves the old file
    const std::string tmp = mPipelineCachePath + ".tmp";
    if (FILE* f = fopen(tmp.c_str(), "wb")) {
        const bool ok = fwrite(data.data(), 1, size, f) == size;
        fclose(f);
        if (ok) {
            std::remove(mPipelineCachePath.c_str());
            std::rename(tmp.c_str(), mPipelineCachePath.c_str());
        }
    }
    mPipelineCacheDirty = false;
    mPipelineCacheSaved = std::chrono::steady_clock::now();
}

// PORT: shader warm-up (Interpreter::QueueShaderWarmup): the programs earlier runs drew, built on worker threads with
// the pipelines vulkan_pipelines.txt lists for them; CreateAndLoadNewShader takes them.
bool GfxRenderingAPIVK::PrebuildShadersAsync(const std::vector<std::pair<uint64_t, uint64_t>>& ids) {
    if (mDevice == VK_NULL_HANDLE || getenv("PORT_VK_NO_PREBUILD") != nullptr) { // debug: build on first use only
        return false;
    }
    std::lock_guard<std::mutex> lock(mPrebuildMutex);
    // The shared warm-up list covers programs. Vulkan also needs the render
    // state for each pipeline, even when the program is already compiled.
    if (sVKResourceManager) {
        auto archive = sVKResourceManager->GetArchiveManager();
        if (auto file = archive->LoadFile("port/vulkan_pipeline_warmup.txt"); file && file->Buffer) {
            std::istringstream recipes(std::string(file->Buffer->begin(), file->Buffer->end()));
            uint64_t id0, id1;
            uint32_t state;
            while (recipes >> std::hex >> id0 >> id1 >> state) {
                if (mKnownPipelines.insert({ id0, id1, state }).second) {
                    mPipelineList[{ id0, id1 }].push_back(state);
                }
            }
        }
    }
    std::set<std::pair<uint64_t, uint64_t>> queued;
    for (const auto& [key, job] : mPrebuildQueue) {
        queued.insert(key);
    }
    auto add = [&](const std::pair<uint64_t, uint64_t>& key) {
        if (queued.insert(key).second) {
            auto it = mPipelineList.find(key);
            mPrebuildQueue.push_back({ key, it != mPipelineList.end() ? it->second : std::vector<uint32_t>{} });
        }
    };
    for (const auto& key : ids) {
        add(key);
    }
    for (const auto& [key, states] : mPipelineList) { // drawn before, though not on the warm-up list
        add(key);
    }
    mPrebuildThreePoint = mCurrentFilterMode == FILTER_THREE_POINT;
    mPrebuildSrgb = mSrgbMode;
    if (mPrebuildThreads.empty() && !mPrebuildQueue.empty()) {
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

float GfxRenderingAPIVK::ShaderPrebuildProgress() {
    std::lock_guard<std::mutex> lock(mPrebuildMutex);
    if (mPrebuildQueue.empty()) return 1.0f;
    return float(mPrebuildNext - mPrebuilding.size()) / float(mPrebuildQueue.size());
}

void GfxRenderingAPIVK::PrebuildWorker() {
    std::unique_lock<std::mutex> lock(mPrebuildMutex);
    while (!mPrebuildStop && mPrebuildNext < mPrebuildQueue.size()) {
        auto [key, states] = mPrebuildQueue[mPrebuildNext++];
        if (mPrebuildSkip.count(key) || mPrebuilt.count(key) || mPrebuilding.count(key)) {
            continue;
        }
        mPrebuilding.insert(key);
        const bool threePoint = mPrebuildThreePoint, srgb = mPrebuildSrgb;
        lock.unlock();
        BuiltProgramVK built = BuildProgram(key.first, key.second, threePoint, srgb);
        for (uint32_t state : states) {
            built.pipelines[state] = CreatePipeline(built, state);
            mPipelineCacheDirty = true;
        }
        lock.lock();
        mPrebuilding.erase(key);
        mPrebuilt[key] = std::move(built);
        mPrebuildBuilt++;
        mPrebuildCond.notify_all();
    }
    if (--mPrebuildRunning == 0 && sStats) {
        fprintf(stderr, "[vkstats] warm-up done: %zu programs queued, %zu built ahead, in %.0f ms\n",
                mPrebuildQueue.size(), mPrebuildBuilt, MsSince(mPrebuildStart));
    }
}

void GfxRenderingAPIVK::ShutdownImGui() {
    if (mImGuiInitialized) {
        vkDeviceWaitIdle(mDevice);
        ImGui_ImplVulkan_Shutdown();
        mImGuiInitialized = false;
    }
}

void GfxRenderingAPIVK::CollectGarbage(FrameSlot& slot) {
    for (auto& [img, mem] : slot.garbageImages) {
        vkDestroyImage(mDevice, img, nullptr);
        vkFreeMemory(mDevice, mem, nullptr);
    }
    slot.garbageImages.clear();
    for (VkImageView view : slot.garbageViews) {
        vkDestroyImageView(mDevice, view, nullptr);
    }
    slot.garbageViews.clear();
    for (VkBuffer buf : slot.garbageBuffers) {
        vkDestroyBuffer(mDevice, buf, nullptr);
    }
    slot.garbageBuffers.clear();
    for (VkDeviceMemory mem : slot.garbageMemory) {
        vkFreeMemory(mDevice, mem, nullptr);
    }
    slot.garbageMemory.clear();
    for (VkPipeline pipeline : slot.garbagePipelines) {
        vkDestroyPipeline(mDevice, pipeline, nullptr);
    }
    slot.garbagePipelines.clear();
    for (VkShaderModule module : slot.garbageModules) {
        vkDestroyShaderModule(mDevice, module, nullptr);
    }
    slot.garbageModules.clear();
}

void GfxRenderingAPIVK::NewFrame() {
    ImGui_ImplVulkan_NewFrame();
    if (mDevice == VK_NULL_HANDLE) {
        return;
    }
    FrameSlot& slot = mFrames[mFrameSlot];

    if (slot.fenceInFlight) {
        const auto statStart = std::chrono::steady_clock::now();
        VK_CHECK(vkWaitForFences(mDevice, 1, &slot.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(mDevice, 1, &slot.fence));
        slot.fenceInFlight = false;
        sStat.fenceMs += MsSince(statStart);
    }
    CollectGarbage(slot);
    VK_CHECK(vkResetCommandPool(mDevice, slot.commandPool, 0));
    slot.commandBuffersUsed = 0; // all of the slot's command buffers are free again
    for (VkDescriptorPool pool : slot.descriptorPools) {
        VK_CHECK(vkResetDescriptorPool(mDevice, pool, 0));
    }
    slot.descriptorPoolIndex = 0;

    mVertexRingOffset = 0;
    mVertexRingCapacity = slot.vertexCapacity;
    mStagingOffset = 0;
    mUploadCmd = VK_NULL_HANDLE;

    // Swapchain housekeeping
    const bool vsync = mConsoleVariable == nullptr || mConsoleVariable->GetInteger(CVAR_VSYNC_ENABLED, 1) != 0;
    if (vsync != mVsync) {
        mVsync = vsync;
        mSwapchainNeedsRecreate = true;
    }
    int dw = 0, dh = 0;
    SDL_GetWindowSizeInPixels(mWindow, &dw, &dh);
    if (dw <= 0 || dh <= 0) {
        return; // minimised: no frame (mFrameActive stays false, nothing is recorded or presented)
    }
    if (mSwapchainNeedsRecreate || (uint32_t)dw != mSwapchainExtent.width || (uint32_t)dh != mSwapchainExtent.height) {
        CreateSwapchain((uint32_t)dw, (uint32_t)dh);
    }

    const auto acquireStart = std::chrono::steady_clock::now();
    VkResult acquire = vkAcquireNextImageKHR(mDevice, mSwapchain, UINT64_MAX, slot.imageAvailable, VK_NULL_HANDLE,
                                             &mSwapchainImageIndex);
    sStat.acquireMs += MsSince(acquireStart);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        CreateSwapchain((uint32_t)dw, (uint32_t)dh);
        acquire = vkAcquireNextImageKHR(mDevice, mSwapchain, UINT64_MAX, slot.imageAvailable, VK_NULL_HANDLE,
                                        &mSwapchainImageIndex);
    }
    if (acquire == VK_SUBOPTIMAL_KHR) {
        mSwapchainNeedsRecreate = true; // the image is still presentable; recreated next frame
    } else if (acquire != VK_SUCCESS) {
        SPDLOG_ERROR("Vulkan: vkAcquireNextImageKHR failed ({})", (int)acquire);
        return;
    }

    // fb 0's texture aliases the acquired swapchain image
    if (!mFramebuffers.empty()) {
        TextureDataVK& tex = mTextures[mFramebuffers[0].mTextureId];
        tex.view = mSwapchainViews[mSwapchainImageIndex];
        tex.image = mSwapchainImages[mSwapchainImageIndex];
        tex.width = mSwapchainExtent.width;
        tex.height = mSwapchainExtent.height;
        mFramebuffers[0].mFramebuffer = mSwapchainFramebuffers[mSwapchainImageIndex];
    }

    mFrameActive = true;
}

void GfxRenderingAPIVK::RenderDrawData(ImDrawData* drawData) {
    if (!mFrameActive) {
        return;
    }
    EnsurePassActive(0);
    FramebufferVK& fb = mFramebuffers[0];
    ImGui_ImplVulkan_RenderDrawData(drawData, fb.mCommandBuffer);
    // ImGui leaves arbitrary pipeline/descriptor/viewport state behind
    fb.mLastPipeline = VK_NULL_HANDLE;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        fb.mLastBoundViews[i] = VK_NULL_HANDLE;
        fb.mLastBoundSamplers[i] = VK_NULL_HANDLE;
    }
    vkCmdSetDepthBias(fb.mCommandBuffer, 0.0f, 0.0f, fb.mLastDepthBias);
    vkCmdSetViewport(fb.mCommandBuffer, 0, 1, &fb.mViewport);
    vkCmdSetScissor(fb.mCommandBuffer, 0, 1, &fb.mScissor);
}

// MARK: - GfxRenderingAPI implementation

const char* GfxRenderingAPIVK::GetName() {
    return "Vulkan";
}

int GfxRenderingAPIVK::GetMaxTextureSize() {
    return (int)mDeviceProps.limits.maxImageDimension2D;
}

GfxClipParameters GfxRenderingAPIVK::GetClipParameters() {
    // Same conventions as Metal: z in 0..1, no VS-side y inversion (the negative viewport gives the framebuffer the
    // same orientation).
    return { true, false };
}

void GfxRenderingAPIVK::Init() {
    sVkRenderThread = std::this_thread::get_id(); // the thread that draws (VK_CHECK)
    if (mDevice == VK_NULL_HANDLE) {
        SPDLOG_CRITICAL("Vulkan: the device could not be created (see the errors above)");
        abort();
    }
    // Texture id 0 is no texture (as OpenGL's name 0: tiles not yet given one sample the dummy texture)
    mTextures.resize(1);
    // fb 0 represents the swapchain
    CreateFramebuffer();
    RebuildScreenFramebuffer();
}

void GfxRenderingAPIVK::UnloadShader(ShaderProgram* oldPrg) {
}

void GfxRenderingAPIVK::LoadShader(ShaderProgram* newPrg) {
    mShaderProgram = (ShaderProgramVK*)newPrg;
}

void GfxRenderingAPIVK::ClearShaderCache() {
    for (auto& [key, prg] : mShaderProgramPool) {
        prg.markedForDeletion = true;
    }
}

BuiltProgramVK GfxRenderingAPIVK::BuildProgram(uint64_t shaderId0, uint64_t shaderId1, bool threePoint, bool srgb) {
    CCFeatures cc_features;
    gfx_cc_get_features(shaderId0, shaderId1, &cc_features);

    BuiltProgramVK built;
    built.threePoint = threePoint;
    built.srgb = srgb;
    built.useAlphaBlend = cc_features.opt_alpha;

    // Vertex layout: the interpreter's packing order, at the template's fixed locations
    uint32_t offset = 0;
    uint32_t count = 0;
    auto attrib = [&](uint32_t location, uint32_t floats) {
        static const VkFormat formats[5] = { VK_FORMAT_UNDEFINED, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
                                             VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT };
        built.attribs[count++] = { location, 0, formats[floats], offset };
        offset += floats * sizeof(float);
    };
    attrib(0, 4);
    for (uint32_t i = 0; i < 2; i++) {
        if (cc_features.usedTextures[i]) {
            attrib(1 + i * 3, 2);
            if (cc_features.clamp[i][0]) {
                attrib(2 + i * 3, 1);
            }
            if (cc_features.clamp[i][1]) {
                attrib(3 + i * 3, 1);
            }
        }
    }
    if (cc_features.opt_fog) {
        attrib(7, 4);
    }
    if (cc_features.opt_grayscale) {
        attrib(8, 4);
    }
    for (int i = 0; i < cc_features.numInputs; i++) {
        attrib(9 + i, cc_features.opt_alpha ? 4 : 3);
    }
    built.attribCount = count;
    built.numFloats = (uint8_t)(offset / sizeof(float));

    const std::string vsSource = BuildShaderSource(cc_features, true, threePoint, srgb);
    const std::string fsSource = BuildShaderSource(cc_features, false, threePoint, srgb);
    if (const char* dumpDir = getenv("PORT_VK_DUMP_SHADERS")) { // debug: the generated GLSL, one file per shader
        char path[1024];
        snprintf(path, sizeof path, "%s/%016llx_%016llx.glsl", dumpDir, (unsigned long long)shaderId0,
                 (unsigned long long)shaderId1);
        if (FILE* f = fopen(path, "w")) {
            fprintf(f, "// vertex\n%s\n// fragment\n%s\n", vsSource.c_str(), fsSource.c_str());
            fclose(f);
        }
    }
    const std::vector<uint32_t> vsSpirv = CompileGlslToSpirv(mShaderCache, vsSource, true);
    const std::vector<uint32_t> fsSpirv = CompileGlslToSpirv(mShaderCache, fsSource, false);
    if (vsSpirv.empty() || fsSpirv.empty()) {
        SPDLOG_ERROR("Vulkan: shader {:016x}/{:016x} does not compile, its draws are skipped", shaderId0, shaderId1);
        return built;
    }
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = vsSpirv.size() * sizeof(uint32_t);
    smci.pCode = vsSpirv.data();
    VK_CHECK(vkCreateShaderModule(mDevice, &smci, nullptr, &built.vs));
    smci.codeSize = fsSpirv.size() * sizeof(uint32_t);
    smci.pCode = fsSpirv.data();
    VK_CHECK(vkCreateShaderModule(mDevice, &smci, nullptr, &built.fs));
    return built;
}

// For a program never handed to the render thread's command buffers (a warm-up build for other settings)
void GfxRenderingAPIVK::DestroyBuiltProgram(BuiltProgramVK& built) {
    for (auto& [key, pipeline] : built.pipelines) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(mDevice, pipeline, nullptr);
        }
    }
    built.pipelines.clear();
    for (VkShaderModule module : { built.vs, built.fs }) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(mDevice, module, nullptr);
        }
    }
    built.vs = VK_NULL_HANDLE;
    built.fs = VK_NULL_HANDLE;
}

ShaderProgram* GfxRenderingAPIVK::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    const auto statStart = std::chrono::steady_clock::now();
    const bool threePoint = mCurrentFilterMode == FILTER_THREE_POINT;
    const auto key = std::make_pair(shaderId0, shaderId1);

    // a program the warm-up built (or is building: wait for it, it is further along than a new build here)
    BuiltProgramVK built;
    bool prebuilt = false;
    {
        std::unique_lock<std::mutex> lock(mPrebuildMutex);
        mPrebuildCond.wait(lock, [&] { return mPrebuilding.count(key) == 0; });
        auto it = mPrebuilt.find(key);
        if (it != mPrebuilt.end()) {
            built = std::move(it->second);
            mPrebuilt.erase(it);
            prebuilt = true;
        } else {
            mPrebuildSkip.insert(key);
        }
    }
    if (prebuilt && (built.threePoint != threePoint || built.srgb != mSrgbMode)) {
        DestroyBuiltProgram(built); // built for other settings (the filter changed since)
        prebuilt = false;
    }
    if (!prebuilt) {
        built = BuildProgram(shaderId0, shaderId1, threePoint, mSrgbMode);
    }

    CCFeatures cc_features;
    gfx_cc_get_features(shaderId0, shaderId1, &cc_features);
    ShaderProgramVK* prg = &mShaderProgramPool[key];
    prg->shader_id0 = shaderId0;
    prg->shader_id1 = shaderId1;
    prg->usedTextures[0] = cc_features.usedTextures[0];
    prg->usedTextures[1] = cc_features.usedTextures[1];
    prg->usedTextures[2] = cc_features.used_masks[0];
    prg->usedTextures[3] = cc_features.used_masks[1];
    prg->usedTextures[4] = cc_features.used_blend[0];
    prg->usedTextures[5] = cc_features.used_blend[1];
    prg->numInputs = cc_features.numInputs;
    prg->markedForDeletion = false;
    prg->built = std::move(built);

    LoadShader((ShaderProgram*)prg);
    sStat.programs++;
    sStat.prebuiltPrograms += prebuilt ? 1 : 0;
    sStat.programMs += MsSince(statStart);
    return (ShaderProgram*)prg;
}

ShaderProgram* GfxRenderingAPIVK::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto it = mShaderProgramPool.find(std::make_pair(shaderId0, shaderId1));
    if (it == mShaderProgramPool.end()) {
        return nullptr;
    }
    if (it->second.markedForDeletion) {
        DestroyShaderProgram(it->second);
        if (mShaderProgram == &it->second) {
            mShaderProgram = nullptr;
        }
        mShaderProgramPool.erase(it);
        return nullptr;
    }
    return (ShaderProgram*)&it->second;
}

// The pipelines and modules may be in command buffers recorded this frame: freed once the frame is done
void GfxRenderingAPIVK::DestroyShaderProgram(ShaderProgramVK& prg) {
    FrameSlot& slot = mFrames[mFrameSlot];
    for (auto& [key, pipeline] : prg.built.pipelines) {
        if (pipeline != VK_NULL_HANDLE) {
            slot.garbagePipelines.push_back(pipeline);
        }
    }
    prg.built.pipelines.clear();
    for (VkShaderModule module : { prg.built.vs, prg.built.fs }) {
        if (module != VK_NULL_HANDLE) {
            slot.garbageModules.push_back(module);
        }
    }
    prg.built.vs = VK_NULL_HANDLE;
    prg.built.fs = VK_NULL_HANDLE;
    for (auto& fb : mFramebuffers) {
        fb.mLastPipeline = VK_NULL_HANDLE;
    }
}

void GfxRenderingAPIVK::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    ShaderProgramVK* p = (ShaderProgramVK*)prg;
    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

uint32_t GfxRenderingAPIVK::NewTexture() {
    mTextures.resize(mTextures.size() + 1);
    return (uint32_t)(mTextures.size() - 1);
}

void GfxRenderingAPIVK::DestroyTextureData(TextureDataVK& tex, bool deferred) {
    if (tex.isSwapchainAlias) {
        return;
    }
    if (tex.imguiSet != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(tex.imguiSet);
        tex.imguiSet = VK_NULL_HANDLE;
    }
    if (tex.imguiSetNearest != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(tex.imguiSetNearest);
        tex.imguiSetNearest = VK_NULL_HANDLE;
    }
    if (tex.view != VK_NULL_HANDLE) {
        if (deferred) {
            FrameSlot& slot = mFrames[mFrameSlot];
            slot.garbageViews.push_back(tex.view);
            slot.garbageImages.push_back({ tex.image, tex.memory });
        } else {
            vkDestroyImageView(mDevice, tex.view, nullptr);
            vkDestroyImage(mDevice, tex.image, nullptr);
            vkFreeMemory(mDevice, tex.memory, nullptr);
        }
    }
    tex.view = VK_NULL_HANDLE;
    tex.image = VK_NULL_HANDLE;
    tex.memory = VK_NULL_HANDLE;
}

void GfxRenderingAPIVK::DeleteTexture(uint32_t texId) {
    if (texId < mTextures.size()) {
        DestroyTextureData(mTextures[texId], true);
    }
}

void GfxRenderingAPIVK::SelectTexture(int tile, uint32_t textureId) {
    mCurrentTile = tile;
    mCurrentTextureIds[tile] = textureId;
}

void GfxRenderingAPIVK::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }

    TextureDataVK& tex = mTextures[mCurrentTextureIds[mCurrentTile]];

    bool oneShot = !mFrameActive;
    VkCommandBuffer cmd;
    if (oneShot) {
        cmd = BeginOneShot();
    } else {
        EnsureUploadCmd();
        cmd = mUploadCmd;
    }

    // PORT: a texture uploaded again during a frame gets a new image (the old one is freed once the frame is done):
    // the uploads go to the GPU ahead of the frame's draws, so overwriting the image in place would change what the
    // draws recorded earlier in the frame sample.
    if (tex.image == VK_NULL_HANDLE || tex.width != width || tex.height != height || !oneShot) {
        DestroyTextureData(tex, true);
        CreateImage(width, height, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, tex.image, tex.memory);
        tex.view = CreateImageView(tex.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
        tex.width = width;
        tex.height = height;
        TransitionToGeneral(cmd, tex.image, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    size_t dataSize = (size_t)width * height * 4;
    FrameSlot& slot = mFrames[mFrameSlot];
    if (oneShot) {
        // Outside a frame: stage through the slot's buffer from offset 0
        if (dataSize > slot.stagingCapacity) {
            // The image transition recorded so far is submitted first; EndOneShot waits idle.
            EndOneShot(cmd);
            GrowStaging(slot, dataSize);
            cmd = BeginOneShot();
        }
        memcpy(slot.stagingMapped, rgba32Buf, dataSize);
    } else {
        if (mStagingOffset + dataSize > slot.stagingCapacity) {
            // Out of staging space this frame: grow (deferred destroy keeps the recorded copies valid)
            GrowStaging(slot, mStagingOffset + dataSize);
            cmd = mUploadCmd;
        }
        memcpy(slot.stagingMapped + mStagingOffset, rgba32Buf, dataSize);
    }

    VkBufferImageCopy copy = {};
    copy.bufferOffset = oneShot ? 0 : mStagingOffset;
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(cmd, slot.stagingBuffer, tex.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    FullBarrier(cmd);

    if (oneShot) {
        EndOneShot(cmd);
    } else {
        mStagingOffset += (dataSize + 255) & ~255ull;
    }
}

void GfxRenderingAPIVK::SetSamplerParameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    TextureDataVK& tex = mTextures[mCurrentTextureIds[tile]];
    tex.linear_filtering = linear_filter;
    tex.filtering = !linear_filter ? FILTER_LINEAR : FILTER_THREE_POINT;
    tex.sampler = GetSampler(linear_filter, cms, cmt);
}

void GfxRenderingAPIVK::SetCurrentPrimDepth(float depth) {
    if (depth != mCurrentPrimDepth) {
        mCurrentPrimDepth = depth;
        mPrimDepthDirty = true;
    }
}

void GfxRenderingAPIVK::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = z_upd;
}

void GfxRenderingAPIVK::SetZmodeDecal(bool decal) {
    mCurrentZmodeDecal = decal;
}

void GfxRenderingAPIVK::SetViewport(int x, int y, int width, int height) {
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    // Negative-height viewport (VK_KHR_maintenance1, core 1.1) flips y so the framebuffer ends up with the same
    // orientation as Metal.
    float top = (float)(mRenderTargetHeight - y - height);
    fb.mViewport.x = (float)x;
    fb.mViewport.y = top + (float)height;
    fb.mViewport.width = (float)width;
    fb.mViewport.height = -(float)height;
    fb.mViewport.minDepth = 0.0f;
    fb.mViewport.maxDepth = 1.0f;
    if (fb.mPassActive) {
        vkCmdSetViewport(fb.mCommandBuffer, 0, 1, &fb.mViewport);
    }
}

void GfxRenderingAPIVK::SetScissor(int x, int y, int width, int height) {
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    TextureDataVK& tex = mTextures[fb.mTextureId];
    int32_t sx = std::max(0, std::min<int>(x, tex.width));
    int32_t sy = std::max(0, std::min<int>(mRenderTargetHeight - y - height, tex.height));
    fb.mScissor.offset = { sx, sy };
    fb.mScissor.extent = { (uint32_t)std::max(0, std::min<int>(width, (int)tex.width - sx)),
                           (uint32_t)std::max(0, std::min<int>(height, (int)tex.height - sy)) };
    if (fb.mPassActive) {
        vkCmdSetScissor(fb.mCommandBuffer, 0, 1, &fb.mScissor);
    }
}

void GfxRenderingAPIVK::SetUseAlpha(bool useAlpha) {
    // Blend state comes from the shader's opt_alpha (baked into the pipeline)
}

// The render state packed into a pipeline key: msaa level (bits 0-3), depth buffer (4), depth test (5), depth
// write (6), decal (7). Thread-safe (warm-up workers build pipelines for the states listed in vulkan_pipelines.txt).
VkPipeline GfxRenderingAPIVK::CreatePipeline(const BuiltProgramVK& prg, uint32_t key) {
    if (prg.vs == VK_NULL_HANDLE || prg.fs == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    const uint32_t msaaLevel = key & 15;
    const bool hasDepth = (key >> 4) & 1;
    const bool depthTest = (key >> 5) & 1;
    const bool depthMask = (key >> 6) & 1;
    const bool decal = (key >> 7) & 1;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = prg.vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = prg.fs;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding = { 0, prg.numFloats * (uint32_t)sizeof(float),
                                                VK_VERTEX_INPUT_RATE_VERTEX };
    VkPipelineVertexInputStateCreateInfo vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = prg.attribCount;
    vertexInput.pVertexAttributeDescriptions = prg.attribs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
    };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // The interpreter culls on the CPU; decals get a slope-scaled bias (dynamic, set per draw)
    VkPipelineRasterizationStateCreateInfo raster = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    raster.depthBiasEnable = decal ? VK_TRUE : VK_FALSE;
    raster.depthClampEnable = mHasDepthClamp ? VK_TRUE : VK_FALSE;

    VkPipelineMultisampleStateCreateInfo msaa = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = (VkSampleCountFlagBits)msaaLevel;

    // As the OpenGL and Metal backends: LESS, LEQUAL for decals, ALWAYS (but still writing) without the depth test
    VkPipelineDepthStencilStateCreateInfo depth = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth.depthTestEnable = hasDepth && (depthTest || depthMask) ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = hasDepth && depthMask ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp =
        depthTest ? (decal ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS) : VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (prg.useAlphaBlend) {
        blendAttachment.blendEnable = VK_TRUE;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        if (mHasDualSrcBlend) {
            // PORT: dual-source, as gfx_metal.cpp: RGB = src * a + dst * (1 - a) with a = the second output's alpha;
            // alpha = coverage (the first output's alpha: 1, or 0 for o_invisible draws, which keep the old alpha)
            blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC1_ALPHA;
            blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
            blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        } else {
            blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        }
    }
    VkPipelineColorBlendStateCreateInfo blend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    static const VkDynamicState dynamicStates[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                                     VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamic = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic.dynamicStateCount = 3;
    dynamic.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vertexInput;
    pci.pInputAssemblyState = &inputAssembly;
    pci.pViewportState = &viewportState;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &msaa;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dynamic;
    pci.layout = mPipelineLayout;
    pci.renderPass = GetRenderPass(msaaLevel, hasDepth);
    pci.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateGraphicsPipelines(mDevice, mPipelineCache, 1, &pci, nullptr, &pipeline);
    if (res != VK_SUCCESS) {
        SPDLOG_ERROR("Vulkan: pipeline creation failed ({}) for render state {:x}", (int)res, key);
        pipeline = VK_NULL_HANDLE;
    }
    return pipeline;
}

// The program's pipeline for the current depth/decal/framebuffer state, created on first use (and recorded for the
// warm-up of later runs).
VkPipeline GfxRenderingAPIVK::GetPipelineForState(ShaderProgramVK* prg, FramebufferVK& fb) {
    const uint32_t key = fb.mMsaaLevel | ((fb.mHasDepthBuffer ? 1u : 0) << 4) | ((mCurrentDepthTest ? 1u : 0) << 5) |
                         ((mCurrentDepthMask ? 1u : 0) << 6) | ((mCurrentZmodeDecal ? 1u : 0) << 7);
    auto it = prg->built.pipelines.find(key);
    if (it != prg->built.pipelines.end()) {
        return it->second;
    }
    const auto statStart = std::chrono::steady_clock::now();
    VkPipeline pipeline = CreatePipeline(prg->built, key);
    sStat.pipelines++;
    sStat.pipelineMs += MsSince(statStart);
    prg->built.pipelines[key] = pipeline;
    if (pipeline != VK_NULL_HANDLE) {
        RecordPipeline(prg->shader_id0, prg->shader_id1, key);
    }
    return pipeline;
}

void GfxRenderingAPIVK::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    if (mShaderProgram == nullptr || mShaderProgram->built.vs == VK_NULL_HANDLE || !mFrameActive) {
        return;
    }
    EnsurePassActive(mCurrentFramebuffer);
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    if (!fb.mPassActive) {
        return; // not a render target
    }
    FrameSlot& slot = mFrames[mFrameSlot];
    VkCommandBuffer cmd = fb.mCommandBuffer;

    VkPipeline pipeline = GetPipelineForState(mShaderProgram, fb);
    if (pipeline == VK_NULL_HANDLE) {
        return;
    }
    if (pipeline != fb.mLastPipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        fb.mLastPipeline = pipeline;
    }

    // Decal depth bias (as gfx_metal.cpp: slope-scaled, -2 at 240p)
    float bias = 0.0f;
    if (mCurrentZmodeDecal) {
        const int n64modeFactor = 120;
        const int noVanishFactor = 100;
        switch (mConsoleVariable != nullptr ? mConsoleVariable->GetInteger(CVAR_Z_FIGHTING_MODE, 0) : 0) {
            case 1:
                bias = -1.0f * (float)mRenderTargetHeight / n64modeFactor;
                break;
            case 2:
                bias = -1.0f * (float)mRenderTargetHeight / noVanishFactor;
                break;
            default:
                bias = -2.0f;
        }
    }
    if (bias != fb.mLastDepthBias) {
        vkCmdSetDepthBias(cmd, 0.0f, 0.0f, bias);
        fb.mLastDepthBias = bias;
    }

    // Textures: a new set whenever a binding changes (sets come from the frame slot's pools)
    VkImageView views[SHADER_MAX_TEXTURES];
    VkSampler samplers[SHADER_MAX_TEXTURES];
    bool texChanged = false;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        const TextureDataVK& tex = mTextures[mCurrentTextureIds[i]];
        const bool used = mShaderProgram->usedTextures[i] && tex.view != VK_NULL_HANDLE;
        views[i] = used ? tex.view : mDummyTexture.view;
        samplers[i] = used && tex.sampler != VK_NULL_HANDLE ? tex.sampler : mDummyTexture.sampler;
        texChanged |= views[i] != fb.mLastBoundViews[i] || samplers[i] != fb.mLastBoundSamplers[i];
    }
    if (texChanged) {
        VkDescriptorSet set = AllocateTextureSet();
        if (set == VK_NULL_HANDLE) {
            return;
        }
        VkDescriptorImageInfo imageInfos[SHADER_MAX_TEXTURES];
        VkWriteDescriptorSet writes[SHADER_MAX_TEXTURES] = {};
        for (uint32_t i = 0; i < SHADER_MAX_TEXTURES; i++) {
            imageInfos[i] = { samplers[i], views[i], VK_IMAGE_LAYOUT_GENERAL };
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &imageInfos[i];
        }
        vkUpdateDescriptorSets(mDevice, SHADER_MAX_TEXTURES, writes, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout, 0, 1, &set, 0, nullptr);
        memcpy(fb.mLastBoundViews, views, sizeof(views));
        memcpy(fb.mLastBoundSamplers, samplers, sizeof(samplers));
    }

    VulkanPushConstants pc = {};
    pc.frameCount = mFrameCount;
    pc.noiseScale = mNoiseScale;
    pc.primDepth = mCurrentPrimDepth;
    for (int i = 0; i < 2; i++) {
        pc.textureFiltering[i] = (int32_t)mTextures[mCurrentTextureIds[i]].filtering;
    }
    vkCmdPushConstants(cmd, mPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // Vertex data: the frame slot's ring, grown (old buffer freed once the frame is done) when a frame needs more
    size_t dataSize = sizeof(float) * buf_vbo_len;
    if (mVertexRingOffset + dataSize > mVertexRingCapacity) {
        slot.garbageBuffers.push_back(slot.vertexBuffer);
        slot.garbageMemory.push_back(slot.vertexMemory);
        mVertexRingCapacity = std::max(mVertexRingCapacity * 2, mVertexRingOffset + dataSize);
        CreateBuffer(mVertexRingCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, slot.vertexBuffer,
                     slot.vertexMemory, &slot.vertexMapped);
        slot.vertexCapacity = mVertexRingCapacity;
        mVertexRingOffset = 0;
        SPDLOG_INFO("Vulkan: grew the vertex buffer to {} MB", mVertexRingCapacity / (1024 * 1024));
    }
    memcpy(slot.vertexMapped + mVertexRingOffset, buf_vbo, dataSize);
    VkDeviceSize vbOffset = mVertexRingOffset;
    vkCmdBindVertexBuffers(cmd, 0, 1, &slot.vertexBuffer, &vbOffset);
    mVertexRingOffset += dataSize;

    vkCmdDraw(cmd, (uint32_t)(buf_vbo_num_tris * 3), 1, 0, 0);
}

void GfxRenderingAPIVK::OnResize() {
    mSwapchainNeedsRecreate = true;
}

void GfxRenderingAPIVK::StartFrame() {
    mFrameCount++;
    if (mFrameCount > 150) {
        mFrameCount = 0;
    }
}

void GfxRenderingAPIVK::EnsurePassActive(int fbId) {
    FramebufferVK& fb = mFramebuffers[fbId];
    if (fb.mPassActive) {
        return;
    }

    if (fb.mCommandBuffer == VK_NULL_HANDLE) {
        fb.mCommandBuffer = AcquireCommandBuffer();
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(fb.mCommandBuffer, &bi));

        if (fbId == 0 && !mSwapchainImageUsed[mSwapchainImageIndex]) {
            // First use of this swapchain image: bring it into GENERAL
            TransitionToGeneral(fb.mCommandBuffer, mSwapchainImages[mSwapchainImageIndex], VK_IMAGE_ASPECT_COLOR_BIT);
            mSwapchainImageUsed[mSwapchainImageIndex] = true;
        } else if (fbId == 0) {
            // Coming back from PRESENT_SRC
            VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = mSwapchainImages[mSwapchainImageIndex];
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(fb.mCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
    }

    if (fb.mFramebuffer == VK_NULL_HANDLE || fb.mRenderPass == VK_NULL_HANDLE) {
        return; // not a render target
    }

    VkRenderPassBeginInfo rpbi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass = fb.mRenderPass;
    rpbi.framebuffer = fb.mFramebuffer;
    rpbi.renderArea = { { 0, 0 }, { fb.mWidth, fb.mHeight } };
    vkCmdBeginRenderPass(fb.mCommandBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    fb.mPassActive = true;

    // Dynamic state must be set before the first draw of the pass
    if (fb.mViewport.width == 0.0f && fb.mViewport.height == 0.0f) {
        fb.mViewport = { 0.0f, (float)fb.mHeight, (float)fb.mWidth, -(float)fb.mHeight, 0.0f, 1.0f };
    }
    if (fb.mScissor.extent.width == 0 && fb.mScissor.extent.height == 0) {
        fb.mScissor = { { 0, 0 }, { fb.mWidth, fb.mHeight } };
    }
    vkCmdSetViewport(fb.mCommandBuffer, 0, 1, &fb.mViewport);
    vkCmdSetScissor(fb.mCommandBuffer, 0, 1, &fb.mScissor);

    vkCmdSetDepthBias(fb.mCommandBuffer, 0.0f, 0.0f, 0.0f);
    fb.mLastDepthBias = 0.0f;

    // Rebound in every pass (ImGui and readback submits change the bindings)
    fb.mLastPipeline = VK_NULL_HANDLE;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        fb.mLastBoundViews[i] = VK_NULL_HANDLE;
        fb.mLastBoundSamplers[i] = VK_NULL_HANDLE;
    }
}

void GfxRenderingAPIVK::EndPass(FramebufferVK& fb) {
    if (fb.mPassActive) {
        vkCmdEndRenderPass(fb.mCommandBuffer);
        fb.mPassActive = false;
    }
}

// PORT: framebuffers whose colour texture was just (re)created have undefined contents; the first pass into one clears
// it to black, as an untouched N64 frame buffer would read (see gfx_metal.cpp)
static std::unordered_set<int> sFreshColorFbs;

void GfxRenderingAPIVK::StartDrawToFramebuffer(int fbId, float noiseScale) {
    if (fbId >= (int)mFramebuffers.size()) {
        return;
    }
    FramebufferVK& fb = mFramebuffers[fbId];
    mRenderTargetHeight = (int32_t)mTextures[fb.mTextureId].height;
    mCurrentFramebuffer = fbId;
    if (std::find(mDrawnFramebuffers.begin(), mDrawnFramebuffers.end(), fbId) == mDrawnFramebuffers.end()) {
        mDrawnFramebuffers.push_back(fbId);
    }

    if (noiseScale != 0.0f) {
        mNoiseScale = 1.0f / noiseScale;
    }

    if (mFrameActive && fb.mRenderTarget) {
        EnsurePassActive(fbId);
        static const bool sNoFreshClear = getenv("PORT_NO_FRESH_CLEAR") != nullptr; // debug: old behaviour
        if (sFreshColorFbs.erase(fbId) != 0 && !sNoFreshClear) {
            ClearFramebuffer(true, false);
        }
    }
}

void GfxRenderingAPIVK::ClearFramebuffer(bool color, bool depth) {
    if (!color && !depth) {
        return;
    }
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    if (!fb.mPassActive) {
        return;
    }

    VkClearAttachment clears[2];
    uint32_t count = 0;
    if (color) {
        clears[count].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clears[count].colorAttachment = 0;
        clears[count].clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        count++;
    }
    if (depth && fb.mHasDepthBuffer) {
        clears[count].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clears[count].colorAttachment = 0;
        clears[count].clearValue.depthStencil = { 1.0f, 0 };
        count++;
    }
    if (count == 0) {
        return;
    }
    VkClearRect rect = { { { 0, 0 }, { fb.mWidth, fb.mHeight } }, 0, 1 };
    vkCmdClearAttachments(fb.mCommandBuffer, count, clears, 1, &rect);
}

bool GfxRenderingAPIVK::ClearFramebufferColor(float r, float g, float b, float a) {
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    if (mCurrentFramebuffer == 0 || !fb.mPassActive) {
        return false;
    }
    VkClearAttachment clear = {};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.color = { { r, g, b, a } };
    VkClearRect rect = { { { 0, 0 }, { fb.mWidth, fb.mHeight } }, 0, 1 };
    vkCmdClearAttachments(fb.mCommandBuffer, 1, &clear, 1, &rect);
    return true;
}

void GfxRenderingAPIVK::ClearDepthRegion(int x, int y, int w, int h) {
    FramebufferVK& fb = mFramebuffers[mCurrentFramebuffer];
    if (!fb.mPassActive || !fb.mHasDepthBuffer) {
        return;
    }
    int32_t sy = std::max(0, (int)mRenderTargetHeight - y - h);
    VkClearAttachment clear = {};
    clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear.clearValue.depthStencil = { 1.0f, 0 };
    VkClearRect rect = { { { std::max(0, x), sy },
                           { (uint32_t)std::max(0, std::min(w, (int)fb.mWidth - x)),
                             (uint32_t)std::max(0, std::min(h, (int)fb.mHeight - sy)) } },
                         0,
                         1 };
    if (rect.rect.extent.width == 0 || rect.rect.extent.height == 0) {
        return;
    }
    vkCmdClearAttachments(fb.mCommandBuffer, 1, &clear, 1, &rect);
}

int GfxRenderingAPIVK::CreateFramebuffer() {
    uint32_t textureId = NewTexture();
    mTextures[textureId].sampler = GetSampler(true, G_TX_WRAP, G_TX_WRAP);

    size_t index = mFramebuffers.size();
    mFramebuffers.resize(mFramebuffers.size() + 1);
    FramebufferVK& fb = mFramebuffers.back();
    fb.mTextureId = textureId;
    fb.mMsaaLevel = 1;
    return (int)index;
}

void GfxRenderingAPIVK::UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height, uint32_t msaaLevel,
                                                    bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                                    bool can_extract_depth) {
    if (fbId == 0) {
        // The swapchain tracks the drawable size in NewFrame
        return;
    }
    if (fbId >= (int)mFramebuffers.size()) {
        return;
    }

    FramebufferVK& fb = mFramebuffers[fbId];
    TextureDataVK& tex = mTextures[fb.mTextureId];

    width = std::max(width, 1u);
    height = std::max(height, 1u);
    msaaLevel = std::max(msaaLevel, 1u);
    while (msaaLevel > 1 && (mDeviceProps.limits.framebufferColorSampleCounts & msaaLevel) == 0) {
        msaaLevel--;
    }

    bool diff = fb.mWidth != width || fb.mHeight != height || fb.mMsaaLevel != msaaLevel ||
                fb.mHasDepthBuffer != has_depth_buffer || fb.mRenderTarget != render_target;
    if (!diff) {
        return;
    }

    vkDeviceWaitIdle(mDevice);

    // PORT: the old picture survives the resize (fast/backends/resize_keep.h): the old colour image is kept until it
    // has been blitted, scaled, into the new one (both GENERAL, both created with the transfer usages)
    const int ow = (int)fb.mWidth, oh = (int)fb.mHeight;
    TextureDataVK oldTex = {};
    bool keep = render_target && fb.mRenderTarget && PortResizeKeepEnabled() && tex.image != VK_NULL_HANDLE &&
                !tex.isSwapchainAlias && ow > 1 && oh > 1 && (fb.mWidth != width || fb.mHeight != height);
    if (keep) {
        oldTex = tex;
        oldTex.imguiSet = VK_NULL_HANDLE;
        oldTex.imguiSetNearest = VK_NULL_HANDLE;
        if (tex.imguiSet != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(tex.imguiSet);
            tex.imguiSet = VK_NULL_HANDLE;
        }
        if (tex.imguiSetNearest != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(tex.imguiSetNearest);
            tex.imguiSetNearest = VK_NULL_HANDLE;
        }
        tex.view = VK_NULL_HANDLE;
        tex.image = VK_NULL_HANDLE;
        tex.memory = VK_NULL_HANDLE;
    }

    // Color texture (sampled + render target + blit source/destination)
    DestroyTextureData(tex, false);
    CreateImage(width, height, VK_SAMPLE_COUNT_1_BIT, mSwapchainFormat,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                tex.image, tex.memory);
    tex.view = CreateImageView(tex.image, mSwapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    tex.width = width;
    tex.height = height;
    if (render_target && !keep) {
        sFreshColorFbs.insert(fbId);
    } else if (keep) {
        sFreshColorFbs.erase(fbId);
    }

    VkCommandBuffer cmd = BeginOneShot();
    TransitionToGeneral(cmd, tex.image, VK_IMAGE_ASPECT_COLOR_BIT);
    if (keep) {
        FullBarrier(cmd);
        VkClearColorValue black = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, tex.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
        FullBarrier(cmd);
        int src[4], dst[4];
        if (PortResizeKeepClipped(ow, oh, (int)width, (int)height, src, dst)) {
            VkImageBlit blit = {};
            blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blit.srcOffsets[0] = { src[0], src[1], 0 };
            blit.srcOffsets[1] = { src[2], src[3], 1 };
            blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blit.dstOffsets[0] = { dst[0], dst[1], 0 };
            blit.dstOffsets[1] = { dst[2], dst[3], 1 };
            vkCmdBlitImage(cmd, oldTex.image, VK_IMAGE_LAYOUT_GENERAL, tex.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                           VK_FILTER_LINEAR);
        }
        FullBarrier(cmd);
    }

    auto destroyTarget = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(mDevice, view, nullptr);
            vkDestroyImage(mDevice, img, nullptr);
            vkFreeMemory(mDevice, mem, nullptr);
            view = VK_NULL_HANDLE;
            img = VK_NULL_HANDLE;
            mem = VK_NULL_HANDLE;
        }
    };
    destroyTarget(fb.mDepthImage, fb.mDepthMemory, fb.mDepthView);
    destroyTarget(fb.mMsaaImage, fb.mMsaaMemory, fb.mMsaaView);
    destroyTarget(fb.mMsaaDepthImage, fb.mMsaaDepthMemory, fb.mMsaaDepthView);

    if (has_depth_buffer) {
        CreateImage(width, height, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, fb.mDepthImage,
                    fb.mDepthMemory);
        fb.mDepthView = CreateImageView(fb.mDepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
        TransitionToGeneral(cmd, fb.mDepthImage, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    if (msaaLevel > 1) {
        CreateImage(width, height, (VkSampleCountFlagBits)msaaLevel, mSwapchainFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, fb.mMsaaImage,
                    fb.mMsaaMemory);
        fb.mMsaaView = CreateImageView(fb.mMsaaImage, mSwapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        TransitionToGeneral(cmd, fb.mMsaaImage, VK_IMAGE_ASPECT_COLOR_BIT);
        if (has_depth_buffer) {
            CreateImage(width, height, (VkSampleCountFlagBits)msaaLevel, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, fb.mMsaaDepthImage, fb.mMsaaDepthMemory);
            fb.mMsaaDepthView = CreateImageView(fb.mMsaaDepthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
            TransitionToGeneral(cmd, fb.mMsaaDepthImage, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
    }
    EndOneShot(cmd);
    if (keep) {
        DestroyTextureData(oldTex, false); // EndOneShot waited for the queue
    }

    fb.mWidth = width;
    fb.mHeight = height;
    fb.mMsaaLevel = msaaLevel;
    fb.mHasDepthBuffer = has_depth_buffer;
    fb.mRenderTarget = render_target;

    if (fb.mFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(mDevice, fb.mFramebuffer, nullptr);
        fb.mFramebuffer = VK_NULL_HANDLE;
    }

    if (render_target) {
        fb.mRenderPass = GetRenderPass(msaaLevel, has_depth_buffer);
        std::vector<VkImageView> attachments;
        if (msaaLevel > 1) {
            attachments.push_back(fb.mMsaaView);
            attachments.push_back(tex.view);
            if (has_depth_buffer) {
                attachments.push_back(fb.mMsaaDepthView);
            }
        } else {
            attachments.push_back(tex.view);
            if (has_depth_buffer) {
                attachments.push_back(fb.mDepthView);
            }
        }
        VkFramebufferCreateInfo fci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fci.renderPass = fb.mRenderPass;
        fci.attachmentCount = (uint32_t)attachments.size();
        fci.pAttachments = attachments.data();
        fci.width = width;
        fci.height = height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(mDevice, &fci, nullptr, &fb.mFramebuffer));
    } else {
        fb.mRenderPass = VK_NULL_HANDLE;
    }
}

// PORT: the copy goes into streamFb's command buffer (submitted in framebuffer-id order), between what was drawn into
// it so far and what comes next
void GfxRenderingAPIVK::CopyFramebufferInStream(int streamFb, int dstFb, int srcFb, int w, int h) {
    if (streamFb >= (int)mFramebuffers.size() || dstFb >= (int)mFramebuffers.size() ||
        srcFb >= (int)mFramebuffers.size() || !mFrameActive) {
        return;
    }
    TextureDataVK& srcTex = mTextures[mFramebuffers[srcFb].mTextureId];
    TextureDataVK& dstTex = mTextures[mFramebuffers[dstFb].mTextureId];
    if (srcTex.image == VK_NULL_HANDLE || dstTex.image == VK_NULL_HANDLE) {
        return;
    }
    EnsurePassActive(streamFb);
    FramebufferVK& stream = mFramebuffers[streamFb];
    EndPass(stream);
    FullBarrier(stream.mCommandBuffer);
    VkImageCopy copy = {};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.extent = { std::min<uint32_t>({ (uint32_t)w, srcTex.width, dstTex.width }),
                    std::min<uint32_t>({ (uint32_t)h, srcTex.height, dstTex.height }), 1 };
    vkCmdCopyImage(stream.mCommandBuffer, srcTex.image, VK_IMAGE_LAYOUT_GENERAL, dstTex.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &copy);
    // with MSAA the picture lives in the multisampled attachment (loaded by every pass, resolved into the texture)
    const FramebufferVK& srcFbo = mFramebuffers[srcFb];
    const FramebufferVK& dstFbo = mFramebuffers[dstFb];
    if (srcFbo.mMsaaLevel > 1 && srcFbo.mMsaaLevel == dstFbo.mMsaaLevel && srcFbo.mMsaaImage != VK_NULL_HANDLE &&
        dstFbo.mMsaaImage != VK_NULL_HANDLE && srcTex.width == dstTex.width && srcTex.height == dstTex.height) {
        VkImageCopy ms = copy;
        ms.extent = { srcTex.width, srcTex.height, 1 };
        vkCmdCopyImage(stream.mCommandBuffer, srcFbo.mMsaaImage, VK_IMAGE_LAYOUT_GENERAL, dstFbo.mMsaaImage,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &ms);
    }
    FullBarrier(stream.mCommandBuffer);
    EnsurePassActive(streamFb);
}

void GfxRenderingAPIVK::CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                        int dstY0, int dstX1, int dstY1) {
    if (fbSrcId >= (int)mFramebuffers.size() || fbDstId >= (int)mFramebuffers.size() || !mFrameActive) {
        return;
    }
    FramebufferVK& src = mFramebuffers[fbSrcId];
    TextureDataVK& srcTex = mTextures[src.mTextureId];
    TextureDataVK& dstTex = mTextures[mFramebuffers[fbDstId].mTextureId];
    if (srcTex.image == VK_NULL_HANDLE || dstTex.image == VK_NULL_HANDLE || src.mCommandBuffer == VK_NULL_HANDLE) {
        return;
    }

    bool wasActive = src.mPassActive;
    EndPass(src);
    FullBarrier(src.mCommandBuffer);

    VkImageBlit blit = {};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[0] = { srcX0, srcY0, 0 };
    blit.srcOffsets[1] = { srcX1, srcY1, 1 };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstOffsets[0] = { dstX0, dstY0, 0 };
    blit.dstOffsets[1] = { dstX1, dstY1, 1 };
    vkCmdBlitImage(src.mCommandBuffer, srcTex.image, VK_IMAGE_LAYOUT_GENERAL, dstTex.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &blit, VK_FILTER_NEAREST);
    FullBarrier(src.mCommandBuffer);

    if (wasActive) {
        EnsurePassActive(fbSrcId);
    }
}

void GfxRenderingAPIVK::ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) {
    if (fbIdSrc >= (int)mFramebuffers.size() || fbIdTarget >= (int)mFramebuffers.size() || !mFrameActive) {
        return;
    }
    FramebufferVK& src = mFramebuffers[fbIdSrc];
    TextureDataVK& srcTex = mTextures[src.mTextureId];
    TextureDataVK& dstTex = mTextures[mFramebuffers[fbIdTarget].mTextureId];

    if (srcTex.width != dstTex.width || srcTex.height != dstTex.height) {
        return;
    }

    // Ending the src render pass executes its MSAA resolve into the sampled texture; then copy that into the target.
    VkCommandBuffer cmd;
    if (fbIdTarget != 0) {
        if (src.mCommandBuffer == VK_NULL_HANDLE) {
            return;
        }
        EndPass(src);
        cmd = src.mCommandBuffer;
    } else {
        EnsurePassActive(0);
        EndPass(mFramebuffers[0]);
        cmd = mFramebuffers[0].mCommandBuffer;
    }
    FullBarrier(cmd);
    VkImageCopy copy = {};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.extent = { srcTex.width, srcTex.height, 1 };
    vkCmdCopyImage(cmd, srcTex.image, VK_IMAGE_LAYOUT_GENERAL, dstTex.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    FullBarrier(cmd);
    if (fbIdTarget == 0) {
        EnsurePassActive(0);
    }
}

// Submits the framebuffer's command buffer (preceded by pending uploads) and waits for completion. Used by the
// synchronous readback paths: a render-to-texture target read back after each pass (RttReadback) goes out here pass by
// pass, in order on the one queue, so Vulkan needs no SetFramebufferReadback (the Metal backend uses it to keep such
// passes off its main queue, where they ran after later ones).
void GfxRenderingAPIVK::SubmitFramebufferAndWait(int fbId) {
    FramebufferVK& fb = mFramebuffers[fbId];
    EndPass(fb);
    VK_CHECK(vkEndCommandBuffer(fb.mCommandBuffer));

    std::vector<VkCommandBuffer> cmds;
    if (mUploadCmd != VK_NULL_HANDLE) {
        VK_CHECK(vkEndCommandBuffer(mUploadCmd));
        cmds.push_back(mUploadCmd);
        mUploadCmd = VK_NULL_HANDLE;
        mStagingOffset = 0;
    }
    cmds.push_back(fb.mCommandBuffer);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = (uint32_t)cmds.size();
    si.pCommandBuffers = cmds.data();
    const auto statStart = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(mQueue, 1, &si, mReadbackFence));
    VK_CHECK(vkWaitForFences(mDevice, 1, &mReadbackFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(mDevice, 1, &mReadbackFence));
    sStat.syncSubmits++;
    sStat.syncMs += MsSince(statStart);

    fb.mCommandBuffer = VK_NULL_HANDLE;
    fb.mPassActive = false;
    auto it = std::find(mDrawnFramebuffers.begin(), mDrawnFramebuffers.end(), fbId);
    if (it != mDrawnFramebuffers.end()) {
        mDrawnFramebuffers.erase(it);
    }
}

// Converts 8-bit pixels (BGRA or RGBA, as the colour targets' format) to the game's RGBA5551.
static void ConvertToRgba16(const uint8_t* px, uint32_t count, bool bgr, uint16_t* out) {
    for (uint32_t i = 0; i < count; i++) {
        uint8_t c0 = px[i * 4 + 0];
        uint8_t g = px[i * 4 + 1];
        uint8_t c2 = px[i * 4 + 2];
        uint8_t a = px[i * 4 + 3];
        uint8_t r = bgr ? c2 : c0;
        uint8_t b = bgr ? c0 : c2;
        out[i] = ((r >> 3) << 11) | ((g >> 3) << 6) | ((b >> 3) << 1) | (a >> 7);
    }
}

void GfxRenderingAPIVK::InitializeFramebuffer(int fbId, uint32_t width, uint32_t height, const uint8_t* rgba) {
    auto& texture = mTextures[mFramebuffers.at(fbId).mTextureId];
    VkBuffer staging;
    VkDeviceMemory memory;
    uint8_t* mapped;
    const size_t size = (size_t)width * height * 4;
    CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, memory, &mapped);
    memcpy(mapped, rgba, size);
    if (mSwapchainFormat == VK_FORMAT_B8G8R8A8_UNORM) {
        for (size_t i = 0; i < size; i += 4) std::swap(mapped[i], mapped[i + 2]);
    }
    auto command = BeginOneShot();
    FullBarrier(command);
    VkBufferImageCopy copy = {};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(command, staging, texture.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    FullBarrier(command);
    EndOneShot(command);
    vkUnmapMemory(mDevice, memory);
    vkDestroyBuffer(mDevice, staging, nullptr);
    vkFreeMemory(mDevice, memory, nullptr);
    sFreshColorFbs.erase(fbId);
}

void GfxRenderingAPIVK::ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) {
    if (fbId >= (int)mFramebuffers.size()) {
        return;
    }
    FramebufferVK& fb = mFramebuffers[fbId];
    TextureDataVK& tex = mTextures[fb.mTextureId];
    const bool bgr = mSwapchainFormat == VK_FORMAT_B8G8R8A8_UNORM;

    if (fbId == 0) {
        // Deferred path (mirrors Metal): serve the pixels of the last frame that asked, request a copy at the end of
        // this frame.
        if (mScreenReadbackDataReady) {
            FrameSlot& slot = mFrames[mScreenReadbackSlot];
            if (slot.fenceInFlight) {
                VK_CHECK(vkWaitForFences(mDevice, 1, &slot.fence, VK_TRUE, UINT64_MAX));
            }
            const uint8_t* px = ReadbackData(mScreenReadback);
            uint32_t count = mScreenReadbackWidth * mScreenReadbackHeight;
            // PORT: keep the full 8-bit pixels too, for screenshots (BGRA, as gfx_metal.cpp)
            if (bgr) {
                gPortScreenBgra.assign(px, px + (size_t)count * 4);
            } else {
                gPortScreenBgra.resize((size_t)count * 4);
                uint8_t* out = gPortScreenBgra.data();
                for (uint32_t i = 0; i < count; i++, out += 4) {
                    out[0] = px[i * 4 + 2];
                    out[1] = px[i * 4 + 1];
                    out[2] = px[i * 4 + 0];
                    out[3] = px[i * 4 + 3];
                }
            }
            gPortScreenW = mScreenReadbackWidth;
            gPortScreenH = mScreenReadbackHeight;
            ConvertToRgba16(px, std::min(count, width * height), bgr, rgba16Buf);
            mScreenReadbackDataReady = false;
        }
        mScreenReadbackRequested = true;
        mScreenReadbackWidth = std::min(width, mSwapchainExtent.width);
        mScreenReadbackHeight = std::min(height, mSwapchainExtent.height);
        return;
    }

    // Offscreen framebuffer: synchronous
    if (!mFrameActive || fb.mCommandBuffer == VK_NULL_HANDLE || tex.image == VK_NULL_HANDLE) {
        return; // PORT: nothing drawn into it this frame: leave the buffer as is
    }

    EnsureReadbackBuffer(mReadback, (size_t)width * height * 4);
    EndPass(fb);
    FullBarrier(fb.mCommandBuffer);
    VkBufferImageCopy copy = {};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { std::min(width, tex.width), std::min(height, tex.height), 1 };
    copy.bufferRowLength = width;
    vkCmdCopyImageToBuffer(fb.mCommandBuffer, tex.image, VK_IMAGE_LAYOUT_GENERAL, mReadback.buffer, 1, &copy);
    FullBarrier(fb.mCommandBuffer);
    SubmitFramebufferAndWait(fbId);

    ConvertToRgba16(ReadbackData(mReadback), width * height, bgr, rgba16Buf);
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIVK::GetPixelDepth(int fbId, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    if (fbId >= (int)mFramebuffers.size() || coordinates.empty() || !mFrameActive) {
        return res;
    }
    FramebufferVK& fb = mFramebuffers[fbId];
    if (fb.mDepthImage == VK_NULL_HANDLE || fb.mCommandBuffer == VK_NULL_HANDLE) {
        for (const auto& coord : coordinates) {
            res.emplace(coord, 0);
        }
        return res;
    }

    EnsureReadbackBuffer(mReadback, coordinates.size() * sizeof(float));

    bool wasActive = fb.mPassActive;
    EndPass(fb);
    FullBarrier(fb.mCommandBuffer);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(coordinates.size());
    size_t i = 0;
    for (const auto& coord : coordinates) {
        VkBufferImageCopy copy = {};
        copy.bufferOffset = i * sizeof(float);
        copy.imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
        int32_t cx = std::clamp((int32_t)coord.first, 0, (int32_t)fb.mWidth - 1);
        int32_t cy = std::clamp((int32_t)(fb.mHeight - 1 - coord.second), 0, (int32_t)fb.mHeight - 1);
        copy.imageOffset = { cx, cy, 0 };
        copy.imageExtent = { 1, 1, 1 };
        regions.push_back(copy);
        i++;
    }
    vkCmdCopyImageToBuffer(fb.mCommandBuffer, fb.mDepthImage, VK_IMAGE_LAYOUT_GENERAL, mReadback.buffer,
                           (uint32_t)regions.size(), regions.data());
    FullBarrier(fb.mCommandBuffer);
    SubmitFramebufferAndWait(fbId);

    const float* depthValues = (const float*)ReadbackData(mReadback);
    i = 0;
    for (const auto& coord : coordinates) {
        res.emplace(coord, (uint16_t)(depthValues[i++] * 65532.0f));
    }

    if (wasActive) {
        // Caller may keep drawing to this framebuffer afterwards
        StartDrawToFramebuffer(fbId, 0.0f);
    }
    return res;
}

void GfxRenderingAPIVK::EndFrame() {
    if (!mFrameActive) {
        return;
    }
    FrameSlot& slot = mFrames[mFrameSlot];

    std::vector<VkCommandBuffer> cmds;
    if (mUploadCmd != VK_NULL_HANDLE) {
        VK_CHECK(vkEndCommandBuffer(mUploadCmd));
        cmds.push_back(mUploadCmd);
        mUploadCmd = VK_NULL_HANDLE;
    }

    std::sort(mDrawnFramebuffers.begin(), mDrawnFramebuffers.end());
    for (int fbId : mDrawnFramebuffers) {
        if (fbId == 0) {
            continue;
        }
        FramebufferVK& fb = mFramebuffers[fbId];
        if (fb.mCommandBuffer == VK_NULL_HANDLE) {
            continue;
        }
        EndPass(fb);
        VK_CHECK(vkEndCommandBuffer(fb.mCommandBuffer));
        cmds.push_back(fb.mCommandBuffer);
    }

    // Screen framebuffer last
    EnsurePassActive(0);
    FramebufferVK& screen = mFramebuffers[0];
    EndPass(screen);

    // PORT: an in-between frame (frame interpolation) leaves the readback request for the next real frame
    if (mScreenReadbackRequested && !gPortReplayFrame && mScreenReadbackWidth > 0 && mScreenReadbackHeight > 0) {
        EnsureReadbackBuffer(mScreenReadback, (size_t)mScreenReadbackWidth * mScreenReadbackHeight * 4);
        FullBarrier(screen.mCommandBuffer);
        VkBufferImageCopy copy = {};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent = { mScreenReadbackWidth, mScreenReadbackHeight, 1 };
        vkCmdCopyImageToBuffer(screen.mCommandBuffer, mSwapchainImages[mSwapchainImageIndex], VK_IMAGE_LAYOUT_GENERAL,
                               mScreenReadback.buffer, 1, &copy);
        mScreenReadbackRequested = false;
        mScreenReadbackDataReady = true;
        mScreenReadbackSlot = mFrameSlot;
    }

    // GENERAL -> PRESENT_SRC for the swapchain image
    {
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = mSwapchainImages[mSwapchainImageIndex];
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(screen.mCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    VK_CHECK(vkEndCommandBuffer(screen.mCommandBuffer));
    cmds.push_back(screen.mCommandBuffer);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &slot.imageAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = (uint32_t)cmds.size();
    si.pCommandBuffers = cmds.data();
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &slot.renderFinished;
    VK_CHECK(vkQueueSubmit(mQueue, 1, &si, slot.fence));
    slot.fenceInFlight = true;

    VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &slot.renderFinished;
    pi.swapchainCount = 1;
    pi.pSwapchains = &mSwapchain;
    pi.pImageIndices = &mSwapchainImageIndex;
    VkResult present = vkQueuePresentKHR(mQueue, &pi);
    if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR) {
        mSwapchainNeedsRecreate = true;
    } else if (present != VK_SUCCESS) {
        SPDLOG_ERROR("Vulkan: vkQueuePresentKHR failed ({})", (int)present);
    }

    // Reset per-framebuffer recording state
    for (auto& fb : mFramebuffers) {
        fb.mCommandBuffer = VK_NULL_HANDLE;
        fb.mPassActive = false;
        fb.mLastPipeline = VK_NULL_HANDLE;
        fb.mLastDepthBias = 0.0f;
        memset(&fb.mViewport, 0, sizeof(fb.mViewport));
        memset(&fb.mScissor, 0, sizeof(fb.mScissor));
    }
    mDrawnFramebuffers.clear();
    mFrameActive = false;
    mFrameSlot = (mFrameSlot + 1) % VK_FRAMES_IN_FLIGHT;

    // the pipeline cache goes to disk every few seconds while new pipelines are created (not while workers add to it;
    // the game exits with _exit, so there is no saving at shutdown)
    if (mPipelineCacheDirty && MsSince(mPipelineCacheSaved) > 5000) {
        std::unique_lock<std::mutex> lock(mPrebuildMutex);
        if (mPrebuildRunning == 0) {
            lock.unlock();
            SavePipelineCache();
        }
    }

    static int sStatFrames = 0;
    if (sStats && ++sStatFrames == 60) {
        fprintf(stderr,
                "[vkstats] 60 frames: %d programs (%d prebuilt) %.0f ms, %d pipelines %.0f ms, %d sync submits %.0f "
                "ms, fence waits %.0f ms, acquire %.0f ms, %d command buffers (%d new; pools hold %zu + %zu)\n",
                sStat.programs, sStat.prebuiltPrograms, sStat.programMs, sStat.pipelines, sStat.pipelineMs, sStat.syncSubmits, sStat.syncMs,
                sStat.fenceMs, sStat.acquireMs, sStat.commandBuffers, sStat.commandBuffersAllocated,
                mFrames[0].commandBuffers.size(), mFrames[VK_FRAMES_IN_FLIGHT - 1].commandBuffers.size());
        sStat = {};
        sStatFrames = 0;
    }
}

void GfxRenderingAPIVK::FinishRender() {
}

void* GfxRenderingAPIVK::GetFramebufferTextureId(int fbId) {
    return GetTextureById(mFramebuffers[fbId].mTextureId);
}

// PORT: the same picture with a nearest-magnifying sampler (gfx_rendering_api.h)
void* GfxRenderingAPIVK::GetFramebufferTextureIdNearest(int fbId) {
    TextureDataVK& tex = mTextures[mFramebuffers[fbId].mTextureId];
    if (tex.view == VK_NULL_HANDLE) {
        return nullptr;
    }
    if (tex.imguiSetNearest == VK_NULL_HANDLE) {
        tex.imguiSetNearest = ImGui_ImplVulkan_AddTexture(mImGuiSamplerNearest, tex.view, VK_IMAGE_LAYOUT_GENERAL);
    }
    return (void*)tex.imguiSetNearest;
}

void GfxRenderingAPIVK::SelectTextureFb(int fbId) {
    SelectTexture(0, mFramebuffers[fbId].mTextureId);
}

ImTextureID GfxRenderingAPIVK::GetTextureById(int id) {
    TextureDataVK& tex = mTextures[id];
    if (tex.view == VK_NULL_HANDLE) {
        return (ImTextureID) nullptr;
    }
    if (tex.imguiSet == VK_NULL_HANDLE) {
        tex.imguiSet = ImGui_ImplVulkan_AddTexture(mImGuiSampler, tex.view, VK_IMAGE_LAYOUT_GENERAL);
    }
    return (ImTextureID)tex.imguiSet;
}

void GfxRenderingAPIVK::SetTextureFilter(FilteringMode mode) {
    mCurrentFilterMode = mode;
    gfx_texture_cache_clear();
}

FilteringMode GfxRenderingAPIVK::GetTextureFilter() {
    return mCurrentFilterMode;
}

void GfxRenderingAPIVK::SetSrgbMode() {
    mSrgbMode = true;
}

} // namespace Fast

bool Vulkan_LoadLibrary() {
    if (SDL_Vulkan_LoadLibrary(nullptr)) {
        return true;
    }
#ifdef __APPLE__
    // Homebrew's loader (vulkan-loader + molten-vk) is not on the default dlopen path
    for (const char* path : { "/opt/homebrew/lib/libvulkan.1.dylib", "/usr/local/lib/libvulkan.1.dylib",
                              "/opt/homebrew/lib/libMoltenVK.dylib", "/usr/local/lib/libMoltenVK.dylib" }) {
        if (SDL_Vulkan_LoadLibrary(path)) {
            return true;
        }
    }
#endif
    return false;
}

bool Vulkan_IsSupported() {
    if (getenv("PORT_NO_VULKAN") != nullptr) { // debug: the Vulkan renderer is not offered (and not probed)
        return false;
    }
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        return false;
    }
    bool supported = false;
    std::string why; // PORT_VK_SELFTEST explains a "not available"
    if (!Vulkan_LoadLibrary()) {
        why = std::string("no Vulkan library: ") + SDL_GetError();
    } else {
        why = "no vkCreateInstance";
        // A loader alone is not enough (it is installed with many drivers that have no Vulkan ICD): an instance must
        // come up and list a device
        auto getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
        auto createInstance =
            getInstanceProcAddr ? (PFN_vkCreateInstance)getInstanceProcAddr(nullptr, "vkCreateInstance") : nullptr;
        auto enumerateExtensions =
            getInstanceProcAddr ? (PFN_vkEnumerateInstanceExtensionProperties)getInstanceProcAddr(
                                      nullptr, "vkEnumerateInstanceExtensionProperties")
                                : nullptr;
        if (createInstance != nullptr && enumerateExtensions != nullptr) {
            uint32_t count = 0;
            enumerateExtensions(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> exts(count);
            enumerateExtensions(nullptr, &count, exts.data());
            bool portability = false;
            for (const auto& e : exts) {
                portability |= strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
            }
            const char* portabilityExt = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
            VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
            app.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
            ici.pApplicationInfo = &app;
            if (portability) {
                ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
                ici.enabledExtensionCount = 1;
                ici.ppEnabledExtensionNames = &portabilityExt;
            }
            VkInstance instance = VK_NULL_HANDLE;
            const VkResult res = createInstance(&ici, nullptr, &instance);
            why = "vkCreateInstance failed (" + std::to_string((int)res) + ")";
            if (res == VK_SUCCESS) {
                auto enumerateDevices =
                    (PFN_vkEnumeratePhysicalDevices)getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
                auto destroyInstance = (PFN_vkDestroyInstance)getInstanceProcAddr(instance, "vkDestroyInstance");
                uint32_t devices = 0;
                if (enumerateDevices != nullptr) {
                    enumerateDevices(instance, &devices, nullptr);
                }
                supported = devices > 0;
                why = std::to_string(devices) + " devices";
                if (destroyInstance != nullptr) {
                    destroyInstance(instance, nullptr);
                }
            }
        }
        SDL_Vulkan_UnloadLibrary();
    }
    if (!supported) {
        SPDLOG_INFO("Vulkan: no driver with a device, the Vulkan renderer is not offered");
    }
    if (getenv("PORT_VK_SELFTEST") != nullptr) {
        fprintf(stderr, "[vulkan] driver check: %s (%s)\n", supported ? "available" : "not available", why.c_str());
    }
    return supported;
}

#endif
