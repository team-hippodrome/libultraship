#define NOMINMAX

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <algorithm>
#include <any>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <stack>
#include <chrono>
#include "fast/resource/type/Light.h"

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif
#include "fast/debug/GfxDebugger.h"
#include "fast/types.h"
#include <string>

#include "fast/interpreter.h"
#include "fast/lus_gbi.h"
#include "fast/backends/gfx_window_manager_api.h"
#include "fast/backends/gfx_rendering_api.h"

#include "ship/window/gui/Gui.h"
#include "ship/resource/ResourceManager.h"
#include "ship/utils/Utils.h"
#include "ship/config/ConsoleVariable.h"

#include "libultraship/libultra/os.h"

#include <spdlog/fmt/fmt.h>

#ifdef _WIN32
#include <windows.h>
#endif

std::stack<std::string> currentDir;

#define SEG_ADDR(seg, addr) (addr | (seg << 24) | 1)
#define SUPPORT_CHECK(x) assert(x)

// SCALE_M_N: upscale/downscale M-bit integer to N-bit
#define SCALE_5_8(VAL_) (((VAL_)*0xFF) / 0x1F)
#define SCALE_8_5(VAL_) ((((VAL_) + 4) * 0x1F) / 0xFF)
#define SCALE_4_8(VAL_) ((VAL_)*0x11)
#define SCALE_8_4(VAL_) ((VAL_) / 0x11)
#define SCALE_3_8(VAL_) ((VAL_)*0x24)
#define SCALE_8_3(VAL_) ((VAL_) / 0x24)

// Based off the current set native dimensions or active framebuffer
// PORT PATCH: render-to-texture state (see GfxDpSetColorImage)
bool gPortDrawLog = false; // PORT debug: set by the port for the frame whose draw states are listed

// PORT debug switches, read once: getenv scans the whole environment, and these are checked per command or per frame
#define PORT_ENV_SWITCH(fn, name)                          \
    static bool fn() {                                     \
        static const bool on = getenv(name) != nullptr;    \
        return on;                                         \
    }
PORT_ENV_SWITCH(PortTexLog, "PORT_TEXLOG")
PORT_ENV_SWITCH(PortMtxLog, "PORT_MTXLOG")
PORT_ENV_SWITCH(PortRectLog, "PORT_RECTLOG")
PORT_ENV_SWITCH(PortNoRtt, "PORT_NO_RTT")
PORT_ENV_SWITCH(PortFixedNativeW, "PORT_NATIVE_W")
PORT_ENV_SWITCH(PortTile1Log, "PORT_TILE1LOG")
PORT_ENV_SWITCH(PortTraceFill, "PORT_TRACE_FILL")
PORT_ENV_SWITCH(PortBadTimgCtx, "PORT_BADTIMG_CTX")
#undef PORT_ENV_SWITCH

static const void* sLastTimgCmd = nullptr; // PORT debug: the last G_SETTIMG (for the TMEM 2048 load log)

// PORT: TMEM model. On the N64 a texture load copies bytes into the 4 KB TMEM; tiles then read from any TMEM address,
// also from the middle of an earlier load (Stadium's arena floor loads a 2 KB RGBA16 texture and a 1 KB I8 detail
// texture with one load and reads the second through a tile at TMEM 256), and the data stays valid after the RDRAM
// source is reused. Fast3D only kept two "slots" pointing at RDRAM, so RAM loads are recorded here as TMEM regions
// with a copy of their bytes, and tiles are resolved against them when a texture is imported.
struct TmemRegion {
    uint32_t start, len;           // TMEM byte range
    const uint8_t* raw;            // RDRAM source (for render-target lookup)
    std::vector<uint8_t> data;     // the bytes as loaded (rows compacted)
};
static std::list<TmemRegion> sTmemRegions;
static void TmemForget(uint32_t start, uint32_t len) {
    // A new load overwrites [start, start + len); like real TMEM, the parts of older loads outside it stay readable.
    // (The arena's near floor reads its second tile from TMEM 2048, which holds the tail of the previous frame's last
    // large load.)
    uint32_t end = start + len;
    std::list<TmemRegion> kept;
    for (auto it = sTmemRegions.begin(); it != sTmemRegions.end();) {
        uint32_t oend = it->start + it->len;
        if (it->start < end && start < oend) {
            if (it->start < start) { // the part before the new load
                kept.push_back({ it->start, start - it->start, it->raw,
                                 std::vector<uint8_t>(it->data.begin(), it->data.begin() + (start - it->start)) });
            }
            if (oend > end) { // the part after it
                uint32_t off = end - it->start;
                kept.push_back({ end, oend - end, it->raw ? it->raw + off : nullptr,
                                 std::vector<uint8_t>(it->data.begin() + off, it->data.end()) });
            }
            it = sTmemRegions.erase(it);
        } else {
            ++it;
        }
    }
    // remainders keep their age (they are older than the new load, which is appended after this)
    sTmemRegions.splice(sTmemRegions.begin(), kept);
}
// PORT: the game's C arrays of u16/u32 words (see tools/gen_word_arrays.py) are little-endian on the host; the N64
// (and this TMEM copy) sees them big-endian. Their address ranges are registered at startup by the generated
// wrappers of the game sources. Returns the word size (2 or 4) when src lies in one, and the offset of src from the
// array start.
struct PortWordRange { uintptr_t start, end; int size; };
static std::vector<PortWordRange>& WordRanges() {
    static std::vector<PortWordRange> ranges; // function-local: the registrations run from static constructors
    return ranges;
}
static bool sWordRangesSorted = false;
extern "C" void Port_RegisterWordArray(const void* start, unsigned long bytes, int wordSize) {
    WordRanges().push_back({ (uintptr_t)start, (uintptr_t)start + bytes, wordSize });
    sWordRangesSorted = false;
}
static int WordArrayAt(const uint8_t* src, size_t* offset) {
    auto& ranges = WordRanges();
    if (!sWordRangesSorted) {
        std::sort(ranges.begin(), ranges.end(), [](const PortWordRange& a, const PortWordRange& b) { return a.start < b.start; });
        sWordRangesSorted = true;
    }
    uintptr_t a = (uintptr_t)src;
    auto it = std::upper_bound(ranges.begin(), ranges.end(), a,
                               [](uintptr_t v, const PortWordRange& r) { return v < r.start; });
    if (it == ranges.begin()) {
        return 0;
    }
    --it;
    if (a >= it->end) {
        return 0;
    }
    *offset = (size_t)(a - it->start);
    return it->size;
}
static void SwapWordsFrom(uint8_t* data, size_t len, int size, size_t offset) {
    // data[0] is at byte 'offset' of the array: swap each whole word, aligned to the array start
    size_t i = (size - offset % size) % size;
    for (; i + size <= len; i += size) {
        if (size == 4) {
            std::swap(data[i], data[i + 3]);
            std::swap(data[i + 1], data[i + 2]);
        } else {
            std::swap(data[i], data[i + 1]);
        }
    }
}

static const uint8_t* TmemLoad(uint32_t tmemWord, const uint8_t* src, uint32_t lineBytes, uint32_t strideBytes, uint32_t rows) {
    uint32_t start = tmemWord * 8, len = lineBytes * rows;
    if (gPortDrawLog) {
#ifndef _WIN32
        Dl_info di = {};
        dladdr(src, &di);
        fprintf(stderr, "[seq] load tmem=%u len=%u src=%p (%s+0x%lx)\n", start, len, (const void*)src,
                di.dli_sname ? di.dli_sname : "?", di.dli_saddr ? (unsigned long)((const uint8_t*)src - (const uint8_t*)di.dli_saddr) : 0ul);
#else
        fprintf(stderr, "[seq] load tmem=%u len=%u src=%p\n", start, len, (const void*)src);
#endif
    }
#ifndef _WIN32
    static const bool sT1b = getenv("PORT_TILE1LOG") != nullptr;
    if (sT1b && (start != 0 || (len > 2048 && start < 2048))) {
        static int n = 0;
        Dl_info di = {};
        dladdr(src, &di);
        static int ctx = 0;
        if (start == 2048 && sLastTimgCmd && ctx++ < 2) {
            const Fast::F3DGfx* c = (const Fast::F3DGfx*)sLastTimgCmd;
            fprintf(stderr, "[detail] settimg at %p:\n", sLastTimgCmd);
            for (int k = -10; k <= 6; k++)
                fprintf(stderr, "    %+3d: %08x %016lx\n", k, (uint32_t)c[k].words.w0, (unsigned long)c[k].words.w1);
        }
        if (n++ < 100000) fprintf(stderr, "[tile1] load tmem=%u len=%u src=%p (%s+0x%lx)\n", start, len, (const void*)src,
                                  di.dli_sname ? di.dli_sname : "?", di.dli_saddr ? (unsigned long)((const uint8_t*)src - (const uint8_t*)di.dli_saddr) : 0ul);
    }
#endif
    static const bool sLog = getenv("PORT_TMEMLOG") != nullptr;
    if (sLog && start <= 2048 && 2048 < start + (len ? len : 8)) {
        static int n = 0;
        if (n++ < 300) fprintf(stderr, "[tmem] load start=%u len=%u src=%p (line %u x %u)\n", start, len, (const void*)src, lineBytes, rows);
    }
    if (((uintptr_t)src >> 32) == 0xFFFFFFFFull) { // PORT debug: where a cut host pointer comes from
        static int n = 0;
        if (n++ < 3 && sLastTimgCmd) {
            const Fast::F3DGfx* c = (const Fast::F3DGfx*)sLastTimgCmd;
            fprintf(stderr, "[badload] src=%p tmem=%u len=%u, last settimg at %p:\n", (const void*)src, start, len, (const void*)c);
            for (int k = -6; k <= 4; k++)
                fprintf(stderr, "    %+3d: %08x %016lx\n", k, (uint32_t)c[k].words.w0, (unsigned long)c[k].words.w1);
        }
    }
    if (src == nullptr || (uintptr_t)src < 0x100000000ull || lineBytes == 0 || rows == 0 || len > (1u << 22)) {
        TmemForget(start, len ? len : 8);
        return src;
    }
    TmemForget(start, len);
    sTmemRegions.push_back({ start, len, src, std::vector<uint8_t>(len) });
    TmemRegion& r = sTmemRegions.back();
    size_t wordOffset;
    int wordSize = WordArrayAt(src, &wordOffset);
    for (uint32_t k = 0; k < rows; k++) {
        memcpy(r.data.data() + (size_t)k * lineBytes, src + (size_t)k * strideBytes, lineBytes);
        if (wordSize != 0) {
            SwapWordsFrom(r.data.data() + (size_t)k * lineBytes, lineBytes, wordSize, wordOffset + (size_t)k * strideBytes);
        }
    }
    return r.data.data();
}
static const TmemRegion* TmemFind(uint32_t tmemWord) {
    uint32_t a = tmemWord * 8;
    for (auto it = sTmemRegions.rbegin(); it != sTmemRegions.rend(); ++it) {
        if (a >= it->start && a < it->start + it->len) {
            return &*it;
        }
    }
    return nullptr;
}

struct RttInfo { int fb; uint32_t w, h; uint32_t stamp[4]; };
static std::unordered_map<uintptr_t, RttInfo> sOffscreenFbs;
// Finds the render-to-texture buffer containing a texture address; rowOff is the row (in texels) of that address.
// The RDRAM buffer of a render target carries a 16-byte marker written when the target is created. The game frees
// and reuses that memory (fonts, models, other images); once the CPU has overwritten the marker, the target is stale
// and textures at those addresses are real RDRAM data again.
// After a readback the first 16 bytes hold the real pixels (CPU copies of the image, e.g. the GB Tower's party
// pictures, showed the marker as coloured dots), and those bytes become the target's stamp.
static void RttStamp(uintptr_t addr, RttInfo* info) {
    uint32_t* w = (uint32_t*)addr;
    w[0] = 0x50525454u; // 'PRTT'
    w[1] = 0x52545421u;
    w[2] = (uint32_t)info->fb;
    w[3] = ~(uint32_t)info->fb;
    memcpy(info->stamp, w, sizeof(info->stamp));
}
static bool RttStampValid(uintptr_t addr, const RttInfo& info) {
    return memcmp((const void*)addr, info.stamp, sizeof(info.stamp)) == 0;
}
static void RttForgetOverlapping(uintptr_t start, uintptr_t end) {
    for (auto it = sOffscreenFbs.begin(); it != sOffscreenFbs.end();) {
        uintptr_t size = (uintptr_t)it->second.w * it->second.h * 2;
        if (it->first < end && start < it->first + size) {
            it = sOffscreenFbs.erase(it);
        } else {
            ++it;
        }
    }
}
// PORT: memory the game allocates again (main pool) is no longer a render target, even where the 16-byte marker
// survives (new data placed past the start of an old target, e.g. a font set inside a freed full-screen image).
// The game thread queues the ranges; the render thread applies them before using the table.
static std::mutex sRttForgetLock;
static std::vector<std::pair<uintptr_t, uintptr_t>> sRttForgetPending;
extern "C" void Port_GfxForgetRenderTargets(const void* start, size_t size) {
    static const bool off = getenv("PORT_NO_RTT_FORGET") != nullptr; // debug
    if (off) return;
    std::lock_guard<std::mutex> g(sRttForgetLock);
    sRttForgetPending.emplace_back((uintptr_t)start, (uintptr_t)start + size);
}
static void RttApplyForgets() {
    std::vector<std::pair<uintptr_t, uintptr_t>> pending;
    {
        std::lock_guard<std::mutex> g(sRttForgetLock);
        pending.swap(sRttForgetPending);
    }
    for (auto& r : pending) {
        RttForgetOverlapping(r.first, r.second);
    }
}
static bool RttLookup(const void* addr, RttInfo* out, float* rowOff, float* colOff = nullptr) {
    uintptr_t a = (uintptr_t)addr;
    static const bool off = getenv("PORT_NO_RTT") != nullptr || getenv("PORT_NO_RTT_SAMPLE") != nullptr; // debug: sample RDRAM instead of render targets
    RttApplyForgets();
    if (off || addr == nullptr || sOffscreenFbs.empty()) return false;
    for (auto it = sOffscreenFbs.begin(); it != sOffscreenFbs.end();) {
        uintptr_t size = (uintptr_t)it->second.w * it->second.h * 2;
        if (a >= it->first && a < it->first + size) {
            if (!RttStampValid(it->first, it->second)) {
                if (PortTexLog()) fprintf(stderr, "[rtt] dropped stale target at %p\n", (void*)it->first);
                it = sOffscreenFbs.erase(it); // memory reused: no longer a render target
                continue;
            }
            *out = it->second;
            *rowOff = (float)((a - it->first) / (it->second.w * 2));
            if (colOff != nullptr) { // a tile loaded from inside a row (LoadTextureTile with uls > 0)
                *colOff = (float)(((a - it->first) % (it->second.w * 2)) / 2);
            }
            return true;
        }
        ++it;
    }
    return false;
}
extern "C" int Port_ViFrameBufferState(const void* addr);
static uintptr_t sPendingOffscreenAddr = 0;
// A loaded image needs preservation only when the game blends into its CPU pixels (G_NOOP 0x51).
static bool sPreserveNextColorImage = false;
static bool sPendingOffscreenPreserve = false;
static uint32_t sPendingOffscreenWidth = 0;
static bool sInOffscreen = false;
static uintptr_t sCurOffscreenAddr = 0; // RDRAM address of the render target being drawn
// PORT: one persistent GPU framebuffer per N64 frame buffer (colour image address). The N64 rotates two or three frame
// buffers that keep their picture, and games update them incrementally (erasing what they drew into the same buffer
// two frames earlier); a single shared buffer leaves trails.
static std::unordered_map<uintptr_t, int> sMainFbs;
static bool sMarginsClearedThisRun = false; // see GfxDpSetColorImage
static bool sBandClearThisRun = false;      // the frame repaints the whole screen: the margin clear includes the edge band
// PORT: 2D screens in widescreen. A frame whose picture is mostly textured rectangles drawn before any 3D (title,
// menus with a picture background) is a 4:3 screen: from then on its drawing is kept inside the 4:3 area, so 3D extras
// over it (the Kids Club menu's flying cards) do not show in the black margins the N64 picture never had. Reset per
// frame (Run). A 3D scene whose backdrop comes first as such rectangles marks its frame (gDPPort3DScene in port_compat.h,
// a G_NOOP of type 0x53: the Elite Four room's sky; PORT_NO_3D_SCENE_MARK=1 ignores it). PORT_NO_2D_LOCK=1: off.
static float s2DRectArea = 0.0f;
static bool sSeen3DThisRun = false;
static bool sLock43ThisRun = false;
// PORT: full-screen 2D effects drawn for the 4:3 picture (the battle's strip wipe) are stretched over a wider picture:
// between the game's markers (gDPPortWideRectsBegin/End in port_compat.h, a G_NOOP of type 0x50) textured
// rectangles map the native width to the whole picture. Reset per frame (Run).
static bool sWideRects = false;
// PORT: HUD panels anchored to a side of a widescreen picture: between the game's markers (gDPPortAnchorLeft/Right/End
// in port_compat.h, a G_NOOP of type 0x52) everything drawn in the main picture moves by the margin beside the 4:3
// frame, so the battle's status boxes, portraits and menus sit at the screen's edges and slide in from there instead of
// out of the middle of the picture; they are not cut at the 4:3 frame's sides. -1 left, +1 right, 0 off; 2: not moved,
// only not cut (gDPPortNoCut: the gallery camera's film reel runs on to the screen's edge as it ran off the N64's).
// Reset per frame (Run). PORT_NO_HUD_ANCHOR=1: off. Nothing changes at 4:3.
static int sAnchor = 0;
// PORT: frame interpolation of frames that draw over the previous picture (Interpreter::ReplayBegin): a copy of each game
// frame buffer's picture from before the frame, and the buffers already copied away/back in this drawing
static bool sReplayRestore = false;
static std::unordered_map<uintptr_t, std::pair<int, uint32_t>> sSnapFbs; // framebuffer, its MSAA level
static std::unordered_set<uintptr_t> sSnapDoneThisRun;
// PORT: 2D in-between frames (Interpreter::mReplayT): the rectangles of the last exact drawing by key, in drawing order
struct Interp2DKey {
    const void* tex;
    int32_t w, h;
    uint32_t extra;
    bool operator==(const Interp2DKey& o) const {
        return tex == o.tex && w == o.w && h == o.h && extra == o.extra;
    }
};
struct Interp2DKeyHash {
    size_t operator()(const Interp2DKey& k) const {
        return std::hash<const void*>()(k.tex) ^ ((size_t)k.w * 73856093u) ^ ((size_t)k.h * 19349663u) ^
               ((size_t)k.extra * 83492791u);
    }
};
using Interp2DRects = std::unordered_map<Interp2DKey, std::vector<std::pair<int32_t, int32_t>>, Interp2DKeyHash>;
static Interp2DRects sRectLast, sRectRec;
static std::unordered_map<Interp2DKey, uint32_t, Interp2DKeyHash> sRectSeen;
static bool sRectLastRunExact = false;
static int sAppliedAnchor = 0; // the anchor the viewport and scissor sent to the backend were moved for
// PORT: while a 2D screen's patterned backdrop is drawn past the 4:3 lock (GfxDpTextureRectangle); while a
// full-screen fade is (GfxSpTri1)
static bool sWidePattern = false;
static bool sWideFade = false;
// PORT_NO_WIDE_PATTERN=1: none of this (patterns, fades and the frames between screens keep the 4:3 lock)
static const bool sNoWidePattern = getenv("PORT_NO_WIDE_PATTERN") != nullptr;
// PORT: whether the frame covered the whole widescreen picture in 2D (a widened pattern backdrop, a backdrop constant
// horizontally, a full-screen fade); the last frame's value tells the port how wide to show a CPU-drawn single-colour
// frame that follows it (Port_LastFrameFilledWide, os_devices.cpp)
static bool sFilledWideThisRun = false;
extern "C" bool Port_LastFrameFilledWide(void) {
    return sFilledWideThisRun;
}
// PORT: whether the RDP (Fast3D) renders into this N64 frame buffer; frames shown with osViSwapBuffer that it does not
// render were drawn by the CPU and are presented from memory (os_devices.cpp)
static std::unordered_map<uintptr_t, uint64_t> sMainFbDrawStamp;
static uint64_t sDrawStamp = 0;
extern "C" uint64_t Port_GpuDrawStampGlobal(void) {
    return sDrawStamp;
}
extern "C" uint64_t Port_GpuDrawStamp(const void* address) {
    auto it = sMainFbDrawStamp.find((uintptr_t)address);
    return it != sMainFbDrawStamp.end() ? it->second : 0;
}


#define HALF_SCREEN_WIDTH(activeFb) ((mFbActive ? activeFb->second.orig_width : mNativeDimensions.width) / 2)
#define HALF_SCREEN_HEIGHT(activeFb) ((mFbActive ? activeFb->second.orig_height : mNativeDimensions.height) / 2)

// Ratios for current window dimensions or active framebuffer scaled size
#define RATIO_X(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_width : dims.width) / (2.0f * HALF_SCREEN_WIDTH(activeFb)))
#define RATIO_Y(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_height : dims.height) / (2.0f * HALF_SCREEN_HEIGHT(activeFb)))

#define TEXTURE_CACHE_MAX_SIZE 1024

namespace Fast {
// PORT: last screen readback as 8-bit BGRA (set by the backend's ReadFramebufferToCPU for the screen), for screenshots
std::vector<uint8_t> gPortScreenBgra;
uint32_t gPortScreenW, gPortScreenH;
// PORT: the frame being finished is an in-between frame (frame interpolation): a screen readback the port asked for is
// kept for the next real frame (Metal reads the screen back one frame late)
bool gPortReplayFrame = false;

static UcodeHandlers ucode_handler_index = ucode_f3dex2;

const static uint32_t f3dex2AttrHandler[] = {
    F3DEX2_G_MTX_PROJECTION, F3DEX2_G_MTX_LOAD,  F3DEX2_G_MTX_PUSH,  F3DEX_G_MTX_NOPUSH,
    F3DEX2_G_CULL_FRONT,     F3DEX2_G_CULL_BACK, F3DEX2_G_CULL_BOTH,
};

const static uint32_t f3dexAttrHandler[] = { F3DEX_G_MTX_PROJECTION, F3DEX_G_MTX_LOAD,   F3DEX_G_MTX_PUSH,
                                             F3DEX_G_MTX_NOPUSH,     F3DEX_G_CULL_FRONT, F3DEX_G_CULL_BACK,
                                             F3DEX_G_CULL_BOTH };

static constexpr std::array ucode_attr_handlers = {
    &f3dexAttrHandler,  // ucode_f3db
    &f3dexAttrHandler,  // ucode_f3d
    &f3dexAttrHandler,  // ucode_f3dex
    &f3dexAttrHandler,  // ucode_f3exb
    &f3dex2AttrHandler, // ucode_f3ex2
    &f3dex2AttrHandler, // ucode_s2dex
};

static uint32_t get_attr(Attribute attr) {
    const auto ucode_map = ucode_attr_handlers[ucode_handler_index];
    // assert(ucode_map->contains(attr) && "Attribute not found in the current ucode handler");
    return (*ucode_map)[attr];
}

static std::string GetPathWithoutFileName(char* filePath) {
    size_t len = strlen(filePath);

    for (size_t i = len - 1; (long)i >= 0; i--) {
        if (filePath[i] == '/' || filePath[i] == '\\') {
            return std::string(filePath).substr(0, i);
        }
    }

    return filePath;
}

constexpr size_t MAX_TRI_BUFFER = 256;

Interpreter::Interpreter() {
    mRsp = new RSP();
    mRdp = new RDP();
    mBufVbo = new float[MAX_TRI_BUFFER * (32 * 3)];
}

Interpreter::~Interpreter() {
    delete mRsp;
    delete mRdp;
    delete[] mBufVbo;
}

static std::weak_ptr<Interpreter> mInstance;
// Cached for free-function callbacks that cannot capture context (plain function pointers).
static std::shared_ptr<Ship::ResourceManager> sResourceManager;
// Set a cached pointer to the instance so we don't need to go through the window every time
void GfxSetInstance(std::shared_ptr<Interpreter> gfx) {
    mInstance = gfx;
}

// N64 prim_depth is 15-bit (0 near, 0x7FFF far).
static constexpr float N64_PRIM_DEPTH_MAX = 32767.0f;

extern "C" uint64_t Port_ViTick();

namespace {
using RenderClock = std::chrono::steady_clock;
struct RenderTiming {
    double ms = 0;
    unsigned calls = 0;
};

bool TraceRenderFrames() {
    static const bool enabled = getenv("PORT_TRACE_RENDER_TIMING") != nullptr;
    return enabled;
}

bool TraceRenderSpikes() {
    static const bool enabled = getenv("PORT_TRACE_RENDER_SPIKES") != nullptr;
    return enabled;
}

bool TraceRenderTiming() {
    return TraceRenderFrames() || TraceRenderSpikes();
}

double RenderSpikeLimit() {
    static const double limit = [] {
        const char* value = getenv("PORT_RENDER_SPIKE_MS");
        const double parsed = value ? atof(value) : 50.0;
        return parsed > 0 ? parsed : 50.0;
    }();
    return limit;
}

struct TimeRenderCall {
    RenderTiming& timing;
    RenderClock::time_point start;

    explicit TimeRenderCall(RenderTiming& result)
        : timing(result), start(TraceRenderTiming() ? RenderClock::now() : RenderClock::time_point{}) {}
    ~TimeRenderCall() {
        if (TraceRenderTiming()) {
            timing.ms += std::chrono::duration<double, std::milli>(RenderClock::now() - start).count();
            ++timing.calls;
        }
    }
};

RenderTiming sUploadTiming, sDrawTiming, sShaderTiming;
size_t sTriangleCount = 0;

void UploadTimed(Fast::GfxRenderingAPI* api, const uint8_t* pixels, uint32_t width, uint32_t height) {
    TimeRenderCall timer(sUploadTiming);
    api->UploadTexture(pixels, width, height);
}
} // namespace

void Interpreter::Flush() {
    if (mBufVboLen > 0) {
        mRapi->SetCurrentPrimDepth((float)mRdp->prim_depth / N64_PRIM_DEPTH_MAX);
        {
            TimeRenderCall timer(sDrawTiming);
            if (TraceRenderTiming()) sTriangleCount += mBufVboNumTris;
            mRapi->DrawTriangles(mBufVbo, mBufVboLen, mBufVboNumTris);
        }
        mBufVboLen = 0;
        mBufVboNumTris = 0;
    }
}

void (*gPortShaderCreated)(uint64_t id0, uint64_t id1) = nullptr;

ShaderProgram* Interpreter::LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1) {
    ShaderProgram* prg = mRapi->LookupShader(id0, id1);
    if (prg == nullptr) {
        TimeRenderCall timer(sShaderTiming);
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        prg = mRapi->CreateAndLoadNewShader(id0, id1);
        mRenderingState.mShaderProgram = prg;
        if (gPortShaderCreated != nullptr) {
            gPortShaderCreated(id0, id1);
        }
    }
    return prg;
}

// PORT: shader warm-up. A program built the first time it is drawn stalls that draw: 1-4 ms with a warm driver cache,
// 100+ ms on a fresh install or after an update (the Metal cache is per binary). The port queues the programs earlier
// runs used before starting the intro (src/port/shader_warmup.cpp).
void Interpreter::QueueShaderWarmup(std::vector<std::pair<uint64_t, uint64_t>> ids) {
    if (mRapi->PrebuildShadersAsync(ids)) {
        return;
    }
    mWarmupQueue = std::move(ids);
    mWarmupNext = 0;
}

float Interpreter::ShaderWarmupProgress() {
    if (mWarmupQueue.empty()) return mRapi->ShaderPrebuildProgress();
    return float(mWarmupNext) / float(mWarmupQueue.size());
}

// The backends without a builder thread (OpenGL, DX11): before a frame's first draw, build queued programs for up to
// 4 ms (at least one per frame), then let the next draw load its program again (building one loads it).
void Interpreter::WarmupStep() {
    if (mWarmupNext >= mWarmupQueue.size()) {
        return;
    }
    static const bool sTrace = getenv("PORT_TRACE_SHADERS") != nullptr || getenv("PORT_TRACE_PACING") != nullptr;
    auto start = std::chrono::steady_clock::now();
    bool built = false;
    mRapi->UnloadShader(mRenderingState.mShaderProgram);
    while (mWarmupNext < mWarmupQueue.size() &&
           (!built || std::chrono::steady_clock::now() - start < std::chrono::milliseconds(4))) {
        auto [id0, id1] = mWarmupQueue[mWarmupNext++];
        if (mRapi->LookupShader(id0, id1) == nullptr) {
            mRapi->UnloadShader(mRapi->CreateAndLoadNewShader(id0, id1));
            built = true;
        }
    }
    mRenderingState.mShaderProgram = nullptr;
    if (mWarmupNext >= mWarmupQueue.size()) {
        if (sTrace) {
            fprintf(stderr, "[shader] warm-up done: %zu programs\n", mWarmupQueue.size());
        }
        mWarmupQueue = {};
        mWarmupNext = 0;
    }
}

const char* Interpreter::CCMUXtoStr(uint32_t ccmux) {
    static constexpr std::array tbl = {
        "G_CCMUX_COMBINED",
        "G_CCMUX_TEXEL0",
        "G_CCMUX_TEXEL1",
        "G_CCMUX_PRIMITIVE",
        "G_CCMUX_SHADE",
        "G_CCMUX_ENVIRONMENT",
        "G_CCMUX_1",
        "G_CCMUX_COMBINED_ALPHA",
        "G_CCMUX_TEXEL0_ALPHA",
        "G_CCMUX_TEXEL1_ALPHA",
        "G_CCMUX_PRIMITIVE_ALPHA",
        "G_CCMUX_SHADE_ALPHA",
        "G_CCMUX_ENV_ALPHA",
        "G_CCMUX_LOD_FRACTION",
        "G_CCMUX_PRIM_LOD_FRAC",
        "G_CCMUX_K5",
    };
    if (ccmux > tbl.size()) {
        return "G_CCMUX_0";
    }
    return tbl[ccmux];
}

// Seems unused
const char* Interpreter::ACMUXtoStr(uint32_t acmux) {
    static constexpr std::array tbl = {
        "G_ACMUX_COMBINED or G_ACMUX_LOD_FRACTION",
        "G_ACMUX_TEXEL0",
        "G_ACMUX_TEXEL1",
        "G_ACMUX_PRIMITIVE",
        "G_ACMUX_SHADE",
        "G_ACMUX_ENVIRONMENT",
        "G_ACMUX_1 or G_ACMUX_PRIM_LOD_FRAC",
        "G_ACMUX_0",
    };
    return tbl[acmux];
}

void Interpreter::GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key) {
    const bool is2Cyc = (key.options & SHADER_OPT(_2CYC)) != 0;

    uint8_t c[2][2][4];
    uint64_t shaderId0 = 0;
    uint64_t shaderId1 = key.options;
    uint8_t shaderInputMapping[2][7] = { { 0 } };
    bool usedTextures[2]{};
    for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
        uint32_t rgbA = (key.combine_mode >> (i * 28)) & 0xf;
        uint32_t rgbB = (key.combine_mode >> (i * 28 + 4)) & 0xf;
        uint32_t rgbC = (key.combine_mode >> (i * 28 + 8)) & 0x1f;
        uint32_t rgbD = (key.combine_mode >> (i * 28 + 13)) & 7;
        uint32_t alphaA = (key.combine_mode >> (i * 28 + 16)) & 7;
        uint32_t alphaB = (key.combine_mode >> (i * 28 + 16 + 3)) & 7;
        uint32_t alphaC = (key.combine_mode >> (i * 28 + 16 + 6)) & 7;
        uint32_t alphaD = (key.combine_mode >> (i * 28 + 16 + 9)) & 7;

        if (rgbA >= 8) {
            rgbA = G_CCMUX_0;
        }
        if (rgbB >= 8) {
            rgbB = G_CCMUX_0;
        }
        if (rgbC >= 16) {
            rgbC = G_CCMUX_0;
        }
        if (rgbD == 7) {
            rgbD = G_CCMUX_0;
        }

        if (rgbA == rgbB || rgbC == G_CCMUX_0) {
            // Normalize
            rgbA = G_CCMUX_0;
            rgbB = G_CCMUX_0;
            rgbC = G_CCMUX_0;
        }
        if (alphaA == alphaB || alphaC == G_ACMUX_0) {
            // Normalize
            alphaA = G_ACMUX_0;
            alphaB = G_ACMUX_0;
            alphaC = G_ACMUX_0;
        }
        if (i == 1) {
            if (rgbA != G_CCMUX_COMBINED && rgbB != G_CCMUX_COMBINED && rgbC != G_CCMUX_COMBINED &&
                rgbD != G_CCMUX_COMBINED) {
                // First cycle RGB not used, so clear it away
                c[0][0][0] = c[0][0][1] = c[0][0][2] = c[0][0][3] = G_CCMUX_0;
            }
            if (rgbC != G_CCMUX_COMBINED_ALPHA && alphaA != G_ACMUX_COMBINED && alphaB != G_ACMUX_COMBINED &&
                alphaD != G_ACMUX_COMBINED) {
                // First cycle ALPHA not used, so clear it away
                c[0][1][0] = c[0][1][1] = c[0][1][2] = c[0][1][3] = G_ACMUX_0;
            }
        }

        c[i][0][0] = rgbA;
        c[i][0][1] = rgbB;
        c[i][0][2] = rgbC;
        c[i][0][3] = rgbD;
        c[i][1][0] = alphaA;
        c[i][1][1] = alphaB;
        c[i][1][2] = alphaC;
        c[i][1][3] = alphaD;
    }
    if (!is2Cyc) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t k = 0; k < 4; k++) {
                c[1][i][k] = i == 0 ? G_CCMUX_0 : G_ACMUX_0;
            }
        }

        // In 1-cycle mode, TEXEL1 returns the same value as TEXEL0.
        // Remap combiner inputs so the shader samples from the correct slot.
        // Ex: TEXEL1/TEXEL1_ALPHA → TEXEL0/TEXEL0_ALPHA
        for (uint32_t k = 0; k < 4; k++) {
            if (c[0][0][k] == G_CCMUX_TEXEL1)
                c[0][0][k] = G_CCMUX_TEXEL0;
            if (c[0][0][k] == G_CCMUX_TEXEL1_ALPHA)
                c[0][0][k] = G_CCMUX_TEXEL0_ALPHA;
            if (c[0][1][k] == G_ACMUX_TEXEL1)
                c[0][1][k] = G_ACMUX_TEXEL0;
        }
    }
    {
        uint8_t inputNumber[32] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
            for (uint32_t j = 0; j < 4; j++) {
                // Mux values 6/7/15 are overloaded by slot. Only B (value 6 = CENTER, 7 = K4)
                // and C (value 6 = SCALE, 15 = K5) carry chroma-key/convert inputs; A and D
                // reuse those values for unrelated constants.
                if (j == 1 && c[i][0][j] == G_CCMUX_CENTER) {
                    c[i][0][j] = G_CCMUX_KEY_CENTER;
                } else if (j == 1 && c[i][0][j] == G_CCMUX_K4) {
                    c[i][0][j] = G_CCMUX_CONVERT_K4;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_SCALE) {
                    c[i][0][j] = G_CCMUX_KEY_SCALE;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_K5) {
                    c[i][0][j] = G_CCMUX_CONVERT_K5;
                }
                uint32_t val = 0;
                switch (c[i][0][j]) {
                    case G_CCMUX_0:
                        val = SHADER_0;
                        break;
                    case G_CCMUX_1:
                        val = SHADER_1;
                        break;
                    case G_CCMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL0_ALPHA:
                        val = SHADER_TEXEL0A;
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1_ALPHA:
                        val = SHADER_TEXEL1A;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_NOISE:
                        val = SHADER_NOISE;
                        break;
                    case G_CCMUX_PRIMITIVE:
                    case G_CCMUX_PRIMITIVE_ALPHA:
                    case G_CCMUX_PRIM_LOD_FRAC:
                    case G_CCMUX_SHADE:
                    case G_CCMUX_ENVIRONMENT:
                    case G_CCMUX_ENV_ALPHA:
                    case G_CCMUX_LOD_FRACTION:
                    case G_CCMUX_KEY_CENTER:
                    case G_CCMUX_KEY_SCALE:
                    case G_CCMUX_CONVERT_K4:
                    case G_CCMUX_CONVERT_K5:
                        if (inputNumber[c[i][0][j]] == 0) {
                            shaderInputMapping[0][nextInputNumber - 1] = c[i][0][j];
                            inputNumber[c[i][0][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][0][j]];
                        break;
                    case G_CCMUX_COMBINED:
                        val = SHADER_COMBINED;
                        break;
                    default:
                        fprintf(stderr, "Unsupported ccmux: %d\n", c[i][0][j]);
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + j * 4);
            }
        }
    }
    {
        uint8_t inputNumber[16] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t val = 0;
                switch (c[i][1][j]) {
                    case G_ACMUX_0:
                        val = SHADER_0;
                        break;
                    case G_ACMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_ACMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_ACMUX_LOD_FRACTION:
                        // case G_ACMUX_COMBINED: same numerical value
                        if (j != 2) {
                            val = SHADER_COMBINED;
                            break;
                        }
                        c[i][1][j] = G_CCMUX_LOD_FRACTION;
                        [[fallthrough]]; // for G_ACMUX_LOD_FRACTION
                    case G_ACMUX_1:
                        // case G_ACMUX_PRIM_LOD_FRAC: same numerical value
                        if (j != 2) {
                            val = SHADER_1;
                            break;
                        }
                        [[fallthrough]]; // for G_ACMUX_PRIM_LOD_FRAC
                    case G_ACMUX_PRIMITIVE:
                    case G_ACMUX_SHADE:
                    case G_ACMUX_ENVIRONMENT:
                        if (inputNumber[c[i][1][j]] == 0) {
                            shaderInputMapping[1][nextInputNumber - 1] = c[i][1][j];
                            inputNumber[c[i][1][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][1][j]];
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + 16 + j * 4);
            }
        }
    }
    comb->shader_id0 = shaderId0;
    comb->shader_id1 = shaderId1;
    comb->usedTextures[0] = usedTextures[0];
    comb->usedTextures[1] = usedTextures[1];
    // comb->prg = gfx_lookup_or_create_mShaderProgram(shader_id0, shader_id1);
    memcpy(comb->shader_input_mapping, shaderInputMapping, sizeof(shaderInputMapping));
}

ColorCombiner* Interpreter::LookupOrCreateColorCombiner(const ColorCombinerKey& key) {
    if (mPrevCombiner != mColorCombinerPool.end() && mPrevCombiner->first == key) {
        return &mPrevCombiner->second;
    }
    mPrevCombiner = mColorCombinerPool.find(key);
    if (mPrevCombiner != mColorCombinerPool.end()) {
        return &mPrevCombiner->second;
    }
    Flush();
    mPrevCombiner = mColorCombinerPool.insert(std::make_pair(key, ColorCombiner())).first;
    GenerateCC(&mPrevCombiner->second, key);
    return &mPrevCombiner->second;
}

void Interpreter::SetResolvedResourceCacheEnabled(bool enabled) {
    mResolvedResourceCacheEnabled = enabled;
    if (!enabled) {
        mResolvedResourceCache.clear();
    }
}

/**
 * Texture binds resolve the same paths every frame; skip the resource
 * manager's string/hash/mutex work by memoizing on the pointer.
 */
std::shared_ptr<Ship::IResource> Interpreter::ResolveResourceCached(const char* path) {
    if (path == nullptr) {
        return nullptr;
    }
    if (!mResolvedResourceCacheEnabled) {
        return mResourceManager->LoadResourceProcess(path);
    }
    auto it = mResolvedResourceCache.find(path);
    if (it != mResolvedResourceCache.end()) {
        return it->second;
    }
    auto res = mResourceManager->LoadResourceProcess(path);
    // Only memoize a hit. The resource manager caches its own misses, so re-asking
    // for one is cheap, and CacheExternalResource can turn a path that missed into a
    // valid resource at runtime. A memoized null would outlive the resource itself.
    if (res != nullptr) {
        mResolvedResourceCache[path] = res;
    }
    return res;
}

void Interpreter::TextureCacheClear() {
    for (const auto& entry : mTextureCache.map) {
        mTextureCache.free_texture_ids.push_back(entry.second.texture_id);
    }
    mTextureCache.map.clear();
    mTextureCache.lru.clear();
    mResolvedResourceCache.clear();
    // Pre-allocate buckets so the map never rehashes during normal operation.
    // Rehashing invalidates all iterators, including those stored in LRU entries.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
    // Null rendering-state pointers — they pointed into map nodes that are now freed.
    std::fill(std::begin(mRenderingState.mTextures), std::end(mRenderingState.mTextures), nullptr);
}

void Interpreter::ShaderCacheClear() {
    mRapi->ClearShaderCache();
}

bool Interpreter::TextureCacheLookup(int i, const TextureCacheKey& key) {
    TextureCacheMap::iterator it = mTextureCache.map.find(key);
    TextureCacheNode** n = &mRenderingState.mTextures[i];

    if (it != mTextureCache.map.end()) {
        mRapi->SelectTexture(i, it->second.texture_id);
        *n = &*it;
        mTextureCache.lru.splice(mTextureCache.lru.end(), mTextureCache.lru,
                                 it->second.lru_location); // move to back
        return true;
    }

    if (mTextureCache.map.size() >= TEXTURE_CACHE_MAX_SIZE) {
        // Remove the texture that was least recently used
        it = mTextureCache.lru.front().it;
        mTextureCache.free_texture_ids.push_back(it->second.texture_id);
        for (int j = 0; j < SHADER_MAX_TEXTURES; j++) {
            if (mRenderingState.mTextures[j] == &*it)
                mRenderingState.mTextures[j] = nullptr;
        }
        mTextureCache.map.erase(it);
        mTextureCache.lru.pop_front();
    }

    uint32_t texture_id;
    if (!mTextureCache.free_texture_ids.empty()) {
        texture_id = mTextureCache.free_texture_ids.back();
        mTextureCache.free_texture_ids.pop_back();
    } else {
        texture_id = mRapi->NewTexture();
    }

    it = mTextureCache.map.insert(std::make_pair(key, TextureCacheValue())).first;
    TextureCacheNode* node = &*it;
    node->second.texture_id = texture_id;
    node->second.lru_location = mTextureCache.lru.insert(mTextureCache.lru.end(), { it });

    mRapi->SelectTexture(i, texture_id);
    mRapi->SetSamplerParameters(i, false, 0, 0);
    *n = node;
    return false;
}

std::string_view Interpreter::GetBaseTexturePath(std::string_view path) {
    if (path.starts_with(Ship::IResource::gAltAssetPrefix)) {
        return path.substr(Ship::IResource::gAltAssetPrefix.length());
    }

    return path;
}

void Interpreter::TextureCacheDelete(const uint8_t* origAddr) {
    // PORT: a scan of the whole cache (at most TEXTURE_CACHE_MAX_SIZE entries): the bucket no longer depends on the
    // address alone (TextureCacheKey::Hasher)
    for (auto it = mTextureCache.map.begin(); it != mTextureCache.map.end();) {
        if (it->first.texture_addr != origAddr) {
            ++it;
            continue;
        }
        for (int j = 0; j < SHADER_MAX_TEXTURES; j++) {
            if (mRenderingState.mTextures[j] == &*it)
                mRenderingState.mTextures[j] = nullptr;
        }
        mTextureCache.lru.erase(it->second.lru_location);
        mTextureCache.free_texture_ids.push_back(it->second.texture_id);
        it = mTextureCache.map.erase(it);
    }
}

// Pick the per-line byte width for texture decode. Prefer the DRAM stride from
// loaded_texture when it looks like real per-line info (differs from total size).
// Fall back to the TMEM tile stride when loaded sizes match total (LoadBlock with
// width=1, where line_size == full_image_line_size == size).
static uint32_t GetEffectiveLineSize(uint32_t lineSizeBytes, uint32_t fullImageLineSizeBytes, uint32_t sizeBytes,
                                     uint32_t tileLineSizeBytes) {
    if ((lineSizeBytes != sizeBytes || fullImageLineSizeBytes != sizeBytes) && lineSizeBytes > 0) {
        return lineSizeBytes;
    }
    return tileLineSizeBytes;
}

static uint32_t GetTileSizeFromCoordinates(float low, float high) {
    // An unset tile (high <= low) defines no region; return 0 so callers skip the tile-region clamp
    // instead of collapsing the texture to the phantom 1-texel size the +4 formula would yield.
    if (high <= low) {
        return 0;
    }
    return static_cast<uint32_t>(lroundf((high - low + 4.0f) / 4.0f));
}

// PORT: fully transparent texels take the colour of their opaque neighbours. Rendering above the N64 resolution
// filters between texels, and the RGB of transparent ones (often a magenta key colour) bled into the edges of
// alpha-cut UI art (HUD digits came out pink). Alpha is unchanged. Only for filtered sampling: point-sampled texels
// never mix, and where alpha is ignored (CPU-drawn frames shown with an opaque draw, PresentCpuFrame) the recoloured
// texels showed (the GB Tower menu's text, whose colours have alpha bit 0, came out black).
static void BleedTransparentTexels(uint8_t* rgba, uint32_t width, uint32_t height) {
    static const bool off = getenv("PORT_NO_ALPHA_BLEED") != nullptr;
    if (off || width == 0 || height == 0 || (uint64_t)width * height > (1u << 20)) {
        return;
    }
    static std::vector<uint8_t> src;
    src.assign(rgba, rgba + (size_t)width * height * 4);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t* d = &rgba[((size_t)y * width + x) * 4];
            if (d[3] != 0) {
                continue;
            }
            uint32_t r = 0, g = 0, b = 0, n = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int xx = (int)x + dx, yy = (int)y + dy;
                    if ((dx == 0 && dy == 0) || xx < 0 || yy < 0 || xx >= (int)width || yy >= (int)height) {
                        continue;
                    }
                    const uint8_t* q = &src[((size_t)yy * width + xx) * 4];
                    if (q[3] != 0) {
                        r += q[0];
                        g += q[1];
                        b += q[2];
                        n++;
                    }
                }
            }
            if (n != 0) {
                d[0] = (uint8_t)(r / n);
                d[1] = (uint8_t)(g / n);
                d[2] = (uint8_t)(b / n);
            }
        }
    }
}

void Interpreter::ImportTextureRgba16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba16: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike =
        renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13; // < 1.625x
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    // HD replacement textures must still clamp to the rendered tile region
    bool isHd = metadata->h_byte_scale != 1 || metadata->v_pixel_scale != 1;
    if ((isHd || pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((isHd || pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes / 2)) + (x);

            uint16_t col16 = (addr[2 * clrIdx] << 8) | addr[2 * clrIdx + 1];
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;

            i++;
        }
    }

    if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) { // see BleedTransparentTexels
        BleedTransparentTexels(mTexUploadBuffer, width, height);
    }
    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureRgba32(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba32: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes * 2);
    uint32_t width = widthBytes / 4;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    // HD replacement textures must still clamp to the rendered tile region
    bool isHd = metadata->h_byte_scale != 1 || metadata->v_pixel_scale != 1;
    if ((isHd || pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((isHd || pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    if (full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 4;
    }

    // Copy pixel by pixel, respecting full image stride (handles sub-tile loads)
    uint32_t fullImageStridePixels = full_image_line_size_bytes / 4;
    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageStridePixels + x;
            mTexUploadBuffer[4 * i + 0] = addr[4 * srcIdx + 0];
            mTexUploadBuffer[4 * i + 1] = addr[4 * srcIdx + 1];
            mTexUploadBuffer[4 * i + 2] = addr[4 * srcIdx + 2];
            mTexUploadBuffer[4 * i + 3] = addr[4 * srcIdx + 3];
            i++;
        }
    }
    if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) { // see BleedTransparentTexels
        BleedTransparentTexels(mTexUploadBuffer, width, height);
    }
    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = widthBytes;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t part = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part >> 1;
            uint8_t alpha = part & 1;
            mTexUploadBuffer[4 * i + 0] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 3] = alpha ? 255 : 0;
            i++;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageLineSizeBytes + x;
            uint8_t intensity = addr[srcIdx] >> 4;
            uint8_t alpha = addr[srcIdx] & 0xf;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(alpha);
            i++;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA16: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (full_image_line_size_bytes / 2)) + (x);

            uint8_t intensity = addr[2 * clrIdx];
            uint8_t alpha = addr[2 * clrIdx + 1];
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            mTexUploadBuffer[4 * i + 0] = r;
            mTexUploadBuffer[4 * i + 1] = g;
            mTexUploadBuffer[4 * i + 2] = b;
            mTexUploadBuffer[4 * i + 3] = alpha;

            i++;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

static int sCi4RawPalette = -1; // PORT: >= 0 while a CI4 texture without TLUT is decoded by ImportTextureI4

void Interpreter::ImportTextureI4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width / 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes * 2)) + (x);

            uint8_t byte = addr[clrIdx / 2];
            uint8_t part = (byte >> (4 - (clrIdx % 2) * 4)) & 0xf;
            uint8_t intensity = sCi4RawPalette >= 0 ? (uint8_t)((sCi4RawPalette << 4) | part) : SCALE_4_8(part);
            mTexUploadBuffer[4 * i + 0] = intensity;
            mTexUploadBuffer[4 * i + 1] = intensity;
            mTexUploadBuffer[4 * i + 2] = intensity;
            mTexUploadBuffer[4 * i + 3] = intensity;

            i++;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureI8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t intensity = addr[y * fullImageLineSizeBytes + x];
            mTexUploadBuffer[4 * i + 0] = intensity;
            mTexUploadBuffer[4 * i + 1] = intensity;
            mTexUploadBuffer[4 * i + 2] = intensity;
            mTexUploadBuffer[4 * i + 3] = intensity;
            i++;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi4(int tile, bool importReplacement) {
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    uint32_t palIdx = mRdp->texture_tile[tile].palette; // 0-15

    const uint8_t* palette;

    if (mRdp->palettes[palIdx / 8] == nullptr) {
        SPDLOG_WARN("CI4: null palette slot {} for palIdx={}", palIdx / 8, palIdx);
        return;
    }
    palette = mRdp->palettes[palIdx / 8] + (palIdx % 8) * 16 * 2;

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    uint32_t resultLineSizeBytes = baseLineSizeBytes;

    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    // CI4: 2 pixels per byte
    uint32_t width = resultLineSizeBytes * 2;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    // HD replacement textures must still clamp to the rendered tile region
    bool isHd = metadata->h_byte_scale != 1 || metadata->v_pixel_scale != 1;
    if ((isHd || pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((isHd || pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = resultLineSizeBytes;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t idx = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint16_t col16 = (palette[idx * 2] << 8) | palette[idx * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
            i++;
        }
    }

    if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) { // see BleedTransparentTexels
        BleedTransparentTexels(mTexUploadBuffer, width, height);
    }
    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    if (mRdp->palettes[0] == nullptr || mRdp->palettes[1] == nullptr) {
        SPDLOG_WARN("CI8: null palette (pal0={}, pal1={})", static_cast<const void*>(mRdp->palettes[0]),
                    static_cast<const void*>(mRdp->palettes[1]));
        return;
    }

    for (uint32_t i = 0, j = 0; i < sizeBytes; j += fullImageLineSizeBytes - lineSizeBytes) {
        for (uint32_t k = 0; k < lineSizeBytes; i++, k++, j++) {
            uint8_t idx = addr[j];
            uint16_t col16 = (mRdp->palettes[idx / 128][(idx % 128) * 2] << 8) |
                             mRdp->palettes[idx / 128][(idx % 128) * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
        }
    }

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    uint32_t resultLineSizeBytes = baseLineSizeBytes;
    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    uint32_t width = resultLineSizeBytes;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    // HD replacement textures must still clamp to the rendered tile region
    bool isHd = metadata->h_byte_scale != 1 || metadata->v_pixel_scale != 1;
    if ((isHd || pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((isHd || pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) { // see BleedTransparentTexels
        BleedTransparentTexels(mTexUploadBuffer, width, height);
    }
    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureImg(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureImg: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    UploadTimed(mRapi, addr, width, height);
}

void Interpreter::ImportTextureRaw(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRaw: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    Fast::TextureType type = metadata->type;
    std::shared_ptr<Fast::Texture> resource = metadata->resource;

    // if texture type is CI4 or CI8 we need to apply tlut to it
    switch (type) {
        case Fast::TextureType::Palette4bpp:
            ImportTextureCi4(tile, importReplacement);
            return;
        case Fast::TextureType::Palette8bpp:
            ImportTextureCi8(tile, importReplacement);
            return;
        default:
            break;
    }

    uint32_t numLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t numOriginallyLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    uint32_t resultOrigLineSize = mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_32b:
            resultOrigLineSize *= 2;
            break;
    }
    uint32_t resultOrigHeight = numOriginallyLoadedBytes / resultOrigLineSize;
    uint32_t resultNewLineSize = resultOrigLineSize * metadata->h_byte_scale;
    uint32_t resultNewHeight = resultOrigHeight * metadata->v_pixel_scale;

    if (resultNewLineSize == 4 * width && resultNewHeight == height) {
        // Can use the texture directly since it has the correct dimensions
        UploadTimed(mRapi, addr, width, height);
        return;
    }

    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    // Get the resource's true image size
    uint32_t resourceImageSizeBytes = resource->ImageDataSize;
    uint32_t safeFullImageLineSizeBytes = fullImageLineSizeBytes;
    uint32_t safeLineSizeBytes = line_size_bytes;
    uint32_t safeLoadedBytes = numLoadedBytes;

    // Sometimes the texture load commands will specify a size larger than the authentic texture
    // Normally the OOB info is read as garbage, but will cause a crash on some platforms
    // Restrict the bytes to a safe amount
    if (numLoadedBytes > resourceImageSizeBytes) {
        safeLoadedBytes = resourceImageSizeBytes;
        safeLineSizeBytes = resourceImageSizeBytes;
        safeFullImageLineSizeBytes = resourceImageSizeBytes;
    }

    // Safely only copy the amount of bytes the resource can allow
    for (uint32_t i = 0, j = 0; i < safeLoadedBytes; i += safeLineSizeBytes, j += safeFullImageLineSizeBytes) {
        memcpy(mTexUploadBuffer + i, addr + j, safeLineSizeBytes);
    }

    // Set the remaining bytes to load as 0
    if (numLoadedBytes > resourceImageSizeBytes) {
        memset(mTexUploadBuffer + resourceImageSizeBytes, 0, numLoadedBytes - resourceImageSizeBytes);
    }

    // Describe the buffer by what was actually packed (the loaded HD stride)
    uint32_t uploadWidth = safeLineSizeBytes / 4;
    uint32_t uploadHeight = safeLineSizeBytes > 0 ? safeLoadedBytes / safeLineSizeBytes : 0;
    const bool singleLineLoad = uploadHeight <= 1 && resultNewLineSize != 0 && resultNewLineSize < safeLineSizeBytes;
    if (uploadWidth > (uint32_t)mRapi->GetMaxTextureSize() || singleLineLoad) {
        if (safeLoadedBytes == (uint64_t)width * height * 4) {
            // buffer holds the full image, use its real dimensions
            uploadWidth = width;
            uploadHeight = height;
        } else {
            // partial load, fall back to the tile-derived dimensions
            uploadWidth = resultNewLineSize / 4;
            uploadHeight = resultNewHeight;
        }
    }
    UploadTimed(mRapi, mTexUploadBuffer, uploadWidth, uploadHeight);
}

void Interpreter::ImportTexture(int i, int tile, bool importReplacement) {
    if (!importReplacement) { // PORT: take the tile's bytes from the TMEM region that covers its TMEM address
        if (const TmemRegion* r = TmemFind(mRdp->texture_tile[tile].tmem)) {
            auto& lt = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
            uint32_t off = mRdp->texture_tile[tile].tmem * 8 - r->start;
            if (lt.addr != r->data.data() + off) {
                lt.addr = r->data.data() + off;
                lt.raw_addr = r->raw ? r->raw + off : nullptr;
                lt.orig_size_bytes = lt.size_bytes = r->len - off;
                if (mRdp->texture_tile[tile].line_size_bytes != 0) {
                    lt.line_size_bytes = lt.full_image_line_size_bytes = mRdp->texture_tile[tile].line_size_bytes;
                }
                lt.tex_flags = 0;
                lt.raw_tex_metadata = {};
                lt.raw_tex_metadata.h_byte_scale = 1;
                lt.raw_tex_metadata.v_pixel_scale = 1;
                lt.masked = false;
                lt.blended = false;
            }
        }
    }
    uint8_t fmt = mRdp->texture_tile[tile].fmt;
    uint8_t siz = mRdp->texture_tile[tile].siz;
    uint32_t texFlags = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags;
    uint32_t tmemIdex = mRdp->texture_tile[tile].tmem_index;
    uint8_t paletteIndex = mRdp->texture_tile[tile].palette;
    uint32_t origSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
    // PORT: with texture lookup off (G_TT_NONE) the RDP reads a CI texel as an intensity: the index itself (CI4: palette
    // number in the high nibble) on all four channels. Decoded through the I4/I8 paths, cached as fmt I.
    static const bool sNoCiRaw = getenv("PORT_NO_CI_RAW") != nullptr; // debug: old behaviour (always look up a palette)
    if (!sNoCiRaw && fmt == G_IM_FMT_CI && (mRdp->other_mode_h & (3U << G_MDSFT_TEXTLUT)) == G_TT_NONE &&
        (siz == G_IM_SIZ_4b || siz == G_IM_SIZ_8b)) {
        fmt = G_IM_FMT_I;
        sCi4RawPalette = siz == G_IM_SIZ_4b ? paletteIndex : -1;
    } else {
        sCi4RawPalette = -1;
    }

    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* origAddr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Identifier.GetPath()))
                  ->second.replacementData
            : mRdp->loaded_texture[tmemIdex].addr;

    // Check if this texture address is a registered GPU framebuffer mirror.
    // If so, bind the GPU FB directly — full resolution, no CPU readback needed.
    if (origAddr != nullptr && !importReplacement) {
        auto fbIt = mFbTextures.find((uintptr_t)origAddr);
        if (fbIt != mFbTextures.end()) {
            Flush();
            mRapi->SelectTextureFb(fbIt->second);
            mRdp->textures_changed[i] = false;
            return;
        }
    }

    if (origAddr == nullptr) {
        // Try the other TMEM slot -- some multi-tile setups only load one slot
        // and expect both tiles to reference it.
        uint32_t otherTmem = tmemIdex ^ 1;
        origAddr = mRdp->loaded_texture[otherTmem].addr;
        if (origAddr == nullptr) {
            SPDLOG_WARN("ImportTexture: null texture address for tile {} (both TMEM slots empty)", tile);
            return;
        }
        SPDLOG_WARN("ImportTexture: tile {} TMEM slot {} empty, falling back to slot {}", tile, tmemIdex, otherTmem);
        tmemIdex = otherTmem;
        origSizeBytes = mRdp->loaded_texture[otherTmem].orig_size_bytes;
        texFlags = mRdp->loaded_texture[otherTmem].tex_flags;
    }

    // Use palette_dram_addr (the original DRAM source) instead of palettes[]
    // (which always points to the staging buffer) so the same texture drawn
    // with different palettes gets distinct cache entries.
    TextureCacheKey key;
    if (fmt == G_IM_FMT_CI) {
        if (siz == G_IM_SIZ_4b) {
            uint8_t palSlot = paletteIndex / 8;
            key = { origAddr,
                    { palSlot == 0 ? mRdp->palette_dram_addr[0] : nullptr,
                      palSlot == 1 ? mRdp->palette_dram_addr[1] : nullptr },
                    fmt,
                    siz,
                    paletteIndex,
                    origSizeBytes };
        } else {
            // CI8 uses both palette halves
            key = { origAddr,     { mRdp->palette_dram_addr[0], mRdp->palette_dram_addr[1] }, fmt, siz, paletteIndex,
                    origSizeBytes };
        }
    } else {
        key = { origAddr, {}, fmt, siz, paletteIndex, origSizeBytes };
    }
    {   // PORT: content hash (FNV-1a over the texels and the palette in use)
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](const uint8_t* p, size_t n) {
            if (p == nullptr) return;
            for (size_t k = 0; k < n; k++) h = (h ^ p[k]) * 1099511628211ull;
        };
        if ((uintptr_t)origAddr >= 0x100000000ull) {
            mix(origAddr, std::min<uint32_t>(origSizeBytes, 1u << 20));
        }
        if (fmt == G_IM_FMT_CI) {
            if (siz == G_IM_SIZ_4b) {
                const uint8_t* pal = mRdp->palettes[paletteIndex / 8];
                mix(pal ? pal + (paletteIndex % 8) * 16 * 2 : nullptr, 16 * 2);
            } else {
                mix(mRdp->palettes[0], 128 * 2);
                mix(mRdp->palettes[1], 128 * 2);
            }
        }
        key.content_hash = h;
        // PORT: RAM textures come from TMEM snapshots at a new address for nearly every load (in battles almost
        // every lookup missed: some 50 uploads a frame of a few hundred distinct textures, 1-2 ms a frame in the GL
        // driver on the AYN Thor). What the decoders produce depends on the bytes and the tile and load state, not on
        // the address, so the key holds those instead.
        // Resource textures (HD ones read more than the hashed bytes) keep their address, which does not change.
        const auto& lt = mRdp->loaded_texture[tmemIdex];
        if ((uintptr_t)origAddr >= 0x100000000ull && !importReplacement && lt.raw_tex_metadata.resource == nullptr) {
            const auto& tt = mRdp->texture_tile[tile];
            uint64_t d = 1469598103934665603ull;
            auto mixWord = [&d](uint64_t v) { d = (d ^ v) * 1099511628211ull; };
            mixWord(lt.size_bytes);
            mixWord(lt.full_image_line_size_bytes);
            mixWord(lt.line_size_bytes);
            mixWord(lt.tex_flags);
            uint32_t scale[2];
            memcpy(&scale[0], &lt.raw_tex_metadata.h_byte_scale, 4);
            memcpy(&scale[1], &lt.raw_tex_metadata.v_pixel_scale, 4);
            mixWord((uint64_t)scale[0] << 32 | scale[1]);
            mixWord((uint64_t)lt.masked << 8 | lt.blended);
            mixWord(tt.line_size_bytes);
            mixWord((uint64_t)tt.cms << 24 | (uint64_t)tt.cmt << 16 | (uint64_t)tt.masks << 8 | tt.maskt);
            uint32_t bounds[4];
            memcpy(bounds, &tt.uls, sizeof bounds); // uls, ult, lrs, lrt
            mixWord((uint64_t)bounds[0] << 32 | bounds[1]);
            mixWord((uint64_t)bounds[2] << 32 | bounds[3]);
            mixWord(sCi4RawPalette + 1);
            mixWord((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT); // BleedTransparentTexels
            key.texture_addr = nullptr;
            key.decode_hash = d;
        }
    }

    if (TextureCacheLookup(i, key)) {
        return;
    }

    // Guard against zero-sized textures that would cause divide-by-zero
    // or GPU API errors in UploadTexture.
    if (mRdp->texture_tile[tile].line_size_bytes == 0 || mRdp->loaded_texture[tmemIdex].size_bytes == 0 ||
        origAddr == nullptr) {
        return;
    }

    if ((texFlags & TEX_FLAG_LOAD_AS_IMG) != 0) {
        ImportTextureImg(tile, importReplacement);
        return;
    }

    // if load as raw is set then we load_raw();
    if ((texFlags & TEX_FLAG_LOAD_AS_RAW) != 0) {
        ImportTextureRaw(tile, importReplacement);
        return;
    }

    switch (fmt) {
        case G_IM_FMT_RGBA:
            if (siz == G_IM_SIZ_16b) {
                ImportTextureRgba16(tile, importReplacement);
            } else if (siz == G_IM_SIZ_32b) {
                ImportTextureRgba32(tile, importReplacement);
            } else {
                SPDLOG_ERROR("RGBA Texture that isn't 16 or 32 bit. Size = {}", siz);
                // OTRTODO: Sometimes, seemingly randomly, we end up here. Could be a bad dlist, could be
                // something F3D does not have supported. Further investigation is needed.
            }
            break;
        case G_IM_FMT_IA:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureIA4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureIA8(tile, importReplacement);
            } else if (siz == G_IM_SIZ_16b) {
                ImportTextureIA16(tile, importReplacement);
            } else {
                SPDLOG_ERROR("IA Texture that isn't 4, 8, or 16 bit. Size = {}", siz);
                ;
            }
            break;
        case G_IM_FMT_CI:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureCi4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureCi8(tile, importReplacement);
            } else if (siz == G_IM_SIZ_16b) {
                // CI+16b is hardware-invalid on N64. The tile's fmt is likely
                // stale from a prior draw. Decode as RGBA16 instead.
                ImportTextureRgba16(tile, importReplacement);
            } else if (siz == G_IM_SIZ_32b) {
                ImportTextureRgba32(tile, importReplacement);
            } else {
                SPDLOG_ERROR("CI Texture with unexpected size = {}", siz);
            }
            break;
        case G_IM_FMT_I:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureI4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureI8(tile, importReplacement);
            } else {
                SPDLOG_ERROR("I Texture that isn't 4 or 8 bit. Size = {}", siz);
            }
            break;
        case G_IM_FMT_YUV:
            SPDLOG_ERROR("YUV Textures not supported");
            break;
        default:
            SPDLOG_ERROR("Invalid texture format. Fmt = {}", fmt);
            break;
    }
}

void Interpreter::ImportTextureMask(int i, int tile) {
    uint32_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    RawTexMetadata metadata = mRdp->loaded_texture[tmemIndex].raw_tex_metadata;

    if (metadata.resource == nullptr) {
        return;
    }

    auto maskIter = mMaskedTextures.find(GetBaseTexturePath(metadata.resource->GetInitData()->Identifier.GetPath()));
    if (maskIter == mMaskedTextures.end()) {
        return;
    }

    const uint8_t* orig_addr = maskIter->second.mask;

    if (orig_addr == nullptr) {
        return;
    }

    TextureCacheKey key = { orig_addr, {}, 0, 0, 0, 0 };

    if (TextureCacheLookup(i, key)) {
        return;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes /
                      mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_4b:
            width *= 2;
            break;
        case G_IM_SIZ_8b:
        default:
            break;
        case G_IM_SIZ_16b:
            width /= 2;
            break;
        case G_IM_SIZ_32b:
            width /= 4;
            break;
    }

    for (uint32_t texIndex = 0; texIndex < width * height; texIndex++) {
        uint8_t masked = orig_addr[texIndex];
        if (masked) {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0xFF;
        } else {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0;
        }
    }

    UploadTimed(mRapi, mTexUploadBuffer, width, height);
}

void Interpreter::NormalizeVector(float v[3]) {
    float s = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= s;
    v[1] /= s;
    v[2] /= s;
}

void Interpreter::TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]) {
    res[0] = a[0] * b[0][0] + a[1] * b[0][1] + a[2] * b[0][2];
    res[1] = a[0] * b[1][0] + a[1] * b[1][1] + a[2] * b[1][2];
    res[2] = a[0] * b[2][0] + a[1] * b[2][1] + a[2] * b[2][2];
}

void Interpreter::MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    memcpy(res, tmp, sizeof(tmp));
}

void Interpreter::CalculateNormalDir(const F3DLight_t* light, float coeffs[3]) {
    float light_dir[3] = { light->dir[0] / 127.0f, light->dir[1] / 127.0f, light->dir[2] / 127.0f };

    Interpreter::TransposedMatrixMul(coeffs, light_dir,
                                     mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
    Interpreter::NormalizeVector(coeffs);
}

void Interpreter::GfxSpMatrix(uint8_t parameters, const int32_t* addr) {
    if ((uintptr_t)addr < 0x10000) { return; } // PORT PATCH: unresolved N64 address, skip the command
    float matrix[4][4];

    // PORT: interpolated matrices only for the picture itself; render targets (UI text, portraits) are drawn as the
    // game built them, so their pixels (read back into game memory) never depend on interpolation
    auto it = sInOffscreen ? mCurMtxReplacements->end() : mCurMtxReplacements->find((Mtx*)addr);
    if (it != mCurMtxReplacements->end()) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float v = it->second.mf[i][j];
                int as_int = (int)(v * 65536.0f);
                matrix[i][j] = as_int * (1.0f / 65536.0f);
            }
        }
    } else {
#ifndef GBI_FLOATS
        // Original GBI where fixed point matrices are used
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j += 2) {
                int32_t int_part = addr[i * 2 + j / 2];
                uint32_t frac_part = addr[8 + i * 2 + j / 2];
                matrix[i][j] = (int32_t)((int_part & 0xffff0000) | (frac_part >> 16)) / 65536.0f;
                matrix[i][j + 1] = (int32_t)((int_part << 16) | (frac_part & 0xffff)) / 65536.0f;
            }
        }
#else
        // For a modified GBI where fixed point values are replaced with floats
        memcpy(matrix, addr, sizeof(matrix));
#endif
    }

    if (PortMtxLog()) {
        fprintf(stderr, "[mtxraw]");
        for (int k = 0; k < 16; k++) fprintf(stderr, " %08x", (uint32_t)addr[k]);
        fputc(10, stderr);
    }
    if (gPortDrawLog && (std::fabs(matrix[3][0]) > 1e5f || std::fabs(matrix[3][1]) > 1e5f || std::fabs(matrix[0][0]) > 1e3f ||
                         std::fabs(matrix[1][1]) > 1e3f)) {
        fprintf(stderr, "[badmtx] addr=%p p=%x raw:", (const void*)addr, parameters);
        for (int k = 0; k < 16; k++) fprintf(stderr, " %08x", (uint32_t)addr[k]);
        fprintf(stderr, " diag=%.3g %.3g %.3g t=%.3g %.3g %.3g\n", matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][0], matrix[3][1], matrix[3][2]);
    }
    if (PortMtxLog())
        fprintf(stderr, "[mtx] p=%x addr=%p diag=%.3g %.3g %.3g %.3g t=%.3g %.3g %.3g %.3g r0=%.3g %.3g %.3g\n", parameters, (const void*)addr,
                matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3], matrix[3][0], matrix[3][1], matrix[3][2], matrix[0][3], matrix[0][0], matrix[0][1], matrix[0][2]);
    const int8_t mtx_projection = get_attr(MTX_PROJECTION);
    const int8_t mtx_load = get_attr(MTX_LOAD);
    const int8_t mtx_push = get_attr(MTX_PUSH);

    if (parameters & mtx_projection) {
        if (parameters & mtx_load) {
            memcpy(mRsp->P_matrix, matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->P_matrix, matrix, mRsp->P_matrix);
        }
    } else { // G_MTX_MODELVIEW
        if ((parameters & mtx_push) && mRsp->modelview_matrix_stack_size < 11) {
            ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                   mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 2], sizeof(matrix));
        }
        if (parameters & mtx_load) {
            if (mRsp->modelview_matrix_stack_size == 0)
                ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix,
                      mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
        }
        mRsp->lights_changed = 1;
    }
    MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], mRsp->P_matrix);
}

void Interpreter::GfxSpPopMatrix(uint32_t count) {
    while (count--) {
        if (mRsp->modelview_matrix_stack_size > 0) {
            --mRsp->modelview_matrix_stack_size;
            if (mRsp->modelview_matrix_stack_size > 0) {
                MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                          mRsp->P_matrix);
            }
        }
    }
    mRsp->lights_changed = true;
}

float Interpreter::AdjXForAspectRatio(float x) const {
    // Skip widescreen adjustment for fixed-size off-screen FBs (HUD elements,
    // small capture buffers), or those which specify a fixed aspect ratio.
    if (mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
        (!mActiveFrameBuffer->second.resize || mActiveFrameBuffer->second.forceFixedAspect)) {
        return x;
    } else {
        return x * (4.0f / 3.0f) / ((float)mCurDimensions.width / (float)mCurDimensions.height);
    }
}

// Scale the width and height value based on the ratio of the viewport to the native size
void Interpreter::AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                            uint32_t nativeHeight) const {
    width = round(width * (mCurDimensions.width / (2.0f * (nativeWidth / 2))));
    height = round(height * (mCurDimensions.height / (2.0f * (nativeHeight / 2))));

    if (width == 0) {
        width = 1;
    }
    if (height == 0) {
        height = 1;
    }
}

void Interpreter::GfxSpVertex(size_t n_vertices, size_t dest_index, const F3DVtx* vertices) {
    if ((uintptr_t)vertices < 0x10000) { return; } // PORT PATCH: unresolved N64 address, skip the command
    if (PortTexLog()) { fprintf(stderr, "[tex] vtx n=%zu:", n_vertices); for (size_t q = 0; q < n_vertices && q < 4; q++) fprintf(stderr, " (%d,%d,%d st=%d,%d)", vertices[q].v.ob[0], vertices[q].v.ob[1], vertices[q].v.ob[2], vertices[q].v.tc[0], vertices[q].v.tc[1]); fputc(10, stderr); }
    for (size_t i = 0; i < n_vertices; i++, dest_index++) {
        const F3DVtx_t* v = &vertices[i].v;
        const F3DVtx_tn* vn = &vertices[i].n;
        struct LoadedVertex* d = &mRsp->loaded_vertices[dest_index];

        if (v == nullptr) {
            return;
        }

        float x = v->ob[0] * mRsp->MP_matrix[0][0] + v->ob[1] * mRsp->MP_matrix[1][0] +
                  v->ob[2] * mRsp->MP_matrix[2][0] + mRsp->MP_matrix[3][0];
        float y = v->ob[0] * mRsp->MP_matrix[0][1] + v->ob[1] * mRsp->MP_matrix[1][1] +
                  v->ob[2] * mRsp->MP_matrix[2][1] + mRsp->MP_matrix[3][1];
        float z = v->ob[0] * mRsp->MP_matrix[0][2] + v->ob[1] * mRsp->MP_matrix[1][2] +
                  v->ob[2] * mRsp->MP_matrix[2][2] + mRsp->MP_matrix[3][2];
        float w = v->ob[0] * mRsp->MP_matrix[0][3] + v->ob[1] * mRsp->MP_matrix[1][3] +
                  v->ob[2] * mRsp->MP_matrix[2][3] + mRsp->MP_matrix[3][3];

        float world_pos[3] = { 0.0 };
        if (mRsp->geometry_mode & G_LIGHTING_POSITIONAL) {
            float(*mtx)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            world_pos[0] = v->ob[0] * mtx[0][0] + v->ob[1] * mtx[1][0] + v->ob[2] * mtx[2][0] + mtx[3][0];
            world_pos[1] = v->ob[0] * mtx[0][1] + v->ob[1] * mtx[1][1] + v->ob[2] * mtx[2][1] + mtx[3][1];
            world_pos[2] = v->ob[0] * mtx[0][2] + v->ob[1] * mtx[1][2] + v->ob[2] * mtx[2][2] + mtx[3][2];
        }

        // PORT: 2D geometry in widescreen (GfxSpTri1): remember an orthographic vertex and, on the 4:3 frame's left or
        // right edge, its unsqueezed x (the intro's sky ends at 0.9988)
        d->ortho = mRsp->MP_matrix[0][3] == 0.0f && mRsp->MP_matrix[1][3] == 0.0f && mRsp->MP_matrix[2][3] == 0.0f &&
                   w > 0.0f;
        d->x_edge = d->ortho && fabsf(x) >= 0.99f * w ? x : NAN;
        x = AdjXForAspectRatio(x);

        short U = v->tc[0] * mRsp->texture_scaling_factor.s >> 16;
        short V = v->tc[1] * mRsp->texture_scaling_factor.t >> 16;

        if (mRsp->geometry_mode & G_LIGHTING) {
            if (mRsp->lights_changed) {
                for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                    CalculateNormalDir(&mRsp->current_lights[i].l, mRsp->current_lights_coeffs[i]);
                }
                /*static const Light_t lookat_x = {{0, 0, 0}, 0, {0, 0, 0}, 0, {127, 0, 0}, 0};
                static const Light_t lookat_y = {{0, 0, 0}, 0, {0, 0, 0}, 0, {0, 127, 0}, 0};*/
                CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
                CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
                // PORT: the RSP keeps the transformed lookat directions as signed bytes (s0.7, shifted down, so floored);
                // texgen dots the normals with those. With a large texture scale (Sing's overlay: gSPTexture 0x8000 on a
                // 32x32 texture, ~16 wraps) the float directions' sub-byte changes moved the whole picture by a texel
                // from frame to frame, which the N64 does not do (battle cross-check, PORTING_NOTES.md)
                for (int a = 0; a < 2; a++) {
                    for (int b = 0; b < 3; b++) {
                        mRsp->current_lookat_coeffs[a][b] =
                            fminf(floorf(mRsp->current_lookat_coeffs[a][b] * 128.0f), 127.0f) / 128.0f;
                    }
                }
                mRsp->lights_changed = false;
            }

            int r = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[0];
            int g = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[1];
            int b = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[2];

            for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                float intensity = 0;
                if ((mRsp->geometry_mode & G_LIGHTING_POSITIONAL) && (mRsp->current_lights[i].p.unk3 != 0)) {
                    // Calculate distance from the light to the vertex
                    float dist_vec[3] = { mRsp->current_lights[i].p.pos[0] - world_pos[0],
                                          mRsp->current_lights[i].p.pos[1] - world_pos[1],
                                          mRsp->current_lights[i].p.pos[2] - world_pos[2] };
                    float dist_sq =
                        dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] +
                        dist_vec[2] * dist_vec[2] * 2; // The *2 comes from GLideN64, unsure of why it does it
                    float dist = sqrt(dist_sq);

                    // Transform distance vector (which acts as a direction light vector) into model's space
                    float light_model[3];
                    TransposedMatrixMul(light_model, dist_vec,
                                        mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);

                    // Calculate intensity for each axis using standard formula for intensity
                    float light_intensity[3];
                    for (int light_i = 0; light_i < 3; light_i++) {
                        light_intensity[light_i] = 4.0f * light_model[light_i] / dist_sq;
                        light_intensity[light_i] = std::clamp(light_intensity[light_i], -1.0f, 1.0f);
                    }

                    // Adjust intensity based on surface normal and sum up total
                    float total_intensity =
                        light_intensity[0] * vn->n[0] + light_intensity[1] * vn->n[1] + light_intensity[2] * vn->n[2];
                    total_intensity = std::clamp(total_intensity, -1.0f, 1.0f);

                    // Attenuate intensity based on attenuation values.
                    // Example formula found at https://ogldev.org/www/tutorial20/tutorial20.html
                    // Specific coefficients for MM's microcode sourced from GLideN64
                    // https://github.com/gonetz/GLideN64/blob/3b43a13a80dfc2eb6357673440b335e54eaa3896/src/gSP.cpp#L636
                    float distf = floorf(dist);
                    float attenuation = (distf * mRsp->current_lights[i].p.unk7 * 2.0f +
                                         distf * distf * mRsp->current_lights[i].p.unkE / 8.0f) /
                                            (float)0xFFFF +
                                        1.0f;
                    intensity = total_intensity / attenuation;
                } else {
                    intensity += vn->n[0] * mRsp->current_lights_coeffs[i][0];
                    intensity += vn->n[1] * mRsp->current_lights_coeffs[i][1];
                    intensity += vn->n[2] * mRsp->current_lights_coeffs[i][2];
                    intensity /= 127.0f;
                }
                if (intensity > 0.0f) {
                    r += intensity * mRsp->current_lights[i].l.col[0];
                    g += intensity * mRsp->current_lights[i].l.col[1];
                    b += intensity * mRsp->current_lights[i].l.col[2];
                }
            }

            d->color.r = r > 255 ? 255 : r;
            d->color.g = g > 255 ? 255 : g;
            d->color.b = b > 255 ? 255 : b;

            if (mRsp->geometry_mode & G_TEXTURE_GEN) {
                float dotx = 0, doty = 0;
                dotx += vn->n[0] * mRsp->current_lookat_coeffs[0][0];
                dotx += vn->n[1] * mRsp->current_lookat_coeffs[0][1];
                dotx += vn->n[2] * mRsp->current_lookat_coeffs[0][2];
                doty += vn->n[0] * mRsp->current_lookat_coeffs[1][0];
                doty += vn->n[1] * mRsp->current_lookat_coeffs[1][1];
                doty += vn->n[2] * mRsp->current_lookat_coeffs[1][2];

                dotx /= 127.0f;
                doty /= 127.0f;

                dotx = Ship::Math::clamp(dotx, -1.0f, 1.0f);
                doty = Ship::Math::clamp(doty, -1.0f, 1.0f);

                if (mRsp->geometry_mode & G_TEXTURE_GEN_LINEAR) {
                    // Not sure exactly what formula we should use to get accurate values
                    /*dotx = (2.906921f * dotx * dotx + 1.36114f) * dotx;
                    doty = (2.906921f * doty * doty + 1.36114f) * doty;
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;*/
                    dotx = acosf(-dotx) /* M_PI */ * 0.159155f;
                    doty = acosf(-doty) /* M_PI */ * 0.159155f;
                } else {
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;
                }

                U = (int32_t)(dotx * mRsp->texture_scaling_factor.s);
                V = (int32_t)(doty * mRsp->texture_scaling_factor.t);
            }
        } else {
            d->color.r = v->cn[0];
            d->color.g = v->cn[1];
            d->color.b = v->cn[2];
        }

        d->u = U;
        d->v = V;

        // trivial clip rejection
        d->clip_rej = 0;
        if (x < -w) {
            d->clip_rej |= 1; // CLIP_LEFT
        }
        if (x > w) {
            d->clip_rej |= 2; // CLIP_RIGHT
        }
        if (y < -w) {
            d->clip_rej |= 4; // CLIP_BOTTOM
        }
        if (y > w) {
            d->clip_rej |= 8; // CLIP_TOP
        }
        // if (z < -w) d->clip_rej |= 16; // CLIP_NEAR
        if (z > w) {
            d->clip_rej |= 32; // CLIP_FAR
        }

        d->x = x;
        d->y = y;
        d->z = z;
        d->w = w;

        if (mRsp->geometry_mode & G_FOG) {
            if (fabsf(w) < 0.001f) {
                // To avoid division by zero
                w = 0.001f;
            }

            float winv = 1.0f / w;
            if (winv < 0.0f) {
                winv = std::numeric_limits<int16_t>::max();
            }

            float fog_z = z * winv * mRsp->fog_mul + mRsp->fog_offset;
            fog_z = Ship::Math::clamp(fog_z, 0.0f, 255.0f);
            d->color.a = fog_z; // Use alpha variable to store fog factor
        } else {
            d->color.a = v->cn[3];
        }
    }
}

void Interpreter::GfxSpModifyVertex(uint16_t vtx_idx, uint8_t where, uint32_t val) {
    SUPPORT_CHECK(where == G_MWO_POINT_ST);

    int16_t s = (int16_t)(val >> 16);
    int16_t t = (int16_t)val;

    LoadedVertex* v = &mRsp->loaded_vertices[vtx_idx];
    v->u = s;
    v->v = t;
}

// PORT: whether a colour combiner (as packed by GfxDpSetCombineMode) reads a texture: a full-screen fade in one colour
// does not, a wipe with a soft edge does (GfxSpTri1)
static bool CombinerUsesTexture(uint64_t cm, bool twoCycle) {
    for (int cyc = 0; cyc < (twoCycle ? 2 : 1); cyc++) {
        const uint32_t rgb = (uint32_t)(cm >> (cyc ? 28 : 0)) & 0xffff;
        const uint32_t alpha = (uint32_t)(cm >> (cyc ? 44 : 16)) & 0xfff;
        const uint32_t ca = rgb & 0xf, cb = (rgb >> 4) & 0xf, cc = (rgb >> 8) & 0x1f, cd = (rgb >> 13) & 7;
        for (uint32_t v : { ca, cb, cd }) {
            if (v == G_CCMUX_TEXEL0 || v == G_CCMUX_TEXEL1) return true;
        }
        if (cc == G_CCMUX_TEXEL0 || cc == G_CCMUX_TEXEL1 || cc == G_CCMUX_TEXEL0_ALPHA || cc == G_CCMUX_TEXEL1_ALPHA) {
            return true;
        }
        for (int k = 0; k < 4; k++) {
            const uint32_t v = (alpha >> (3 * k)) & 7;
            if (v == G_ACMUX_TEXEL0 || v == G_ACMUX_TEXEL1) return true;
        }
    }
    return false;
}

void Interpreter::GfxSpTri1(uint8_t vtx1_idx, uint8_t vtx2_idx, uint8_t vtx3_idx, bool is_rect) {
    struct LoadedVertex* v1 = &mRsp->loaded_vertices[vtx1_idx];
    struct LoadedVertex* v2 = &mRsp->loaded_vertices[vtx2_idx];
    struct LoadedVertex* v3 = &mRsp->loaded_vertices[vtx3_idx];
    struct LoadedVertex* v_arr[3] = { v1, v2, v3 };

    // if (rand()%2) return;
    static int sTot = 0, sClip = 0;
    static bool sLog = getenv("PORT_TRILOG") != nullptr;
    if (sLog) {
        sTot++;
        if (v1->clip_rej & v2->clip_rej & v3->clip_rej) sClip++;
        if (sTot % 50 == 0)
            fprintf(stderr, "[tri] total=%d clipped=%d v1=(%.1f %.1f %.1f w=%.1f) rej=%x\n", sTot, sClip, v1->x, v1->y, v1->z, v1->w, v1->clip_rej);
    }

    if (v1->clip_rej & v2->clip_rej & v3->clip_rej) {
        // The whole triangle lies outside the visible area
        return;
    }
    if (!is_rect && !mFbActive && (v1->w != 1.0f || v2->w != 1.0f || v3->w != 1.0f)) {
        sSeen3DThisRun = true; // a perspective triangle in the main picture: not a 2D screen (see s2DRectArea)
    }
    // PORT: 2D (orthographic) triangles in a widescreen picture. A tall one (half the picture height or more) with a
    // vertex on the 4:3 frame's side is a backdrop or a full-screen effect (the intro's sky, the scene fade): that
    // vertex stays on the screen's edge, so it stretches over the wider picture. Others (panels the battle menu slides
    // out) are cut at the 4:3 frame's sides as the N64 screen cut them. PORT_NO_BACKDROP_STRETCH=1: no stretching;
    // PORT_NO_2D_CLIP=1: no cutting.
    static const bool sNoBackdropStretch = getenv("PORT_NO_BACKDROP_STRETCH") != nullptr;
    static const bool sNo2DClip = getenv("PORT_NO_2D_CLIP") != nullptr;
    struct LoadedVertex stretched[3];
    bool clip43 = false;
    bool bothEdges = false; // every vertex on a side of the 4:3 frame, both sides (a full-screen quad, not a wipe)
    if (!is_rect && !mFbActive && v1->ortho && v2->ortho && v3->ortho && AdjXForAspectRatio(1.0f) < 0.999f) {
        float ymin = 1e9f, ymax = -1e9f;
        bool edge = false;
        for (auto* v : v_arr) {
            ymin = std::min(ymin, v->y / v->w);
            ymax = std::max(ymax, v->y / v->w);
            edge |= !std::isnan(v->x_edge);
        }
        bool left = false, right = false, inside = false;
        for (auto* v : v_arr) {
            inside |= std::isnan(v->x_edge);
            left |= !std::isnan(v->x_edge) && v->x_edge < 0.0f;
            right |= !std::isnan(v->x_edge) && v->x_edge > 0.0f;
        }
        bothEdges = left && right && !inside;
        if (edge && ymax - ymin >= 1.0f && !sNoBackdropStretch) {
            for (int i = 0; i < 3; i++) {
                stretched[i] = *v_arr[i];
                if (!std::isnan(stretched[i].x_edge)) {
                    stretched[i].x = stretched[i].x_edge;
                }
                v_arr[i] = &stretched[i];
            }
        } else if (!sNo2DClip && sAnchor == 0) {
            clip43 = true;
        }
    }
    // PORT: a full-screen fade (stretched above: in one colour, from edge to edge) covers the whole picture on a 2D
    // screen too, past its 4:3 lock: over a widened pattern backdrop the sides stayed bright while the middle faded,
    // and between a widescreen screen and a 4:3 one the picture jumped in width in the middle of the fade. Wipes (a
    // textured soft edge, or a shape over part of the screen) keep the lock.
    const bool wideFade = v_arr[0] == &stretched[0] && bothEdges && !sNoWidePattern &&
                          !CombinerUsesTexture(mRdp->combine_mode, (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) ==
                                                                       G_CYC_2CYCLE);
    sFilledWideThisRun |= wideFade;
    static bool sLastClip43 = false;
    if (clip43 != sLastClip43 || wideFade != sWideFade) {
        sLastClip43 = clip43;
        sWideFade = wideFade;
        mRdp->viewport_or_scissor_changed = true;
    }
    if (gPortDrawLog) { // PORT debug: triangles that span far beyond the screen
        float ext = 0;
        for (auto* v : v_arr) {
            if (v->w != 0) ext = std::max(ext, std::max(std::fabs(v->x / v->w), std::fabs(v->y / v->w)));
        }
        if (ext > 4.0f) {
            static int n = 0;
            if (n++ < 12)
                fprintf(stderr, "[bigtri] ext=%.1f tex=%p v=(%.0f %.0f %.0f %.1f) (%.0f %.0f %.0f %.1f) (%.0f %.0f %.0f %.1f)\n", ext,
                        (const void*)mRdp->loaded_texture[0].raw_addr, v1->x, v1->y, v1->z, v1->w, v2->x, v2->y, v2->z, v2->w,
                        v3->x, v3->y, v3->z, v3->w);
        }
    }

    const uint32_t cull_both = get_attr(CULL_BOTH);
    const uint32_t cull_front = get_attr(CULL_FRONT);
    const uint32_t cull_back = get_attr(CULL_BACK);

    if ((mRsp->geometry_mode & cull_both) != 0) {
        float dx1 = v1->x / (v1->w) - v2->x / (v2->w);
        float dy1 = v1->y / (v1->w) - v2->y / (v2->w);
        float dx2 = v3->x / (v3->w) - v2->x / (v2->w);
        float dy2 = v3->y / (v3->w) - v2->y / (v2->w);
        float cross = dx1 * dy2 - dy1 * dx2;

        if ((v1->w < 0) ^ (v2->w < 0) ^ (v3->w < 0)) {
            // If one vertex lies behind the eye, negating cross will give the correct result.
            // If all vertices lie behind the eye, the triangle will be rejected anyway.
            cross = -cross;
        }

        // G_EX_INVERT_CULLING is a LUS extension, not tied to a specific ucode,
        // so apply it regardless of the active microcode handler.
        if ((mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) {
            cross = -cross;
        }

        auto cull_type = mRsp->geometry_mode & cull_both;

        if (cull_type == cull_front) {
            if (cross <= 0) {
                return;
            }
        } else if (cull_type == cull_back) {
            if (cross >= 0) {
                return;
            }
        } else if (cull_type == cull_both) {
            // Why is this even an option?
            return;
        }
    }

    // depth_test is set when the fragment has a depth value to compare (either from vertex Z via
    // RSP G_ZBUFFER, or from the prim-depth register via G_ZS_PRIM) and Z_CMP is requested.
    bool zbuffer_enabled = (mRsp->geometry_mode & G_ZBUFFER) == G_ZBUFFER;
    bool prim_depth_enabled = (mRdp->other_mode_l & G_ZS_PRIM) != 0;
    bool depth_test = (zbuffer_enabled || prim_depth_enabled) && (mRdp->other_mode_l & Z_CMP) == Z_CMP;
    bool depth_mask = (mRdp->other_mode_l & Z_UPD) == Z_UPD;
    uint8_t depth_test_and_mask = (depth_test ? 1 : 0) | (depth_mask ? 2 : 0);
    if (depth_test_and_mask != mRenderingState.depth_test_and_mask) {
        Flush();
        mRapi->SetDepthTestAndMask(depth_test, depth_mask);
        mRenderingState.depth_test_and_mask = depth_test_and_mask;
    }

    bool zmode_decal = (mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC;
    if (zmode_decal != mRenderingState.decal_mode) {
        Flush();
        mRapi->SetZmodeDecal(zmode_decal);
        mRenderingState.decal_mode = zmode_decal;
    }

    const int anchor = !mFbActive && AdjXForAspectRatio(1.0f) < 0.999f ? sAnchor : 0;
    if (mRdp->viewport_or_scissor_changed || anchor != sAppliedAnchor) {
        // PORT: anchored HUD (sAnchor): viewport and scissor move by the margin beside the 4:3 frame
        const int32_t shift = (anchor == 2 ? 0 : anchor) *
                              (int32_t)lroundf(mCurDimensions.width * (1.0f - AdjXForAspectRatio(1.0f)) / 2.0f);
        XYWidthHeight viewport = mRdp->viewport;
        viewport.x = (int16_t)(viewport.x + shift);
        sAppliedAnchor = anchor;
        if (memcmp(&viewport, &mRenderingState.viewport, sizeof(viewport)) != 0) {
            Flush();
            mRapi->SetViewport(viewport.x, viewport.y, viewport.width, viewport.height);
            mRenderingState.viewport = viewport;
        }
        XYWidthHeight scissor = mRdp->scissor;
        if (shift != 0) {
            const int32_t x0 = std::max<int32_t>(0, scissor.x + shift);
            const int32_t x1 = std::max<int32_t>(x0, scissor.x + shift + (int32_t)scissor.width);
            scissor.x = (int16_t)x0;
            scissor.width = (uint32_t)(x1 - x0);
        }
        if ((sLock43ThisRun || clip43) && anchor == 0 && !mFbActive && !sWidePattern && !sWideFade) { // PORT: a 2D screen or 2D shape stays inside the 4:3 area
            const float halfW = mNativeDimensions.width / 2.0f;
            // (rounded: truncated, 79.99998 became 79 and the area ended a pixel short of the picture's right edge,
            // whose last column then kept an older frame: a bright line there while a menu faded out)
            XYWidthHeight area43 = { (int16_t)lroundf(mNativeDimensions.width / 2.0f + AdjXForAspectRatio(-halfW)),
                                     (int16_t)mNativeDimensions.height,
                                     (uint32_t)lroundf(AdjXForAspectRatio(2.0f * halfW)), mNativeDimensions.height };
            AdjustVIewportOrScissor(&area43);
            const int32_t x0 = std::max<int32_t>(scissor.x, area43.x);
            const int32_t x1 = std::min<int32_t>(scissor.x + (int32_t)scissor.width, area43.x + (int32_t)area43.width);
            const int32_t y0 = std::max<int32_t>(scissor.y, area43.y);
            const int32_t y1 = std::min<int32_t>(scissor.y + (int32_t)scissor.height, area43.y + (int32_t)area43.height);
            scissor = { (int16_t)x0, (int16_t)y0, (uint32_t)std::max(0, x1 - x0), (uint32_t)std::max(0, y1 - y0) };
        }
        if (memcmp(&scissor, &mRenderingState.scissor, sizeof(scissor)) != 0) {
            Flush();
            mRapi->SetScissor(scissor.x, scissor.y, scissor.width, scissor.height);
            mRenderingState.scissor = scissor;
        }
        mRdp->viewport_or_scissor_changed = false;
    }

    uint64_t cc_id = mRdp->combine_mode;
    uint64_t cc_options = 0;
    bool use_alpha = ((mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20) &&
                      (mRdp->other_mode_l & (3 << 16)) == (G_BL_1MA << 16)) ||
                     ((mRdp->other_mode_l & (3 << 22)) == (G_BL_CLR_MEM << 22) &&
                      (mRdp->other_mode_l & (3 << 18)) == (G_BL_1MA << 18));
    uint8_t blend_src = mRdp->other_mode_l >> 30;
    bool use_blend_color = blend_src == G_BL_CLR_BL;
    bool use_fog = blend_src == G_BL_CLR_FOG || use_blend_color;
    bool fog_by_fog_alpha = blend_src == G_BL_CLR_FOG && ((mRdp->other_mode_l >> 26) & 3) == G_BL_A_FOG;
    bool texture_edge = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
    bool use_noise = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_DITHER;
    bool use_2cyc = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE;
    static const bool sFloorLog = getenv("PORT_TILE1LOG") != nullptr;
    if (sFloorLog && (cc_id == 0x001ffe35034a8c21ull || cc_id == 0x001ffe3507ff8cf1ull)) {
        static int nf = 0;
        if (nf++ < 100000) fprintf(stderr, "[tile1] FLOOR region=%d\n", TmemFind(mRdp->texture_tile[1].tmem) != nullptr);
    }
    if (gPortDrawLog && sInOffscreen && mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
        mActiveFrameBuffer->second.orig_width == 76) {
        static int no = 0;
        if (no++ < 40) fprintf(stderr, "[offdraw] vp=(%.0f,%.0f %.0fx%.0f) sc=(%.0f,%.0f %.0fx%.0f) v1=(%.1f,%.1f,%.1f,w=%.1f) rej=%x cc=%llx\n",
                                mRdp->viewport.x, mRdp->viewport.y, mRdp->viewport.width, mRdp->viewport.height,
                                mRdp->scissor.x, mRdp->scissor.y, mRdp->scissor.width, mRdp->scissor.height,
                                v1->x, v1->y, v1->z, v1->w, v1->clip_rej, (unsigned long long)cc_id);
    }
    if (gPortDrawLog) { // PORT debug: one line per distinct draw state in the logged frame (Engine.cpp sets it)
        static std::set<std::tuple<uint64_t, uint32_t, uint32_t, const void*, uint32_t>> seen;
        const void* tex = mRdp->loaded_texture[mRdp->texture_tile[mRdp->first_tile_index].tmem_index].addr;
        auto key = std::make_tuple(cc_id, mRdp->other_mode_l, mRdp->other_mode_h, tex,
                                   ((uint32_t)mRdp->prim_color.r << 24) | (mRdp->prim_color.g << 16) | (mRdp->prim_color.b << 8) | mRdp->env_color.r);
        if (cc_id == 0x001ffe35034a8c21ull || cc_id == 0x001ffe3507ff8cf1ull) {
            const TmemRegion* r1 = TmemFind(mRdp->texture_tile[1].tmem);
            fprintf(stderr, "[seq] FLOOR draw cc=%llx tile1 tmem=%u fmt=%u siz=%u line=%u region=%s start=%u len=%u raw=%p v=(%.0f,%.0f)\n",
                    (unsigned long long)cc_id, mRdp->texture_tile[1].tmem, mRdp->texture_tile[1].fmt, mRdp->texture_tile[1].siz,
                    mRdp->texture_tile[1].line_size_bytes, r1 ? "yes" : "no", r1 ? r1->start : 0, r1 ? r1->len : 0,
                    r1 ? (const void*)r1->raw : nullptr, v1->x / v1->w, v1->y / v1->w);
        }
        if (seen.insert(key).second) {
            fprintf(stderr, "[draw] cc=%016llx oml=%08x omh=%08x geo=%08x dec=%d tex=%p fmt=%d siz=%d prim=%02x%02x%02x%02x env=%02x%02x%02x%02x v1=%02x%02x%02x%02x\n",
                    (unsigned long long)cc_id, mRdp->other_mode_l, mRdp->other_mode_h, mRsp->geometry_mode, (int)zmode_decal, tex,
                    mRdp->texture_tile[mRdp->first_tile_index].fmt, mRdp->texture_tile[mRdp->first_tile_index].siz,
                    mRdp->prim_color.r, mRdp->prim_color.g, mRdp->prim_color.b, mRdp->prim_color.a,
                    mRdp->env_color.r, mRdp->env_color.g, mRdp->env_color.b, mRdp->env_color.a,
                    v1->color.r, v1->color.g, v1->color.b, v1->color.a);
            int t1 = (mRdp->first_tile_index + 1) & 7;
            const auto& lt0 = mRdp->loaded_texture[mRdp->texture_tile[mRdp->first_tile_index].tmem_index];
            const auto& lt1 = mRdp->loaded_texture[mRdp->texture_tile[t1].tmem_index];
            fprintf(stderr, "        tile0 %d tmem=%u idx=%d size=%u | tile1 %d tmem=%u idx=%d fmt=%d siz=%d tex=%p size=%u\n",
                    mRdp->first_tile_index, mRdp->texture_tile[mRdp->first_tile_index].tmem, mRdp->texture_tile[mRdp->first_tile_index].tmem_index,
                    lt0.orig_size_bytes, t1, mRdp->texture_tile[t1].tmem, mRdp->texture_tile[t1].tmem_index, mRdp->texture_tile[t1].fmt,
                    mRdp->texture_tile[t1].siz, lt1.addr, lt1.orig_size_bytes);
            for (int q = 0; q < 2; q++) {
                int tt = q ? t1 : mRdp->first_tile_index;
                const TmemRegion* r = TmemFind(mRdp->texture_tile[tt].tmem);
                fprintf(stderr, "        tile%d tmem=%u region: %s start=%u len=%u raw=%p\n", tt, mRdp->texture_tile[tt].tmem,
                        r ? "yes" : "none", r ? r->start : 0, r ? r->len : 0, r ? (const void*)r->raw : nullptr);
                if (r) {
                    const uint8_t* rb = r->data.data() + (mRdp->texture_tile[tt].tmem * 8 - r->start);
                    fprintf(stderr, "          region bytes:");
                    for (int k = 0; k < 16; k++) fprintf(stderr, " %02x", rb[k]);
                    fprintf(stderr, "\n");
                }
            }
            for (int q = 0; q < 2; q++) {
                const uint8_t* b = q ? lt1.addr : lt0.addr;
                if (b && (uintptr_t)b > 0x100000000ull) {
                    fprintf(stderr, "        tex%d bytes:", q);
                    for (int k = 0; k < 24; k++) fprintf(stderr, " %02x", b[k]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }
    bool alpha_threshold = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD;
    // PORT: in copy mode the RDP's alpha compare skips texels whose alpha bit is 0, whatever the blender says (Stadium
    // draws its order-screen icons this way with G_RM_NOOP). The shader only discards when alpha is in use; copy-mode
    // texels have 1-bit alpha, so alpha blending gives the same picture.
    if (alpha_threshold && (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_COPY) {
        use_alpha = true;
    }
    bool invisible =
        (mRdp->other_mode_l & (3 << 24)) == (G_BL_0 << 24) && (mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20);
    // PORT: an invisible (depth-only) draw keeps the memory colour whatever its B input is (G_BL_1MA or G_BL_1):
    // it only works through blending. With B = 1 (the Sandshrew game's hole mask) use_alpha stayed off and the draw
    // painted its colour, black hexagons over the ground.
    if (invisible) {
        use_alpha = true;
    }
    bool use_grayscale = mRdp->grayscale;
    bool use_prim_depth = (mRdp->other_mode_l & G_ZS_PRIM) != 0;

    if (texture_edge) {
        if (use_alpha) {
            alpha_threshold = true;
            texture_edge = false;
        }
        use_alpha = true;
    }

    if (use_alpha) {
        cc_options |= SHADER_OPT(ALPHA);
    }
    if (use_fog) {
        cc_options |= SHADER_OPT(FOG);
    }
    if (texture_edge) {
        cc_options |= SHADER_OPT(TEXTURE_EDGE);
    }
    if (use_noise) {
        cc_options |= SHADER_OPT(NOISE);
    }
    if (use_2cyc) {
        cc_options |= SHADER_OPT(_2CYC);
    }
    if (alpha_threshold) {
        cc_options |= SHADER_OPT(ALPHA_THRESHOLD);
    }
    if (invisible) {
        cc_options |= SHADER_OPT(INVISIBLE);
    }
    if (use_grayscale) {
        cc_options |= SHADER_OPT(GRAYSCALE);
    }
    if (use_prim_depth) {
        cc_options |= SHADER_OPT(PRIM_DEPTH);
    }

    if (!mShaderStack.empty()) {
        cc_options |= (mShaderStack.top() << SHADER_ID_SHIFT);
    } else {
        cc_options |= -1 << SHADER_ID_SHIFT;
    }

    if (mRdp->loaded_texture[0].masked) {
        cc_options |= SHADER_OPT(TEXEL0_MASK);
    }
    if (mRdp->loaded_texture[1].masked) {
        cc_options |= SHADER_OPT(TEXEL1_MASK);
    }
    if (mRdp->loaded_texture[0].blended) {
        cc_options |= SHADER_OPT(TEXEL0_BLEND);
    }
    if (mRdp->loaded_texture[1].blended) {
        cc_options |= SHADER_OPT(TEXEL1_BLEND);
    }

    ColorCombinerKey key;
    key.combine_mode = mRdp->combine_mode;
    key.options = cc_options;

    ColorCombiner* comb = LookupOrCreateColorCombiner(key);

    uint32_t tm = 0;
    uint32_t tex_width[2], tex_height[2], tex_width2[2], tex_height2[2];
    float rttRow[2] = { 0, 0 };
    float rttCol[2] = { 0, 0 };
    uint32_t effective_tile[2];

    for (int i = 0; i < 2; i++) {
        uint32_t tile = mRdp->first_tile_index + i;

        // No LOD support: force both slots to the base mip level.
        if (i == 1 && mRdp->first_tile_index >= 2) {
            tile = mRdp->first_tile_index;
        }
        effective_tile[i] = tile;

        if (comb->usedTextures[i]) {
            if (gPortDrawLog && i == 1) { // PORT-TRACE: draws that sample the second tile
                static std::set<std::tuple<uint64_t, uint32_t, const void*>> seen1;
                const TmemRegion* r = TmemFind(mRdp->texture_tile[tile].tmem);
                if (seen1.insert({ cc_id, mRdp->texture_tile[tile].tmem, r ? (const void*)r->raw : nullptr }).second) {
                    fprintf(stderr, "[tex1] cc=%016llx comb=%08x%08x tile%u tmem=%u fmt=%u siz=%u line=%u region=%s start=%u len=%u raw=%p tile0 tmem=%u v=(%.0f,%.0f)\n",
                            (unsigned long long)cc_id, (uint32_t)(mRdp->combine_mode >> 32), (uint32_t)mRdp->combine_mode, tile,
                            mRdp->texture_tile[tile].tmem, mRdp->texture_tile[tile].fmt, mRdp->texture_tile[tile].siz,
                            mRdp->texture_tile[tile].line_size_bytes, r ? "yes" : "no", r ? r->start : 0, r ? r->len : 0,
                            r ? (const void*)r->raw : nullptr, mRdp->texture_tile[mRdp->first_tile_index].tmem,
                            v1->x / v1->w, v1->y / v1->w);
                    const auto& ts = mRdp->texture_tile[tile];
                    fprintf(stderr, "       tile1 uls=%.0f ult=%.0f lrs=%.0f lrt=%.0f cms=%u cmt=%u masks=%u maskt=%u shifts=%u shiftt=%u\n",
                            (double)ts.uls, (double)ts.ult, (double)ts.lrs, (double)ts.lrt, ts.cms, ts.cmt, ts.masks, ts.maskt, ts.shifts, ts.shiftt);
                    if (r) {
                        const uint8_t* b = r->data.data() + (mRdp->texture_tile[tile].tmem * 8 - r->start);
                        for (int row = 0; row < 4; row++) {
                            fprintf(stderr, "       ");
                            for (int k = 0; k < 16; k++) fprintf(stderr, " %02x", b[row * 16 + k]);
                            fprintf(stderr, "\n");
                        }
                    }
                }
            }
            if (mRdp->textures_changed[i]) {
                Flush();
                RttInfo rtt;
                float rttRowTmp;
                if (i == 0 && !sInOffscreen && RttLookup(mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_addr, &rtt, &rttRowTmp)) {
                    if (PortTexLog()) fprintf(stderr, "[rtt] bind fb %d (%ux%u) for %p row %.0f\n", rtt.fb, rtt.w, rtt.h, mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_addr, rttRowTmp);
                    mRapi->SelectTextureFb(rtt.fb); // render-to-texture buffer: bind the GPU framebuffer instead of uploading RAM
                } else {
                    ImportTexture(i, tile, false);
                }
                if (mRdp->loaded_texture[i].masked) {
                    ImportTextureMask(SHADER_FIRST_MASK_TEXTURE + i, tile);
                }
                if (mRdp->loaded_texture[i].blended) {
                    ImportTexture(SHADER_FIRST_REPLACEMENT_TEXTURE + i, tile, true);
                }
                mRdp->textures_changed[i] = false;
            }

            uint8_t cms = mRdp->texture_tile[tile].cms;
            uint8_t cmt = mRdp->texture_tile[tile].cmt;

            uint32_t loaded_line_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
            uint32_t loaded_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
            uint32_t loaded_full_line =
                mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
            uint32_t tex_size_bytes;
            uint32_t line_size;
            if ((loaded_line_size != loaded_size || loaded_full_line != loaded_size) && loaded_line_size > 0) {
                line_size = loaded_line_size;
                tex_size_bytes = loaded_size;
            } else {
                line_size = mRdp->texture_tile[tile].line_size_bytes;
                tex_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
                // RGBA32: texture_tile stores TMEM-interleaved stride (half of actual DRAM stride).
                if (mRdp->texture_tile[tile].siz == G_IM_SIZ_32b) {
                    line_size *= 2;
                }
            }

            if (line_size == 0) {
                line_size = 1;
            }

            tex_height[i] = tex_size_bytes / line_size;
            switch (mRdp->texture_tile[tile].siz) {
                case G_IM_SIZ_4b:
                    line_size <<= 1;
                    break;
                case G_IM_SIZ_8b:
                    break;
                case G_IM_SIZ_16b:
                    line_size /= G_IM_SIZ_16b_LINE_BYTES;
                    break;
                case G_IM_SIZ_32b:
                    line_size /= 4; // RGBA32: 4 bytes per pixel (line_size is now actual DRAM stride)
                    break;
            }
            tex_width[i] = line_size;

            tex_width2[i] = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
            tex_height2[i] = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);

            // Same pyramid-like ratio gate as ImportTexture: only clamp when loaded pixels
            // are close to rendered pixels (mipmap), not when much bigger (window scroll).
            uint32_t loadedPx = tex_width[i] * tex_height[i];
            uint32_t renderedPx = tex_width2[i] * tex_height2[i];
            bool pyrLike = renderedPx > 0 && loadedPx > renderedPx && loadedPx * 8 < renderedPx * 13;
            // Same wrap-period trim as the import paths. The >= tex_width2 guard skips a stale
            // mask left by an FB blit (the pause background), which would otherwise tile the FB.
            uint32_t maskW = mRdp->texture_tile[tile].masks;
            uint32_t maskH = mRdp->texture_tile[tile].maskt;
            if (maskW != 0 && (1u << maskW) >= tex_width2[i] && (1u << maskW) < tex_width[i]) {
                tex_width[i] = 1u << maskW;
            }
            if (maskH != 0 && (1u << maskH) >= tex_height2[i] && (1u << maskH) < tex_height[i]) {
                tex_height[i] = 1u << maskH;
            }
            // HD replacements must clamp to the tile region
            bool isHd = mRdp->loaded_texture[i].raw_tex_metadata.h_byte_scale != 1 ||
                        mRdp->loaded_texture[i].raw_tex_metadata.v_pixel_scale != 1;
            if ((isHd || pyrLike || (cms & G_TX_CLAMP)) && tex_width2[i] > 0 && tex_width2[i] < tex_width[i]) {
                tex_width[i] = tex_width2[i];
            }
            if ((isHd || pyrLike || (cmt & G_TX_CLAMP)) && tex_height2[i] > 0 && tex_height2[i] < tex_height[i]) {
                tex_height[i] = tex_height2[i];
            }

            rttRow[i] = 0;
            rttCol[i] = 0;
            {
                RttInfo rtt;
                float row, col = 0;
                if (i == 0 && !sInOffscreen && RttLookup(mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_addr, &rtt, &row, &col)) {
                    tex_width[i] = rtt.w; // UVs are relative to the whole buffer, the tile starts at (col, row)
                    tex_height[i] = rtt.h;
                    rttRow[i] = row;
                    rttCol[i] = col;
                }
            }
            uint32_t tex_width1 = tex_width[i] << (cms & G_TX_MIRROR);
            uint32_t tex_height1 = tex_height[i] << (cmt & G_TX_MIRROR);

            if ((cms & G_TX_CLAMP) && ((cms & G_TX_MIRROR) || tex_width1 != tex_width2[i])) {
                tm |= 1 << 2 * i;
                cms &= ~G_TX_CLAMP;
            }
            if ((cmt & G_TX_CLAMP) && ((cmt & G_TX_MIRROR) || tex_height1 != tex_height2[i])) {
                tm |= 1 << 2 * i + 1;
                cmt &= ~G_TX_CLAMP;
            }

            if (mRenderingState.mTextures[i] == nullptr) {
                continue;
            }

            bool linear_filter = (mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT;
            if (linear_filter != mRenderingState.mTextures[i]->second.linear_filter ||
                cms != mRenderingState.mTextures[i]->second.cms || cmt != mRenderingState.mTextures[i]->second.cmt) {
                Flush();

                // Set the same sampler params on the blended texture. Needed for opengl.
                if (mRdp->loaded_texture[i].blended) {
                    mRapi->SetSamplerParameters(SHADER_FIRST_REPLACEMENT_TEXTURE + i, linear_filter, cms, cmt);
                }

                mRapi->SetSamplerParameters(i, linear_filter, cms, cmt);
                mRenderingState.mTextures[i]->second.linear_filter = linear_filter;
                mRenderingState.mTextures[i]->second.cms = cms;
                mRenderingState.mTextures[i]->second.cmt = cmt;
            }
        }
    }

    struct ShaderProgram* prg = comb->prg[tm];
    if (prg == NULL) {
        comb->prg[tm] = prg =
            LookupOrCreateShaderProgram(comb->shader_id0, comb->shader_id1 | tm * SHADER_OPT(TEXEL0_CLAMP_S));
    }
    if (prg != mRenderingState.mShaderProgram) {
        Flush();
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        mRapi->LoadShader(prg);
        mRenderingState.mShaderProgram = prg;
    }
    if (use_alpha != mRenderingState.alpha_blend) {
        Flush();
        mRapi->SetUseAlpha(use_alpha);
        mRenderingState.alpha_blend = use_alpha;
    }
    uint8_t numInputs;
    bool usedTextures[2];

    mRapi->ShaderGetInfo(prg, &numInputs, usedTextures);

    struct GfxClipParameters clip_parameters = mRapi->GetClipParameters();

    for (int i = 0; i < 3; i++) {
        float z = v_arr[i]->z, w = v_arr[i]->w;
        if (clip_parameters.z_is_from_0_to_1) {
            z = (z + w) / 2.0f;
        }

        mBufVbo[mBufVboLen++] = v_arr[i]->x;
        mBufVbo[mBufVboLen++] = clip_parameters.invertY ? -v_arr[i]->y : v_arr[i]->y;
        mBufVbo[mBufVboLen++] = z;
        mBufVbo[mBufVboLen++] = w;

        for (int t = 0; t < 2; t++) {
            if (!usedTextures[t]) {
                continue;
            }
            float u = v_arr[i]->u / 32.0f;
            float v = v_arr[i]->v / 32.0f;

            uint32_t uv_tile = effective_tile[t];
            int shifts = mRdp->texture_tile[uv_tile].shifts;
            int shiftt = mRdp->texture_tile[uv_tile].shiftt;
            if (shifts != 0) {
                if (shifts <= 10) {
                    u /= 1 << shifts;
                } else {
                    u *= 1 << (16 - shifts);
                }
            }
            if (shiftt != 0) {
                if (shiftt <= 10) {
                    v /= 1 << shiftt;
                } else {
                    v *= 1 << (16 - shiftt);
                }
            }

            u -= mRdp->texture_tile[uv_tile].uls / 4.0f;
            v -= mRdp->texture_tile[uv_tile].ult / 4.0f;
            v += rttRow[t];
            u += rttCol[t];

            if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) {
                // Linear filter adds 0.5f to the coordinates
                if (!is_rect) {
                    u += 0.5f;
                    v += 0.5f;
                }
            }

            mBufVbo[mBufVboLen++] = u / tex_width[t];
            mBufVbo[mBufVboLen++] = v / tex_height[t];

            bool clampS = tm & (1 << 2 * t);
            bool clampT = tm & (1 << 2 * t + 1);

            if (clampS) {
                mBufVbo[mBufVboLen++] = (tex_width2[t] - 0.5f + rttCol[t]) / tex_width[t];
            }

            if (clampT) {
                mBufVbo[mBufVboLen++] = (tex_height2[t] - 0.5f + rttRow[t]) / tex_height[t];
            }
        }

        if (use_fog) {
            if (use_blend_color) {
                // Shroud/blend mode: blend toward blend_color using fog alpha as factor
                mBufVbo[mBufVboLen++] = mRdp->blend_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.b / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.a / 255.0f;
            } else {
                mBufVbo[mBufVboLen++] = mRdp->fog_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.b / 255.0f;
                if (fog_by_fog_alpha) {
                    // PORT: G_RM_FOG_PRIM_A (CLR_FOG * A_FOG): a constant tint by the fog register's alpha, e.g. the
                    // battle status flashes (burn, freeze, ...). Shade alpha would paint the model solid fog colour.
                    mBufVbo[mBufVboLen++] = mRdp->fog_color.a / 255.0f;
                } else if (is_rect) {
                    // PORT: rectangles have no shade on the RDP (0), so a blender that fogs by shade alpha leaves
                    // them as they are (the Pokédex area map's label boxes, drawn with the map's fog render mode)
                    mBufVbo[mBufVboLen++] = 0.0f;
                } else {
                    mBufVbo[mBufVboLen++] = v_arr[i]->color.a / 255.0f; // fog factor (not alpha)
                }
            }
        }

        if (use_grayscale) {
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.r / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.g / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.b / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.a / 255.0f; // lerp interpolation factor (not alpha)
        }

        for (int j = 0; j < numInputs; j++) {
            RGBA* color;
            RGBA tmp;
            for (int k = 0; k < 1 + (use_alpha ? 1 : 0); k++) {
                switch (comb->shader_input_mapping[k][j]) {
                        // Note: CCMUX constants and ACMUX constants used here have same value, which is why this works
                        // (except LOD fraction).
                    case G_CCMUX_PRIMITIVE:
                        color = &mRdp->prim_color;
                        break;
                    case G_CCMUX_SHADE:
                        color = &v_arr[i]->color;
                        break;
                    case G_CCMUX_ENVIRONMENT:
                        color = &mRdp->env_color;
                        break;
                    case G_CCMUX_PRIMITIVE_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_ENV_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->env_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_PRIM_LOD_FRAC: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_LOD_FRACTION: {
                        if (mRdp->other_mode_l & G_TL_LOD) {
                            // "Hack" that works for Bowser - Peach painting
                            float distance_frac = (v1->w - 3000.0f) / 3000.0f;
                            if (distance_frac < 0.0f) {
                                distance_frac = 0.0f;
                            }
                            if (distance_frac > 1.0f) {
                                distance_frac = 1.0f;
                            }
                            tmp.r = tmp.g = tmp.b = tmp.a = distance_frac * 255.0f;
                        } else {
                            tmp.r = tmp.g = tmp.b = tmp.a = 255.0f;
                        }
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_KEY_CENTER:
                        color = &mRdp->key_center;
                        break;
                    case G_CCMUX_KEY_SCALE:
                        color = &mRdp->key_scale;
                        break;
                    case G_CCMUX_CONVERT_K4: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[4];
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_CONVERT_K5: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[5];
                        color = &tmp;
                        break;
                    }
                    case G_ACMUX_PRIM_LOD_FRAC:
                        tmp.a = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    default:
                        memset(&tmp, 0, sizeof(tmp));
                        color = &tmp;
                        break;
                }
                if (k == 0) {
                    mBufVbo[mBufVboLen++] = color->r / 255.0f;
                    mBufVbo[mBufVboLen++] = color->g / 255.0f;
                    mBufVbo[mBufVboLen++] = color->b / 255.0f;
                } else {
                    if (use_fog && !use_blend_color && color == &v_arr[i]->color) {
                        // Shade alpha is 100% for standard fog, blend color mode preserves
                        // it since fog alpha is the blend factor
                        mBufVbo[mBufVboLen++] = 1.0f;
                    } else {
                        mBufVbo[mBufVboLen++] = color->a / 255.0f;
                    }
                }
            }
        }

        // struct RGBA *color = &v_arr[i]->color;
        // mBufVbo[mBufVboLen++] = color->r / 255.0f;
        // mBufVbo[mBufVboLen++] = color->g / 255.0f;
        // mBufVbo[mBufVboLen++] = color->b / 255.0f;
        // mBufVbo[mBufVboLen++] = color->a / 255.0f;
    }

    if (++mBufVboNumTris == MAX_TRI_BUFFER) {
        // if (++mBufVbo_num_tris == 1) {
        Flush();
    }
}

void Interpreter::GfxSpGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->geometry_mode &= ~clear;
    mRsp->geometry_mode |= set;
}

void Interpreter::GfxSpExtraGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->extra_geometry_mode &= ~clear;
    mRsp->extra_geometry_mode |= set;
}

void Interpreter::AdjustVIewportOrScissor(XYWidthHeight* area) {
    if (!mFbActive) {
        // Adjust the y origin based on the y-inversion for the active framebuffer
        GfxClipParameters clipParameters = mRapi->GetClipParameters();
        if (clipParameters.invertY) {
            area->y -= area->height;
        } else {
            area->y = mNativeDimensions.height - area->y;
        }

        // Scale by computing both edges and rounding them outward: the fields
        // are integers, so scaling width/x independently truncates and can
        // leave the right/bottom edge up to 2px short of the true edge
        // (visible as an unrendered column at the window edge).
        {
            const float rx = RATIO_X(mActiveFrameBuffer, mCurDimensions);
            const float ry = RATIO_Y(mActiveFrameBuffer, mCurDimensions);
            const float x2 = (area->x + (float)area->width) * rx;
            const float y2 = (area->y + (float)area->height) * ry;
            area->x = (int16_t)floorf(area->x * rx);
            area->y = (int16_t)floorf(area->y * ry);
            area->width = (uint32_t)(ceilf(x2) - area->x);
            area->height = (uint32_t)(ceilf(y2) - area->y);
        }

        if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                              mCurDimensions.height == mGameWindowViewport.height)) {
            area->x += mGameWindowViewport.x;
            area->y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
        }
    } else {
        area->y = mActiveFrameBuffer->second.orig_height - area->y;

        if (mActiveFrameBuffer->second.resize) {
            // Same as above.
            const float rx = RATIO_X(mActiveFrameBuffer, mCurDimensions);
            const float ry = RATIO_Y(mActiveFrameBuffer, mCurDimensions);
            const float x2 = (area->x + (float)area->width) * rx;
            const float y2 = (area->y + (float)area->height) * ry;
            area->x = (int16_t)floorf(area->x * rx);
            area->y = (int16_t)floorf(area->y * ry);
            area->width = (uint32_t)(ceilf(x2) - area->x);
            area->height = (uint32_t)(ceilf(y2) - area->y);
        }
    }
}

void Interpreter::CalcAndSetViewport(const F3DVp_t* viewport) {
    // 2 bits fraction
    float width = 2.0f * viewport->vscale[0] / 4.0f;
    float height = 2.0f * viewport->vscale[1] / 4.0f;
    float x = (viewport->vtrans[0] / 4.0f) - width / 2.0f;
    float y = ((viewport->vtrans[1] / 4.0f) + height / 2.0f);

    mRdp->viewport.x = x;
    mRdp->viewport.y = y;
    mRdp->viewport.width = width;
    mRdp->viewport.height = height;

    AdjustVIewportOrScissor(&mRdp->viewport);
    if (gPortDrawLog && mFbActive && mActiveFrameBuffer != mFrameBuffers.end() && mActiveFrameBuffer->second.orig_width == 76)
        fprintf(stderr, "[offvp] vscale=%d,%d vtrans=%d,%d -> vp=(%d,%d %ux%u)\n", viewport->vscale[0], viewport->vscale[1],
                viewport->vtrans[0], viewport->vtrans[1], (int)mRdp->viewport.x, (int)mRdp->viewport.y,
                (unsigned)mRdp->viewport.width, (unsigned)mRdp->viewport.height);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data) {
    if ((uintptr_t)data < 0x10000) { return; } // PORT PATCH: unresolved N64 address, skip the command
    switch (index) {
        case F3DEX2_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX2_G_MV_LIGHT: {
            int lightidx = offset / 24 - 2;
            if (lightidx >= 0 && lightidx <= MAX_LIGHTS) { // skip lookat
                // NOTE: reads out of bounds if it is an ambient light
                memcpy(mRsp->current_lights + lightidx, data, sizeof(F3DLight));
            } else if (lightidx < 0) {
                memcpy(mRsp->lookat + offset / 24, data, sizeof(F3DLight_t)); // TODO Light?
            }
            break;
        }
    }
}

void Interpreter::GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX_G_MV_LOOKATY:
        case F3DEX_G_MV_LOOKATX:
            memcpy(mRsp->lookat + (index - F3DEX_G_MV_LOOKATY) / 2, data, sizeof(F3DLight_t));
            break;
        case F3DEX_G_MV_L0:
        case F3DEX_G_MV_L1:
        case F3DEX_G_MV_L2:
        case F3DEX_G_MV_L3:
        case F3DEX_G_MV_L4:
        case F3DEX_G_MV_L5:
        case F3DEX_G_MV_L6:
        case F3DEX_G_MV_L7:
            // NOTE: reads out of bounds if it is an ambient light
            memcpy(mRsp->current_lights + (index - F3DEX_G_MV_L0) / 2, data, sizeof(F3DLight_t));
            break;
    }
}

void Interpreter::GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            mRsp->current_num_lights = data / 24 + 1; // add ambient light
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            if (segNumber == 15 && PortTexLog()) fprintf(stderr, "[seg] 15 = 0x%lx\n", (unsigned long)data);
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            // Ambient light is included
            // The 31st bit is a flag that lights should be recalculated
            mRsp->current_num_lights = (data - 0x80000000U) / 32;
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on) {
    if (PortTexLog()) fprintf(stderr, "[tex] texture sc=%x tc=%x tile=%u on=%u\n", sc, tc, tile, on);
    mRsp->texture_scaling_factor.s = sc;
    mRsp->texture_scaling_factor.t = tc;
    if (mRdp->first_tile_index != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }

    mRdp->first_tile_index = tile;
}

void Interpreter::GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry) {
    if (sPendingOffscreenAddr != 0) {
        uint32_t w = sPendingOffscreenWidth, h = (lry - uly) / 4;
        if (h == 0) h = 16;
        int id = CreateFrameBuffer(w, h, w, h, false, false);
        mRapi->SetFramebufferReadback(id);
        RttApplyForgets();
        RttForgetOverlapping(sPendingOffscreenAddr, sPendingOffscreenAddr + (uintptr_t)w * h * 2);
        static const bool initializeAll = getenv("PORT_RTT_INIT_ALL") != nullptr; // comparison with the broad upload path
        if (sPendingOffscreenPreserve || initializeAll) {
            RttInitialize(id, sPendingOffscreenAddr, w, h);
        }
        SetFrameBuffer(id, 1.0f);
        mActiveFrameBuffer = mFrameBuffers.find(id);
        mFbActive = true;
        sInOffscreen = true;
        // a new GPU target starts without viewport/scissor: send them again even if they equal the cached ones
        mRdp->viewport_or_scissor_changed = true;
        mRenderingState.viewport = {};
        mRenderingState.scissor = {};
        sOffscreenFbs[sPendingOffscreenAddr] = RttInfo{ id, w, h, {} };
        RttStamp(sPendingOffscreenAddr, &sOffscreenFbs[sPendingOffscreenAddr]);
        sCurOffscreenAddr = sPendingOffscreenAddr;
        sPendingOffscreenAddr = 0;
        if (PortTexLog()) fprintf(stderr, "[rtt] created fb %d for %ux%u\n", id, w, h);
    }
    float x = ulx / 4.0f;
    float y = lry / 4.0f;
    float width = (lrx - ulx) / 4.0f;
    float height = (lry - uly) / 4.0f;

    mRdp->scissor.x = x;
    mRdp->scissor.y = y;
    mRdp->scissor.width = width;
    mRdp->scissor.height = height;

    AdjustVIewportOrScissor(&mRdp->scissor);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath,
                                       uint32_t texFlags, RawTexMetadata rawTexMetdata, const void* addr) {
    if (PortTexLog()) fprintf(stderr, "[tex] timg fmt=%u siz=%u w=%u addr=%p\n", format, size, width, addr);
    // fprintf(stderr, "GfxDpSetTextureImage: %s (width=%d; size=0x%X)\n",
    //         rawTexMetdata.resource ? rawTexMetdata.resource->GetInitData()->Identifier.GetPath().c_str() : nullptr,
    //         width, size);
    mRdp->texture_to_load.addr = (const uint8_t*)addr;
    mRdp->texture_to_load.siz = size;
    mRdp->texture_to_load.width = width;
    mRdp->texture_to_load.tex_flags = texFlags;
    mRdp->texture_to_load.raw_tex_metadata = rawTexMetdata;
}

void Interpreter::GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                               uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks,
                               uint32_t shifts) {
    if (PortTexLog())
        fprintf(stderr, "[tex] tile%u fmt=%u siz=%u line=%u tmem=%u pal=%u cms=%u masks=%u cmt=%u maskt=%u\n", tile, fmt,
                siz, line, tmem, palette, cms, masks, cmt, maskt);
    // OTRTODO:
    // SUPPORT_CHECK(tmem == 0 || tmem == 256);

    if (cms == G_TX_WRAP && masks == G_TX_NOMASK) {
        cms = G_TX_CLAMP;
    }
    if (cmt == G_TX_WRAP && maskt == G_TX_NOMASK) {
        cmt = G_TX_CLAMP;
    }

    mRdp->texture_tile[tile].palette = palette; // palette should set upper 4 bits of color index in 4b mode
    mRdp->texture_tile[tile].fmt = fmt;
    mRdp->texture_tile[tile].siz = siz;
    mRdp->texture_tile[tile].cms = cms;
    mRdp->texture_tile[tile].cmt = cmt;
    mRdp->texture_tile[tile].masks = masks;
    mRdp->texture_tile[tile].maskt = maskt;
    mRdp->texture_tile[tile].shifts = shifts;
    mRdp->texture_tile[tile].shiftt = shiftt;
    mRdp->texture_tile[tile].line_size_bytes = line * 8;

    mRdp->texture_tile[tile].tmem = tmem;
    if (gPortDrawLog) fprintf(stderr, "[seq] settile %u fmt=%u siz=%u line=%u tmem=%u\n", tile, fmt, siz, line, tmem);
    static const bool sT1 = getenv("PORT_TILE1LOG") != nullptr;
    if (sT1 && (tile == 1 || tmem != 0)) {
        static int n = 0;
        if (n++ < 400) fprintf(stderr, "[tile1] settile %u fmt=%u siz=%u line=%u tmem=%u\n", tile, fmt, siz, line, tmem);
    }
    // mRdp->texture_tile[tile].tmem_index = tmem / 256; // tmem is the 64-bit word offset, so 256 words means 2 kB

    mRdp->texture_tile[tile].tmem_index =
        tmem != 0; // assume one texture is loaded at address 0 and another texture at any other address

    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
    if (PortTexLog()) fprintf(stderr, "[tex] tsize%u %u %u %u %u\n", tile, uls, ult, lrs, lrt);
    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;
    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadTlut(uint8_t tile, uint32_t high_index) {
    SUPPORT_CHECK(mRdp->texture_to_load.siz == G_IM_SIZ_16b);

    uint16_t tmem = mRdp->texture_tile[tile].tmem;
    const uint8_t* src = mRdp->texture_to_load.addr;
    uint32_t entryCount = high_index + 1;
    uint32_t byteCount = entryCount * 2;
    if (gPortDrawLog) fprintf(stderr, "[seq] tlut tmem=%u entries=%u src=%p\n", tmem * 8, entryCount, (const void*)src);
    // PORT: a palette in a C u16/u32 array is little-endian on the host (see WordArrayAt): copy it in N64 order
    static uint8_t sTlutCopy[512];
    {
        size_t off;
        int ws = (src != nullptr && byteCount <= sizeof(sTlutCopy)) ? WordArrayAt(src, &off) : 0;
        if (ws != 0) {
            memcpy(sTlutCopy, src, byteCount);
            SwapWordsFrom(sTlutCopy, byteCount, ws, off);
            src = sTlutCopy;
        }
    }

    if (tmem >= 256) {
        // N64 TMEM palette area starts at tmem word 256. Each CI4 palette = 16 entries = 16 tmem words.
        uint32_t paletteByteOffset = (tmem - 256) * 2;

        if (high_index == 255 && paletteByteOffset == 0) {
            // CI8: full 256-entry palette spanning both halves
            memcpy(mRdp->palette_staging[0], src, 256);
            memcpy(mRdp->palette_staging[1], src + 256, 256);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[0] = src;
            mRdp->palette_dram_addr[1] = src + 256;
        } else if (paletteByteOffset < 256) {
            // Palettes 0-7 range
            uint32_t copyLen = (paletteByteOffset + byteCount <= 256) ? byteCount : (256 - paletteByteOffset);
            memcpy(mRdp->palette_staging[0] + paletteByteOffset, src, copyLen);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palette_dram_addr[0] = src;
        } else {
            // Palettes 8-15 range
            uint32_t offset = paletteByteOffset - 256;
            uint32_t copyLen = (offset + byteCount <= 256) ? byteCount : (256 - offset);
            memcpy(mRdp->palette_staging[1] + offset, src, copyLen);
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[1] = src;
        }
    } else {
        // tmem < 256: non-standard location, fall back to direct pointer
        mRdp->palettes[1] = src;
        mRdp->palette_dram_addr[1] = src;
    }
}

static void PortDumpTex(const uint8_t* addr, uint32_t lrs, uint32_t siz, uint32_t dxt) {
    static const char* dir = getenv("PORT_TEXDUMP");
    static int n = 0;
    if (!dir || siz != 2 || lrs != 2047 || dxt != (getenv("PORT_TEXDUMP_DXT") ? (uint32_t)atoi(getenv("PORT_TEXDUMP_DXT")) : 64u)) return; // keeps a ring of the 40 most recent
    char path[512];
    snprintf(path, sizeof path, "%s/tex_%02d.bin", dir, n++ % 40);
    if (FILE* f = fopen(path, "wb")) { fwrite(addr, 1, 4096, f); fclose(f); }
}

void Interpreter::GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt) {
    PortDumpTex(mRdp->texture_to_load.addr, lrs, mRdp->texture_to_load.siz, dxt);
    if (PortTexLog()) { fprintf(stderr, "[tex] loadblock t%u siz=%u lrs=%u dxt=%u data:", tile, mRdp->texture_to_load.siz, lrs, dxt); for (int k = 0; k < 24; k++) fprintf(stderr, "%02x", mRdp->texture_to_load.addr[k]); fputc(10, stderr); }
    SUPPORT_CHECK(uls == 0);
    SUPPORT_CHECK(ult == 0);

    // The lrs field rather seems to be number of pixels to load
    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = -1;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }
    uint32_t orig_size_bytes =
        word_size_shift > 0 ? (lrs + 1) << word_size_shift : (lrs + 1) >> (-(int64_t)word_size_shift);
    uint32_t size_bytes = orig_size_bytes;
    if (mRdp->texture_to_load.raw_tex_metadata.h_byte_scale != 1 ||
        mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale != 1) {
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;
    }
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    // Compute actual per-line DRAM stride from SetTextureImage width when available.
    // The standard gDPLoadTextureBlock macro sets width=1, but manually-built DL
    // commands may set the real pixel width.
    uint32_t actual_line_bytes = size_bytes;
    const RawTexMetadata& blkMeta = mRdp->texture_to_load.raw_tex_metadata;
    bool blkHd = blkMeta.h_byte_scale != 1 || blkMeta.v_pixel_scale != 1;
    if (mRdp->texture_to_load.width > 1 && blkHd && blkMeta.height > 0 && size_bytes % blkMeta.height == 0) {
        // HD-upscaled textures report a sentinel SetTextureImage width, so the per-line stride
        // derived from it is bogus. The resource's stored height is authoritative.
        actual_line_bytes = size_bytes / blkMeta.height;
    } else if (mRdp->texture_to_load.width > 1) {
        uint32_t candidate;
        switch (mRdp->texture_to_load.siz) {
            case G_IM_SIZ_4b:
                candidate = (mRdp->texture_to_load.width + 1) / 2;
                break;
            case G_IM_SIZ_8b:
                candidate = mRdp->texture_to_load.width;
                break;
            case G_IM_SIZ_16b:
                candidate = mRdp->texture_to_load.width * 2;
                break;
            case G_IM_SIZ_32b:
                candidate = mRdp->texture_to_load.width * 4;
                break;
            default:
                candidate = mRdp->texture_to_load.width;
                break;
        }
        if (candidate > 0 && candidate < size_bytes && size_bytes % candidate == 0) {
            actual_line_bytes = candidate;
        }
    }
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = actual_line_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = actual_line_bytes;
    // assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_addr = mRdp->texture_to_load.addr;
    if (mRdp->texture_to_load.raw_tex_metadata.resource == nullptr) { // PORT: snapshot RAM textures (TMEM semantics)
        int slot = mRdp->texture_tile[tile].tmem_index;
        mRdp->loaded_texture[slot].addr = TmemLoad(mRdp->texture_tile[tile].tmem, mRdp->texture_to_load.addr, size_bytes, size_bytes, 1);
    }
    // fprintf(stderr, "GfxDpLoadBlock: line_size = 0x%x; orig = 0x%x; bpp=%d; lrs=%d\n", size_bytes,
    // orig_size_bytes,
    //         mRdp->texture_to_load.siz, lrs);

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Identifier.GetPath())
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

void Interpreter::GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt) {
    if (PortTexLog()) fprintf(stderr, "[tex] loadtile t%u %u %u %u %u\n", tile, uls, ult, lrs, lrt);
    SUPPORT_CHECK(tile == G_TX_LOADTILE);

    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }

    uint32_t offset_x = uls >> G_TEXTURE_IMAGE_FRAC;
    uint32_t offset_y = ult >> G_TEXTURE_IMAGE_FRAC;
    uint32_t tile_width = ((lrs - uls) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t tile_height = ((lrt - ult) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t full_image_width = mRdp->texture_to_load.width;

    uint32_t offset_x_in_bytes = offset_x << word_size_shift;
    uint32_t tile_line_size_bytes = tile_width << word_size_shift;
    uint32_t full_image_line_size_bytes = full_image_width << word_size_shift;

    uint32_t orig_size_bytes = tile_line_size_bytes * tile_height;
    uint32_t size_bytes = orig_size_bytes;
    uint32_t start_offset_bytes = full_image_line_size_bytes * offset_y + offset_x_in_bytes;

    float h_byte_scale = mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
    float v_pixel_scale = mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;

    if (h_byte_scale != 1 || v_pixel_scale != 1) {
        start_offset_bytes = h_byte_scale * (v_pixel_scale * offset_y * full_image_line_size_bytes + offset_x_in_bytes);
        size_bytes *= h_byte_scale * v_pixel_scale;
        full_image_line_size_bytes *= h_byte_scale;
        tile_line_size_bytes *= h_byte_scale;
    }

    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = full_image_line_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = tile_line_size_bytes;

    //    assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr + start_offset_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_addr = mRdp->texture_to_load.addr + start_offset_bytes;
    if (mRdp->texture_to_load.raw_tex_metadata.resource == nullptr && tile_line_size_bytes > 0) {
        // PORT: snapshot RAM textures (TMEM semantics); rows are compacted to the tile's line size
        int slot = mRdp->texture_tile[tile].tmem_index;
        uint32_t rows = size_bytes / tile_line_size_bytes;
        mRdp->loaded_texture[slot].addr = TmemLoad(mRdp->texture_tile[tile].tmem, mRdp->texture_to_load.addr + start_offset_bytes,
                                                   tile_line_size_bytes, full_image_line_size_bytes, rows);
        mRdp->loaded_texture[slot].full_image_line_size_bytes = tile_line_size_bytes;
    }

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Identifier.GetPath())
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;

    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

/*static uint8_t color_comb_component(uint32_t v) {
    switch (v) {
        case G_CCMUX_TEXEL0:
            return CC_TEXEL0;
        case G_CCMUX_TEXEL1:
            return CC_TEXEL1;
        case G_CCMUX_PRIMITIVE:
            return CC_PRIM;
        case G_CCMUX_SHADE:
            return CC_SHADE;
        case G_CCMUX_ENVIRONMENT:
            return CC_ENV;
        case G_CCMUX_TEXEL0_ALPHA:
            return CC_TEXEL0A;
        case G_CCMUX_LOD_FRACTION:
            return CC_LOD;
        default:
            return CC_0;
    }
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return color_comb_component(a) |
           (color_comb_component(b) << 3) |
           (color_comb_component(c) << 6) |
           (color_comb_component(d) << 9);
}

static void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha) {
    mRdp->combine_mode = rgb | (alpha << 12);
}*/

void Interpreter::GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2) {
    mRdp->combine_mode = rgb | (alpha << 16) | ((uint64_t)rgb_cyc2 << 28) | ((uint64_t)alpha_cyc2 << 44);
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 0xf) | ((b & 0xf) << 4) | ((c & 0x1f) << 8) | ((d & 7) << 13);
}

static inline uint32_t alpha_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 7) | ((b & 7) << 3) | ((c & 7) << 6) | ((d & 7) << 9);
}

// Preserve RDRAM pixels when a colour image becomes a GPU target. The castle's defeated-trainer
// mark blends into a loaded portrait, rather than clearing and redrawing the entire image.
void Interpreter::RttInitialize(int id, uintptr_t address, uint32_t width, uint32_t height) {
    static const bool trace = getenv("PORT_TRACE_RTT_INIT") != nullptr;
    const auto started = trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto* source = reinterpret_cast<const uint8_t*>(address);
    std::vector<uint8_t> pixels((size_t)width * height * 4);
    for (size_t i = 0; i < pixels.size() / 4; ++i) {
        const uint16_t value = (uint16_t(source[i * 2]) << 8) | source[i * 2 + 1];
        pixels[i * 4] = SCALE_5_8(value >> 11);
        pixels[i * 4 + 1] = SCALE_5_8((value >> 6) & 31);
        pixels[i * 4 + 2] = SCALE_5_8((value >> 1) & 31);
        pixels[i * 4 + 3] = (value & 1) ? 255 : 0;
    }
    mRapi->InitializeFramebuffer(id, width, height, pixels.data());
    if (trace) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        fprintf(stderr, "[rtt-init] %ux%u %.3f ms\n", width, height, ms);
    }
    static const bool check = getenv("PORT_CHECK_RTT_INIT") != nullptr;
    if (check) {
        SetFrameBuffer(id, 1.0f);
        std::vector<uint16_t> copied((size_t)width * height);
        mRapi->ReadFramebufferToCPU(id, width, height, copied.data());
        size_t mismatches = 0;
        for (size_t i = 0; i < copied.size(); ++i) {
            const uint16_t expected = (uint16_t(source[i * 2]) << 8) | source[i * 2 + 1];
            if (copied[i] != expected && mismatches++ == 0) {
                fprintf(stderr, "[rtt-init-check] first difference at %zu: expected %04x, got %04x\n",
                        i, expected, copied[i]);
            }
        }
        fprintf(stderr, "[rtt-init-check] %ux%u: %zu mismatches of %zu pixels\n",
                width, height, mismatches, copied.size());
    }
}

// Sign-extend a 9-bit value (used for G_SETCONVERT K0..K5).
static inline int16_t sign_extend_9(uint32_t v) {
    return (int16_t)((v & 0x100) ? (int32_t)(v | 0xFFFFFE00u) : (int32_t)v);
}

void Interpreter::GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->grayscale_color.r = r;
    mRdp->grayscale_color.g = g;
    mRdp->grayscale_color.b = b;
    mRdp->grayscale_color.a = a;
}

void Interpreter::GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->env_color.r = r;
    mRdp->env_color.g = g;
    mRdp->env_color.b = b;
    mRdp->env_color.a = a;
}

void Interpreter::GfxDpSetPrimColor(uint8_t m, uint8_t l, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->prim_lod_fraction = l;
    mRdp->prim_color.r = r;
    mRdp->prim_color.g = g;
    mRdp->prim_color.b = b;
    mRdp->prim_color.a = a;
}

void Interpreter::GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->fog_color.r = r;
    mRdp->fog_color.g = g;
    mRdp->fog_color.b = b;
    mRdp->fog_color.a = a;
}

void Interpreter::GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->blend_color.r = r;
    mRdp->blend_color.g = g;
    mRdp->blend_color.b = b;
    mRdp->blend_color.a = a;
}

void Interpreter::GfxDpSetFillColor(uint32_t packed_color) {
    uint16_t col16 = (uint16_t)packed_color;
    uint32_t r = col16 >> 11;
    uint32_t g = (col16 >> 6) & 0x1f;
    uint32_t b = (col16 >> 1) & 0x1f;
    uint32_t a = col16 & 1;
    mRdp->fill_color.r = SCALE_5_8(r);
    mRdp->fill_color.g = SCALE_5_8(g);
    mRdp->fill_color.b = SCALE_5_8(b);
    mRdp->fill_color.a = a * 255;
}

void Interpreter::GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    uint32_t saved_other_mode_h = mRdp->other_mode_h;
    uint32_t cycle_type = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = (mRdp->other_mode_h & ~(3U << G_MDSFT_TEXTFILT)) | G_TF_POINT;
    }

    // U10.2 coordinates
    float ulxf = ulx;
    float ulyf = uly;
    float lrxf = lrx;
    float lryf = lry;

    ulxf = ulxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    ulyf = -(ulyf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;
    lrxf = lrxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    lryf = -(lryf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;

    if (!sWideRects) { // PORT: stretched 2D effects keep the full width (sWideRects)
        ulxf = AdjXForAspectRatio(ulxf);
        lrxf = AdjXForAspectRatio(lrxf);
    }

    struct LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    struct LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    struct LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    struct LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];

    ul->x = ulxf;
    ul->y = ulyf;
    ul->z = -1.0f;
    ul->w = 1.0f;

    ll->x = ulxf;
    ll->y = lryf;
    ll->z = -1.0f;
    ll->w = 1.0f;

    lr->x = lrxf;
    lr->y = lryf;
    lr->z = -1.0f;
    lr->w = 1.0f;

    ur->x = lrxf;
    ur->y = ulyf;
    ur->z = -1.0f;
    ur->w = 1.0f;

    // The coordinates for texture rectangle shall bypass the viewport setting
    struct XYWidthHeight default_viewport;
    if (!mFbActive) {
        default_viewport = { 0, (int16_t)mNativeDimensions.height, mNativeDimensions.width, mNativeDimensions.height };
    } else {
        default_viewport = { 0, (int16_t)mActiveFrameBuffer->second.orig_height, mActiveFrameBuffer->second.orig_width,
                             mActiveFrameBuffer->second.orig_height };
    }

    struct XYWidthHeight viewport_saved = mRdp->viewport;
    uint32_t geometry_mode_saved = mRsp->geometry_mode;

    AdjustVIewportOrScissor(&default_viewport);

    mRdp->viewport = default_viewport;
    mRdp->viewport_or_scissor_changed = true;
    mRsp->geometry_mode = 0;

    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 3, true);
    GfxSpTri1(MAX_VERTICES + 1, MAX_VERTICES + 2, MAX_VERTICES + 3, true);

    mRsp->geometry_mode = geometry_mode_saved;
    mRdp->viewport = viewport_saved;
    mRdp->viewport_or_scissor_changed = true;

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = saved_other_mode_h;
    }
}

// PORT: 2D in-between frames. An exact drawing (mReplayT 1) records each rectangle of the main picture by key (texture,
// size, texture coordinates or fill colour). Before a frame's first in-between drawing, Interp2DPrepare walks its
// display list for its rectangles and pairs them with the last exact drawing's, for the whole frame at once: same key
// at the same place first (not moving), then the most common moves (a sliding panel's background, border and text all
// move alike), so the parts of a panel stay together. A move needs a key that occurred once on both sides (a cursor,
// a panel) or at least 4 rectangles moving alike; moves of more than 48 pixels (a cut, a jump) and new rectangles are
// not blended. In-between drawings then place each rectangle between its old and new place.
static std::unordered_map<Interp2DKey, std::vector<std::pair<int32_t, int32_t>>, Interp2DKeyHash> sRectMove;
static void Interp2DPlaceImpl(float t, const Interp2DKey& key, int32_t& ulx, int32_t& uly, int32_t& lrx, int32_t& lry) {
    static const bool sOff = getenv("PORT_INTERP_2D") != nullptr && atoi(getenv("PORT_INTERP_2D")) == 0;
    if (sOff) return;
    const uint32_t n = sRectSeen[key]++;
    if (t >= 0.999f) {
        sRectRec[key].push_back({ ulx, uly });
        return;
    }
    auto it = sRectMove.find(key);
    if (it == sRectMove.end() || n >= it->second.size()) return;
    const int32_t dx = it->second[n].first, dy = it->second[n].second;
    if (dx == 0 && dy == 0) return;
    const int32_t sx = (int32_t)lroundf(-(1.0f - t) * dx), sy = (int32_t)lroundf(-(1.0f - t) * dy);
    ulx += sx;
    lrx += sx;
    uly += sy;
    lry += sy;
}

void Interpreter::GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                                        int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
    if (!mFbActive) { // PORT: 2D in-between frames (mReplayT)
        const TmemRegion* r = TmemFind(mRdp->texture_tile[tile].tmem);
        Interp2DPlaceImpl(mReplayT, { r ? (const void*)r->raw : nullptr, lrx - ulx, lry - uly,
                        ((uint32_t)(uint16_t)uls << 16) | (uint16_t)ult },
                      ulx, uly, lrx, lry);
    }
    if (PortTexLog()) fprintf(stderr, "[rect] %d,%d - %d,%d tile=%u st=%d,%d d=%d,%d flip=%d\n", ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, (int)flip);
    if (gPortDrawLog && PortRectLog()) // PORT debug: state of each textured rectangle in the logged frame
        fprintf(stderr, "[trect] %d,%d-%d,%d cc=%016llx omh=%08x oml=%08x fmt=%u siz=%u pal=%u prim=%02x%02x%02x%02x env=%02x%02x%02x%02x tex=%p\n",
                ulx >> 2, uly >> 2, lrx >> 2, lry >> 2, (unsigned long long)mRdp->combine_mode, mRdp->other_mode_h, mRdp->other_mode_l,
                mRdp->texture_tile[tile].fmt, mRdp->texture_tile[tile].siz, mRdp->texture_tile[tile].palette, mRdp->prim_color.r,
                mRdp->prim_color.g, mRdp->prim_color.b, mRdp->prim_color.a, mRdp->env_color.r, mRdp->env_color.g, mRdp->env_color.b,
                mRdp->env_color.a, (const void*)mRdp->loaded_texture[0].raw_addr);
    // PORT: a rectangle whose left edge lies right of its right edge draws nothing on the RDP. The game produces them
    // when a panel slides past the screen's left edge: the negative x wraps around in the 10.2 coordinates (e.g. the HP
    // bar at 1010..34 as the battle menu leaves), which drew a line across the whole picture (checked against ares).
    if (lrx <= ulx || lry <= uly) {
        return;
    }
    uint64_t saved_combine_mode = mRdp->combine_mode;
    if ((mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_COPY) {
        // Per RDP Command Summary Set Tile's shift s and this dsdx should be set to 4 texels
        // Divide by 4 to get 1 instead
        dsdx >>= 2;

        // Color combiner is turned off in copy mode
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_TEXEL0), alpha_comb(0, 0, 0, G_ACMUX_TEXEL0), 0, 0);

        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    // uls and ult are S10.5
    // dsdx and dtdy are S5.10
    // lrx, lry, ulx, uly are U10.2
    // lrs, lrt are S10.5
    if (flip) {
        dsdx = -dsdx;
        dtdy = -dtdy;
    }
    // PORT: a rectangle across the whole native width whose texture does not change horizontally (dsdx 0: every row
    // one colour, e.g. the battle stadiums' sky gradient) reaches past both screen edges, so it also fills the sides
    // of a widescreen picture; pictures (dsdx != 0) keep their 4:3 place. Outside the screen at 4:3.
    bool extended = false;
    bool constantBackdrop = false; // (drawn past a 2D screen's 4:3 lock like a widened pattern: sWidePattern)
    // PORT: one rectangle over the whole N64 screen in a frame with 3D is a full-screen effect (the flash when a
    // Pokémon is sent out): it is stretched over a widescreen picture like the marked effects (sWideRects)
    const bool savedWideRects = sWideRects;
    if (!flip && !mFbActive && sSeen3DThisRun && ulx <= 0 && uly <= 0 &&
        lrx >= ((int32_t)(mNativeDimensions.width - 1) * 4) && lry >= ((int32_t)(mNativeDimensions.height - 1) * 4)) {
        sWideRects = true;
        extended = true;
    }
    if (!extended && !flip && dsdx == 0 && ulx <= 0 && lrx >= ((int32_t)(mNativeDimensions.width - 1) * 4) &&
        (lry - uly) >= (int32_t)mNativeDimensions.height * 2) { // half the height or more: a backdrop, not a HUD line
        ulx = -(int32_t)mNativeDimensions.width * 4;
        lrx = 2 * (int32_t)mNativeDimensions.width * 4;
        extended = true;
        constantBackdrop = !mFbActive && !sNoWidePattern;
        sFilledWideThisRun |= constantBackdrop;
    }
    // PORT: a 2D screen's patterned backdrop (the menus' "POKéMON" wallpaper) is one rectangle over the whole native
    // picture whose texture repeats horizontally (a WRAP/MIRROR tile, several periods across). In a widescreen picture
    // it reaches both screen edges with the texture coordinates continued, so the pattern goes on into the sides at
    // its own size instead of being stretched, and it is drawn past the 2D screen's 4:3 lock (sWidePattern) while the
    // menu on top stays inside it. Pictures (clamped tiles) keep their 4:3 place, and so do patterns drawn as
    // full-width bands (the GB screens, the gallery's film screen: there other parts fill the rows between the bands).
    // PORT_NO_WIDE_PATTERN=1: off.
    bool widePattern = false;
    if (!extended && !flip && !mFbActive && !sNoWidePattern && dsdx > 0 && ulx <= 0 &&
        lrx >= ((int32_t)(mNativeDimensions.width - 1) * 4) && (lry - uly) >= (int32_t)mNativeDimensions.height * 2 &&
        AdjXForAspectRatio(1.0f) < 0.999f) {
        const auto& t = mRdp->texture_tile[tile];
        const int32_t period = (32 << t.masks) << ((t.cms & G_TX_MIRROR) ? 1 : 0); // S10.5 units
        if (!(t.cms & G_TX_CLAMP) && t.masks != 0 && (((lrx - ulx) * dsdx) >> 7) >= 2 * period) {
            const int32_t newUlx = -(int32_t)mNativeDimensions.width * 4;
            const int32_t shift = (((ulx - newUlx) * dsdx) >> 7) % period; // whole periods dropped: same pattern
            uls = (int16_t)(uls - shift);
            ulx = newUlx;
            lrx = 2 * (int32_t)mNativeDimensions.width * 4;
            extended = true;
            widePattern = true;
            sFilledWideThisRun = true;
        }
    }
    // PORT: in a widescreen picture, other rectangles are cut at the 4:3 frame's sides as the N64 screen edge cut them:
    // panels the game slides out past the edge (the battle menu after choosing a move) would otherwise stay visible in
    // the margins. A rectangle entirely within the outermost 4 pixels is dropped: what a panel leaves at the edge as it
    // slides out (a strip at x 317-320), on a TV hidden by overscan, in a widescreen picture a line in the middle of
    // the scene. Nothing changes at 4:3.
    if (!extended && !flip && !sWideRects && !mFbActive && !sLock43ThisRun && AdjXForAspectRatio(1.0f) < 0.999f) {
        const int32_t band = (int32_t)mNativeDimensions.width / 80 * 4; // 4 pixels (low resolution)
        const int32_t left = 0, right = (int32_t)mNativeDimensions.width * 4;
        if ((lrx <= left + band || ulx >= right - band) && sAnchor != 2) { // (not-cut drawing runs on past the edge)
            mRdp->combine_mode = saved_combine_mode;
            return;
        }
        if (sAnchor != 0) {
            // anchored (sAnchor): not cut at the 4:3 frame's sides
        } else if (ulx < left) {
            uls += (int16_t)(((int32_t)(left - ulx) * dsdx) >> 7);
            ulx = left;
        }
        if (lrx > right && sAnchor == 0) {
            lrx = right;
        }
    }
    // PORT: textured rectangles covering most of the picture before any 3D make this a 2D screen (s2DRectArea)
    static const bool sNo2DLock = getenv("PORT_NO_2D_LOCK") != nullptr;
    if (!sNo2DLock && !mFbActive && !sSeen3DThisRun && !sLock43ThisRun) {
        const float nw = (float)mNativeDimensions.width, nh = (float)mNativeDimensions.height;
        const float x0 = std::max(0.0f, ulx / 4.0f), x1 = std::min(nw, lrx / 4.0f);
        const float y0 = std::max(0.0f, uly / 4.0f), y1 = std::min(nh, lry / 4.0f);
        if (x1 > x0 && y1 > y0) {
            s2DRectArea += (x1 - x0) * (y1 - y0);
        }
        if (s2DRectArea >= 0.9f * nw * nh) {
            sLock43ThisRun = true;
            mRdp->viewport_or_scissor_changed = true;
        }
    }
    int16_t width = !flip ? lrx - ulx : lry - uly;
    int16_t height = !flip ? lry - uly : lrx - ulx;
    float lrs = ((uls << 7) + dsdx * width) >> 7;
    float lrt = ((ult << 7) + dtdy * height) >> 7;

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls;
    ul->v = ult;
    lr->u = lrs;
    lr->v = lrt;
    if (!flip) {
        ll->u = uls;
        ll->v = lrt;
        ur->u = lrs;
        ur->v = ult;
    } else {
        ll->u = lrs;
        ll->v = ult;
        ur->u = uls;
        ur->v = lrt;
    }

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    // PORT: a rectangle that shows its texture once (all its texture coordinates within the tile) is drawn with the
    // tile clamped. The N64 samples texel centres only (1:1), so a wrapping tile never reaches past the texture's
    // edge there; filtered and scaled to another size (fullscreen), the edge pixels did and blended in the other side
    // of the texture: dark lines around the pieces of the menus' rounded boxes (16x16 corners on wrapping 32-texel
    // tiles). Tiled patterns (coordinates beyond the tile) keep wrapping. PORT_NO_RECT_CLAMP=1: off.
    static const bool sNoRectClamp = getenv("PORT_NO_RECT_CLAMP") != nullptr;
    auto& rectTile = mRdp->texture_tile[tile];
    const uint8_t savedCms = rectTile.cms, savedCmt = rectTile.cmt;
    if (!sNoRectClamp) {
        // S10.5 texture coordinates against the tile's origin (10.2) and size
        auto within = [](float a, float b, float tileUl, float tileLr) {
            const float origin = tileUl * 8.0f, size = (float)GetTileSizeFromCoordinates(tileUl, tileLr) * 32.0f;
            return std::min(a, b) >= origin && std::max(a, b) <= origin + size;
        };
        if (!(rectTile.cms & G_TX_CLAMP) && within(ul->u, lr->u, rectTile.uls, rectTile.lrs)) {
            rectTile.cms |= G_TX_CLAMP;
        }
        if (!(rectTile.cmt & G_TX_CLAMP) && within(ul->v, lr->v, rectTile.ult, rectTile.lrt)) {
            rectTile.cmt |= G_TX_CLAMP;
        }
    }

    sWidePattern = widePattern || constantBackdrop;
    GfxDrawRectangle(ulx, uly, lrx, lry);
    sWidePattern = false;
    rectTile.cms = savedCms;
    rectTile.cmt = savedCmt;
    sWideRects = savedWideRects;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls,
                                      int16_t ult, int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt) {

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls * 32;
    ul->v = ult * 32;
    lr->u = lrs * 32;
    lr->v = lrt * 32;
    ll->u = uls * 32;
    ll->v = lrt * 32;
    ur->u = lrs * 32;
    ur->v = ult * 32;

    // ensure we have the correct texture size, format and starting position
    mRdp->texture_tile[tile].siz = G_IM_SIZ_8b;
    mRdp->texture_tile[tile].fmt = G_IM_FMT_RGBA;
    mRdp->texture_tile[tile].cms = 0;
    mRdp->texture_tile[tile].cmt = 0;
    mRdp->texture_tile[tile].shifts = 0;
    mRdp->texture_tile[tile].shiftt = 0;
    mRdp->texture_tile[tile].uls = 0 * 4;
    mRdp->texture_tile[tile].ult = 0 * 4;
    mRdp->texture_tile[tile].lrs = w * 4;
    mRdp->texture_tile[tile].lrt = h * 4;
    mRdp->texture_tile[tile].line_size_bytes = w << (mRdp->texture_tile[tile].siz >> 1);

    auto& loadtex = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    loadtex.full_image_line_size_bytes = loadtex.line_size_bytes = mRdp->texture_tile[tile].line_size_bytes;
    loadtex.size_bytes = loadtex.orig_size_bytes = loadtex.line_size_bytes * h;

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
}

void Interpreter::GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    if (!mFbActive && mRdp->color_image_address != mRdp->z_buf_address) { // PORT: 2D in-between frames (mReplayT)
        Interp2DPlaceImpl(mReplayT, { nullptr, lrx - ulx, lry - uly,
                        (uint32_t)mRdp->fill_color.r << 24 | mRdp->fill_color.g << 16 | mRdp->fill_color.b << 8 |
                            mRdp->fill_color.a },
                      ulx, uly, lrx, lry);
    }
    if (gPortDrawLog && PortRectLog())
        fprintf(stderr, "[frect] %d,%d-%d,%d fill=%08x omh=%08x cc=%016llx prim=%02x%02x%02x%02x env=%02x%02x%02x%02x\n", ulx, uly, lrx, lry,
                mRdp->fill_color.r << 24 | mRdp->fill_color.g << 16 | mRdp->fill_color.b << 8 | mRdp->fill_color.a, mRdp->other_mode_h,
                (unsigned long long)mRdp->combine_mode, mRdp->prim_color.r, mRdp->prim_color.g, mRdp->prim_color.b, mRdp->prim_color.a,
                mRdp->env_color.r, mRdp->env_color.g, mRdp->env_color.b, mRdp->env_color.a);
    if (mRdp->color_image_address == mRdp->z_buf_address) {
        // Fullscreen Z clears are redundant — already done by glClear at frame start.
        bool isFullScreen = (ulx <= 0 && uly <= 0 && lrx >= (int32_t)(mNativeDimensions.width - 1) * 4 &&
                             lry >= (int32_t)(mNativeDimensions.height - 1) * 4);
        if (isFullScreen) {
            return;
        }

        // Partial depth clear (e.g. HUD model regions): clear the actual depth buffer
        // via a scissored depth clear instead of drawing a colored rect to the color buffer.
        Flush();

        // Convert U10.2 coords to pixel coords and add +1 pixel for fill mode
        int32_t expanded_lrx = lrx + (1 << 2);
        int32_t expanded_lry = lry + (1 << 2);
        float x = ulx / 4.0f;
        float y = expanded_lry / 4.0f;
        float w = (expanded_lrx - ulx) / 4.0f;
        float h = (expanded_lry - uly) / 4.0f;
        float halfNativeWidth = (float)HALF_SCREEN_WIDTH(mActiveFrameBuffer);
        x = halfNativeWidth + AdjXForAspectRatio(x - halfNativeWidth);
        w = AdjXForAspectRatio(w);

        struct XYWidthHeight area;
        area.x = (int16_t)x;
        area.y = (int16_t)y;
        area.width = (uint32_t)w;
        area.height = (uint32_t)h;
        AdjustVIewportOrScissor(&area);

        mRapi->ClearDepthRegion(area.x, area.y, area.width, area.height);
        return;
    }
    uint32_t mode = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    // PORT: a fill of a whole render-to-texture image (e.g. the VS screen's Pokémon pictures, filled with transparent
    // 0x0000 before the model is drawn) must write the fill colour's alpha bit: opaque draws output alpha 1, which
    // would give the pictures a solid black background where the N64 leaves them see-through.
    static const bool sNoFillAlpha = getenv("PORT_NO_RTT_FILL_ALPHA") != nullptr; // debug
    if (mode == G_CYC_FILL && sInOffscreen && !sNoFillAlpha && mActiveFrameBuffer != mFrameBuffers.end() && ulx <= 0 && uly <= 0 &&
        (uint32_t)(lrx >> 2) + 1 >= mActiveFrameBuffer->second.orig_width &&
        (uint32_t)(lry >> 2) + 1 >= mActiveFrameBuffer->second.orig_height) {
        Flush();
        if (PortTexLog()) fprintf(stderr, "[rtt] fill clear %p %ux%u color %02x%02x%02x%02x\n", (void*)sCurOffscreenAddr,
                                           mActiveFrameBuffer->second.orig_width, mActiveFrameBuffer->second.orig_height,
                                           mRdp->fill_color.r, mRdp->fill_color.g, mRdp->fill_color.b, mRdp->fill_color.a);
        if (mRapi->ClearFramebufferColor(mRdp->fill_color.r / 255.0f, mRdp->fill_color.g / 255.0f,
                                         mRdp->fill_color.b / 255.0f, mRdp->fill_color.a / 255.0f)) {
            return;
        }
    }

    // Expand fullscreen fill rects to cover widescreen viewports.
    // Without this, screen clears and fades only cover the native 4:3 area.
    if (ulx == 0 && uly == 0) {
        bool isFullScreen = (lrx == ((int32_t)(mNativeDimensions.width - 1) * 4) &&
                             lry == ((int32_t)(mNativeDimensions.height - 1) * 4));
        if (isFullScreen) {
            // PORT: a margin of the native width on every side (-256..512 pixels at 320x240, as before). The fixed
            // -1024..2048 stopped at x 512 on 640x480 screens: the Trade Machine's clear left its right fifth, so
            // pieces of earlier panels stayed below the machine.
            const int32_t w = (int32_t)mNativeDimensions.width;
            ulx = -w * 16 / 5;
            uly = -w * 16 / 5;
            lrx = w * 32 / 5;
            lry = w * 32 / 5;
        }
    }

    if (mode == G_CYC_COPY || mode == G_CYC_FILL) {
        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    for (int i = MAX_VERTICES; i < MAX_VERTICES + 4; i++) {
        LoadedVertex* v = &mRsp->loaded_vertices[i];
        v->color = mRdp->fill_color;
    }

    uint64_t saved_combine_mode = mRdp->combine_mode;

    if (mode == G_CYC_FILL) {
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_SHADE), alpha_comb(0, 0, 0, G_ACMUX_SHADE), 0, 0);
    }

    GfxDrawRectangle(ulx, uly, lrx, lry);
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpSetZImage(void* zBufAddr) {
    mRdp->z_buf_address = zBufAddr;
}

// PORT PATCH: render-to-texture. Stadium draws some UI text into small RGBA16 buffers in RDRAM (G_SETCIMG to a buffer that
// is not one of the 320/640 pixel wide frame buffers) and samples them as textures later. Such a color image gets an
// offscreen framebuffer keyed by its address; G_SETTIMG with that address binds the framebuffer texture.

// PORT: the N64 RDP writes rendered pixels to RDRAM, and the game can load them into TMEM from there (the arena's
// near floor samples TMEM that the portrait images were loaded into). Render targets live on the GPU, so when the game
// stops drawing into one, its pixels are copied back into the RDRAM buffer as big-endian RGBA5551, all of them; the
// first 16 bytes then become the target's stamp (RttStamp), so the target itself stays bound for normal texture use.
void Interpreter::RttReadback() {
    Flush();
    static const bool off = getenv("PORT_NO_RTT_READBACK") != nullptr;
    auto it = sOffscreenFbs.find(sCurOffscreenAddr);
    if (off || mReplayPass || sCurOffscreenAddr == 0 || it == sOffscreenFbs.end() || !RttStampValid(it->first, it->second)) {
        return;
    }
    uint32_t w = it->second.w, h = it->second.h;
    static std::vector<uint16_t> buf;
    buf.assign((size_t)w * h, 0);
    mRapi->ReadFramebufferToCPU(it->second.fb, w, h, buf.data());
    if (PortTexLog()) {
        size_t nz = 0, opaque = 0;
        for (uint16_t v : buf) {
            nz += (v & 0xFFFE) != 0;
            opaque += v & 1;
        }
        fprintf(stderr, "[rtt] readback fb %d %ux%u: %zu coloured, %zu opaque pixels\n", it->second.fb, w, h, nz, opaque);
    }
    uint8_t* dst = (uint8_t*)it->first;
    for (size_t i = 0; i < buf.size(); i++) {
        dst[i * 2] = (uint8_t)(buf[i] >> 8);
        dst[i * 2 + 1] = (uint8_t)buf[i];
    }
    memcpy(it->second.stamp, dst, sizeof(it->second.stamp)); // see RttStamp
}

// PORT: a screen-sized picture the game never shows (osViSwapBuffer) is drawn to be used as a texture (Gallery
// enlarge renders the photo at 640x480, then draws it; the Bonus stickers screen builds its sheet of thumbnails in a
// 640x480 image): render-to-texture too. Before two frame buffers have been shown, every screen-sized image counts as
// a frame buffer. Also used by the state passes for frames that are not drawn (TrackSegmentsOnly, FrameRepaintsScreen).
static bool IsUnshownScreenImage(uint32_t pixelWidth, const void* address) {
    static const bool sNoFbCheck = getenv("PORT_NO_VIFB_CHECK") != nullptr; // debug
    return (pixelWidth == 320 || pixelWidth == 640) && address != nullptr && !sNoFbCheck && !PortNoRtt() &&
           Port_ViFrameBufferState(address) == 0;
}

void Interpreter::GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address) {
    const bool preserve = sPreserveNextColorImage;
    sPreserveNextColorImage = false;
    bool mainFbSelected = false;
    if (PortTexLog() || gPortDrawLog) fprintf(stderr, "[cimg] fmt=%u siz=%u w=%u addr=%p\n", format, size, width, address);
    mRdp->color_image_address = address;
    uint32_t pixelWidth = width + 1; // the field holds width - 1
    bool offscreen = pixelWidth != 320 && pixelWidth != 640 && address != nullptr && !PortNoRtt();
    if (!offscreen && IsUnshownScreenImage(pixelWidth, address)) {
        offscreen = true;
    }
    if (offscreen) {
        Flush();
        // PORT: a screen-sized off-screen picture (Gallery enlarge: 640x480) is drawn in the 640x480 space; with the
        // native size of the screen drawn before (320x240) its scene came out 2x zoomed. Smaller targets keep the
        // current space, as before. The main colour image sets it again when the game returns to it.
        if (pixelWidth > 400 && !PortFixedNativeW()) {
            mNativeDimensions.width = 640;
            mNativeDimensions.height = 480;
        }
        auto it = sOffscreenFbs.find((uintptr_t)address);
        if (it != sOffscreenFbs.end()) {
            if (sInOffscreen) {
                RttReadback();
            }
            SetFrameBuffer(it->second.fb, 1.0f);
            mActiveFrameBuffer = mFrameBuffers.find(it->second.fb);
            mFbActive = true;
            sInOffscreen = true;
            mRdp->viewport_or_scissor_changed = true; // see GfxDpSetScissor
            mRenderingState.viewport = {};
            mRenderingState.scissor = {};
            sCurOffscreenAddr = (uintptr_t)address;
            sPendingOffscreenAddr = 0; // PORT: a colour image set just before (no scissor yet) is not drawn into
        } else {
            if (sInOffscreen) {
                RttReadback();
                sCurOffscreenAddr = 0;
            }
            sPendingOffscreenAddr = (uintptr_t)address; // created at the scissor that follows, which carries the height
            sPendingOffscreenWidth = pixelWidth;
            sPendingOffscreenPreserve = preserve;
        }
        return;
    }
    sPendingOffscreenAddr = 0;
    if (sInOffscreen) {
        RttReadback();
        sCurOffscreenAddr = 0;
        mFbActive = false;
        mActiveFrameBuffer = mFrameBuffers.end();
        mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
        mRdp->viewport_or_scissor_changed = true;
        mRenderingState.viewport = {};
        mRenderingState.scissor = {};
        sInOffscreen = false;
    }
    if (mRendersToFb && address != nullptr && address != mRdp->z_buf_address) {
        auto mf = sMainFbs.find((uintptr_t)address);
        int fb;
        if (mf == sMainFbs.end()) {
            if (sMainFbs.empty()) {
                fb = mGameFb; // the first frame buffer uses the existing target
            } else {
                fb = mRapi->CreateFramebuffer();
                if (!ViewportMatchesRendererResolution()) {
                    mRapi->UpdateFramebufferParameters(fb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel,
                                                       true, true, true, true);
                } else {
                    // PORT: shown as a texture (GUI) unless MSAA resolves it into the window: Y-inverted like the
                    // other case (OpenGL showed the game upside down when the window matched the render size)
                    mRapi->UpdateFramebufferParameters(fb, mGfxCurrentWindowDimensions.width,
                                                       mGfxCurrentWindowDimensions.height, mMsaaLevel, mMsaaLevel <= 1,
                                                       true, true, true);
                }
            }
            sMainFbs[(uintptr_t)address] = fb;
        } else {
            fb = mf->second;
        }
        if (!mReplayPass) {
            sMainFbDrawStamp[(uintptr_t)address] = ++sDrawStamp;
        }
        mainFbSelected = true;
        if (fb != mGameFb) {
            Flush();
            mGameFb = fb;
            mRapi->StartDrawToFramebuffer(mGameFb, (float)mCurDimensions.height / mNativeDimensions.height);
            mRapi->ClearFramebuffer(false, true);
            mRdp->viewport_or_scissor_changed = true;
            mRenderingState.viewport = {};
            mRenderingState.scissor = {};
        }
        // PORT: frame interpolation of a frame that draws over the previous picture (ReplayBegin): the first drawing
        // copies this buffer's picture away, each repeat copies it back before drawing
        if (sReplayRestore && sSnapDoneThisRun.insert((uintptr_t)address).second) {
            // (the copy has the game buffer's size, MSAA level and orientation: see the resize in StartFrame)
            const bool window = ViewportMatchesRendererResolution();
            const uint32_t w = window ? mGfxCurrentWindowDimensions.width : mCurDimensions.width;
            const uint32_t h = window ? mGfxCurrentWindowDimensions.height : mCurDimensions.height;
            auto& entry = sSnapFbs[(uintptr_t)address];
            if (entry.first == 0) {
                entry.first = mRapi->CreateFramebuffer();
            }
            if (entry.second != mMsaaLevel) {
                entry.second = mMsaaLevel;
                mRapi->UpdateFramebufferParameters(entry.first, w, h, mMsaaLevel, window ? mMsaaLevel <= 1 : true, true,
                                                   true, true);
            }
            const int snap = entry.first;
            Flush();
            if (!mReplayPass) {
                mRapi->CopyFramebufferInStream(fb, snap, fb, (int)w, (int)h);
            } else {
                mRapi->CopyFramebufferInStream(fb, fb, snap, (int)w, (int)h);
            }
        }
    }
    // Stadium switches between 320x240 and 640x480 (hi-res) color images; the native coordinate space of
    // texture rectangles, fills and scissors follows the framebuffer width.
    static const bool fixedNative = getenv("PORT_NATIVE_W") != nullptr;
    if (!fixedNative && !mFbActive && address != mRdp->z_buf_address) {
        mNativeDimensions.width = pixelWidth > 400 ? 640 : 320;
        mNativeDimensions.height = pixelWidth > 400 ? 480 : 240;
    }
    // PORT: with a window wider than 4:3 the game's frame buffers have margins the N64 picture does not cover. 3D scenes
    // draw into them, 2D screens (title, menus) do not, so each of the two or three rotating frame buffers kept
    // whatever an earlier scene left there and the margins flickered as the buffers took turns. The first frame
    // buffer the game selects in a frame gets its margins cleared to black; scenes that reach into them redraw them.
    if (mainFbSelected && !sMarginsClearedThisRun) {
        sMarginsClearedThisRun = true;
        static const bool sNoMarginClear = getenv("PORT_NO_MARGIN_CLEAR") != nullptr; // debug: old behaviour
        float aspect = mCurDimensions.height > 0 ? (float)mCurDimensions.width / mCurDimensions.height : 0.0f;
        if (!sNoMarginClear && aspect > 4.0f / 3.0f * 1.01f) {
            Flush();
            const auto savedFill = mRdp->fill_color;
            const uint32_t savedOmh = mRdp->other_mode_h, savedOml = mRdp->other_mode_l;
            const uint64_t savedCombine = mRdp->combine_mode;
            const auto savedScissor = mRdp->scissor;
            mRdp->fill_color = { 0, 0, 0, 255 };
            mRdp->other_mode_h = (savedOmh & ~(3U << G_MDSFT_CYCLETYPE)) | G_CYC_FILL;
            mRdp->other_mode_l = 0; // no depth, no blending
            mRdp->scissor = { 0, 0, (uint32_t)mCurDimensions.width, (uint32_t)mCurDimensions.height };
            mRdp->viewport_or_scissor_changed = true;
            const int32_t right = (int32_t)mNativeDimensions.width * 4, reach = 4 * 4096; // not "far": windows.h macro
            // also the picture's outermost 4 (low-resolution) pixels: the menus' scissor stops 2 pixels short of the
            // right edge, so those columns kept each buffer's old picture and flickered next to the black margin (on
            // the N64 they stay as cleared once); whatever the game draws there comes after this clear. Only on
            // frames that repaint the whole screen: a screen that draws its background once and then only what
            // changes (the rental list) kept a black band at both sides, the picture 8 pixels narrower in 16:9.
            const int32_t band = sBandClearThisRun ? (int32_t)mNativeDimensions.width / 80 * 4 : 0;
            GfxDpFillRectangle(-reach, -reach, band - 4, reach);            // left of the picture (fill mode adds one pixel)
            GfxDpFillRectangle(right - band, -reach, right + reach, reach); // right of it
            Flush();
            mRdp->fill_color = savedFill;
            mRdp->other_mode_h = savedOmh;
            mRdp->other_mode_l = savedOml;
            mRdp->combine_mode = savedCombine;
            mRdp->scissor = savedScissor;
            mRdp->viewport_or_scissor_changed = true;
            mRenderingState.scissor = {};
        }
    }
}

void Interpreter::GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode) {
    uint64_t mask = (((uint64_t)1 << num_bits) - 1) << shift;
    uint64_t om = mRdp->other_mode_l | ((uint64_t)mRdp->other_mode_h << 32);
    om = (om & ~mask) | mode;
    mRdp->other_mode_l = (uint32_t)om;
    mRdp->other_mode_h = (uint32_t)(om >> 32);
}

void Interpreter::GfxDpSetOtherMode(uint32_t h, uint32_t l) {
    mRdp->other_mode_h = h;
    mRdp->other_mode_l = l;
}

void Interpreter::Gfxs2dexBgCopy(F3DuObjBg* bg) {
    /*
    bg->b.imageX = 0;
    bg->b.imageW = width * 4;
    bg->b.frameX = frameX * 4;
    bg->b.imageY = 0;
    bg->b.imageH = height * 4;
    bg->b.frameY = frameY * 4;
    bg->b.imagePtr = source;
    bg->b.imageLoad = G_BGLT_LOADTILE;
    bg->b.imageFmt = fmt;
    bg->b.imageSiz = siz;
    bg->b.imagePal = 0;
    bg->b.imageFlip = 0;
    */

    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex =
            std::static_pointer_cast<Fast::Texture>(mResourceManager->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    s16 dsdx = 4 << 10;
    s16 uls = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (bg->b.imageW - bg->b.imageX) << 3;
    }

    SUPPORT_CHECK(bg->b.imageSiz == G_IM_SIZ_16b);
    GfxDpSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, nullptr, texFlags, rawTexMetadata, (void*)data);
    GfxDpSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, G_IM_SIZ_16b, bg->b.imageW >> 4, 0, G_TX_RENDERTILE, bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);
    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameX + bg->b.imageW - 4, bg->b.frameY + bg->b.imageH - 4,
                          G_TX_RENDERTILE, uls, bg->b.imageY << 3, dsdx, 1 << 10, false);
}

void Interpreter::Gfxs2dexBg1cyc(F3DuObjBg* bg) {
    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex =
            std::static_pointer_cast<Fast::Texture>(mResourceManager->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    // TODO: Implement bg scaling correctly
    s16 uls = bg->b.imageX >> 2;
    s16 lrs = bg->b.imageW >> 2;

    s16 dsdxRect = 1 << 10;
    s16 ulsRect = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdxRect = -dsdxRect;
        ulsRect = (bg->b.imageW - bg->b.imageX) << 3;
    }

    GfxDpSetTextureImage(bg->b.imageFmt, bg->b.imageSiz, bg->b.imageW >> 2, nullptr, texFlags, rawTexMetadata,
                         (void*)data);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, (((lrs - uls) * bg->b.imageSiz) + 7) >> 3, 0, G_TX_RENDERTILE,
                 bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);

    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameW, bg->b.frameH, G_TX_RENDERTILE, ulsRect,
                          bg->b.imageY << 3, dsdxRect, 1 << 10, false);
}

void Interpreter::Gfxs2dexRecyCopy(F3DuObjSprite* spr) {
    s16 dsdx = 4 << 10;
    [[maybe_unused]] s16 uls = spr->s.objX << 3;
    // Flip flag only flips horizontally
    if (spr->s.imageFlags == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (spr->s.imageW - spr->s.objX) << 3;
    }

    int realX = spr->s.objX >> 2;
    int realY = spr->s.objY >> 2;
    int realW = (((spr->s.imageW)) >> 5);
    int realH = (((spr->s.imageH)) >> 5);
    float realSW = spr->s.scaleW / 1024.0f;
    float realSH = spr->s.scaleH / 1024.0f;

    int testX = (realX + (realW / realSW));
    int testY = (realY + (realH / realSH));

    GfxDpTextureRectangle(realX << 2, realY << 2, testX << 2, testY << 2, G_TX_RENDERTILE,
                          (s32)mRdp->texture_tile[0].uls << 3, (s32)mRdp->texture_tile[0].ult << 3,
                          (float)(1 << 10) * realSW, (float)(1 << 10) * realSH, false);
}

// PORT PATCH: N64-style 32-bit addresses (segment-relative 0x0Nxxxxxx, fragment link addresses 0x8Nxxxxxx) can
// appear in display lists built by game code. Host pointers are always >= 4 GB on 64-bit macOS, so anything
// below that is resolved by the port's callback.
void* (*gPortAddressResolver)(uintptr_t) = nullptr;

void* Interpreter::SegAddr(uintptr_t w1) {
    if ((w1 >> 32) == 0xFFFFFFFFull) {
        w1 &= 0xFFFFFFFFull; // PORT: sign-extended 32-bit N64 address
    }
    if (gPortAddressResolver != nullptr && w1 != 0 && w1 < 0x100000000ull) {
        // A segment base set by this frame's display lists (gSPSegment) wins, like on the RSP; the port's resolver
        // (the game's own CPU-side segment table, fragment link addresses, ...) handles everything else.
        if ((w1 >> 28) == 0 && mSegmentPointers[(w1 >> 24) & 0xF] != 0) {
            return (void*)(mSegmentPointers[(w1 >> 24) & 0xF] + (w1 & 0x00FFFFFF));
        }
        return gPortAddressResolver(w1);
    }
    // Segmented?
    if (w1 & 1) {
        uint32_t segNum = (uint32_t)(w1 >> 24);

        uint32_t offset = w1 & 0x00FFFFFE;

        if (mSegmentPointers[segNum] != 0) {
            return (void*)(mSegmentPointers[segNum] + offset);
        } else {
            return (void*)w1;
        }
    } else {
        return (void*)w1;
    }
}

#define C0(pos, width) ((cmd->words.w0 >> (pos)) & ((1U << width) - 1))
#define C1(pos, width) ((cmd->words.w1 >> (pos)) & ((1U << width) - 1))

void GfxExecStack::start(F3DGfx* dlist) {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
    cmd_stack.push(dlist);
    disp_stack.clear();
}

void GfxExecStack::stop() {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
}

F3DGfx*& GfxExecStack::currCmd() {
    return cmd_stack.top();
}

void GfxExecStack::openDisp(const char* file, int line) {
    disp_stack.push_back({ file, line });
}
void GfxExecStack::closeDisp() {
    disp_stack.pop_back();
}
const std::vector<GfxExecStack::CodeDisp>& GfxExecStack::getDisp() const {
    return disp_stack;
}

void GfxExecStack::branch(F3DGfx* caller) {
    F3DGfx* old = cmd_stack.top();
    cmd_stack.pop();
    cmd_stack.push(nullptr);
    cmd_stack.push(old);

    gfx_path.push_back(caller);
}

void GfxExecStack::call(F3DGfx* caller, F3DGfx* callee) {
    cmd_stack.push(callee);
    gfx_path.push_back(caller);
}

F3DGfx* GfxExecStack::ret() {
    F3DGfx* cmd = cmd_stack.top();

    cmd_stack.pop();
    if (!gfx_path.empty()) {
        gfx_path.pop_back();
    }

    while (cmd_stack.size() > 0 && cmd_stack.top() == nullptr) {
        cmd_stack.pop();
        if (!gfx_path.empty()) {
            gfx_path.pop_back();
        }
    }
    return cmd;
}

void gfx_set_framebuffer(int fb, float noise_scale);
void gfx_reset_framebuffer();
void gfx_copy_framebuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);

// The main type of the handler function. These function will take a pointer to a pointer to a Gfx. It needs to be a
// double pointer because we sometimes need to increment and decrement the underlying pointer Returns false if the
// current opcode should be incremented after the handler ends.
typedef bool (*GfxOpcodeHandlerFunc)(F3DGfx** gfx);

bool gfx_load_ucode_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mRsp->fog_mul = 0;
    gfx->mRsp->fog_offset = 0;
    return false;
}

bool gfx_cull_dl_handler_f3dex2(F3DGfx** cmd) {
    // TODO:
    return false;
}

bool gfx_marker_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = (*cmd0);
    gfx->mMarkerOn = true;
    return false;
}

bool gfx_invalidate_tex_cache_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    const uintptr_t texAddr = (*cmd)->words.w1;

    if (texAddr == 0) {
        gfx->TextureCacheClear();
    } else {
        gfx->TextureCacheDelete((const uint8_t*)texAddr);
    }
    return false;
}

bool gfx_noop_handler_f3dex2(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* filename = (const char*)(cmd)->words.w1;
    uint32_t p = C0(16, 8);
    uint32_t l = C0(0, 16);
    if (p == 0x50) { // PORT: gDPPortWideRectsBegin/End (sWideRects)
        sWideRects = l != 0;
        return false;
    }
    if (p == 0x52) { // PORT: gDPPortAnchorLeft/Right/End (sAnchor)
        static const bool sOff = getenv("PORT_NO_HUD_ANCHOR") != nullptr;
        sAnchor = sOff ? 0 : l == 1 ? -1 : l == 2 ? 1 : l == 3 ? 2 : 0;
        return false;
    }
    if (p == 0x51) { // PORT: gDPPortPreserveColorImage
        sPreserveNextColorImage = l != 0;
        return false;
    }
    if (p == 0x53) { // PORT: gDPPort3DScene: the main picture is a 3D scene's, not a 2D screen (s2DRectArea)
        static const bool sOff = getenv("PORT_NO_3D_SCENE_MARK") != nullptr;
        Interpreter* gfx = mInstance.lock().get();
        if (l != 0 && !sOff && gfx != nullptr && !gfx->mFbActive) {
            sSeen3DThisRun = true;
        }
        return false;
    }
    if (p == 7) {
        g_exec_stack.openDisp(filename, l);
    } else if (p == 8) {
        if (g_exec_stack.disp_stack.size() == 0) {
            SPDLOG_WARN("CLOSE_DISPS without matching open {}:{}", p, l);
        } else {
            g_exec_stack.closeDisp();
        }
    }
    return false;
}

extern "C" uintptr_t Port_DlBufferBase(void) __attribute__((weak));
extern "C" void Port_DlBufferWhere(uintptr_t p, int* which, long* off) __attribute__((weak));
extern "C" int Port_DlBufferActive(void) __attribute__((weak));
bool gfx_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;
    if (gPortDrawLog) { // PORT debug: a matrix address that holds display-list commands
        const uint32_t* m = (const uint32_t*)gfx->SegAddr(mtxAddr);
        if ((uintptr_t)m >= 0x100000000ull && (m[0] >> 24) >= 0xD7 && m[1] == 0 && (m[0] >> 24) != 0xFF) {
            int wm = -1, wc = -1; long om = 0, oc = 0;
            if (Port_DlBufferWhere) { Port_DlBufferWhere((uintptr_t)m, &wm, &om); Port_DlBufferWhere((uintptr_t)cmd, &wc, &oc); }
            fprintf(stderr, "[mtxwhere] mtx buf%d+0x%lx cmd buf%d+0x%lx active=%d\n", wm, om, wc, oc, Port_DlBufferActive ? Port_DlBufferActive() : -1);
            fprintf(stderr, "[mtxdl] off=0x%lx cmd=%p w1=0x%lx -> %p; list:", (unsigned long)(((uintptr_t)m - (Port_DlBufferBase ? Port_DlBufferBase() : 0)) & 0xFFFFFFF), (void*)cmd, (unsigned long)mtxAddr, (const void*)m);
            for (int k = -6; k <= 2; k++) fprintf(stderr, " [%d]%08x/%lx", k, (uint32_t)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
            fputc('\n', stderr);
        }
    }

    gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, (const int32_t*)gfx->SegAddr(mtxAddr));
    return false;
}
// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(16, 8), (const int32_t*)gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)sResourceManager->GetResourceRawPointer((const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)sResourceManager->GetResourceRawPointer((const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(16, 8), mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_filepath_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_filepath_handler_custom_f3d(cmd0);
    }
}

bool gfx_mtx_otr_handler_custom_f3dex2(F3DGfx** cmd0) {
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx = (const int32_t*)sResourceManager->GetResourceRawPointer(hash);

    if (mtx != NULL) {
        Interpreter* gfx = mInstance.lock().get();
        cmd--;
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
        cmd++;
    }

    return false;
}

bool gfx_mtx_otr_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx = (const int32_t*)sResourceManager->GetResourceRawPointer(hash);
    if (mtx != nullptr) {
        cmd--;
        gfx->GfxSpMatrix(C0(16, 8), mtx);
        cmd++;
    }
    return false;
}

bool gfx_mtx_otr_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_handler_custom_f3d(cmd0);
    }
}

bool gfx_pop_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix((uint32_t)(cmd->words.w1 / 64));

    return false;
}

bool gfx_pop_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix(1);

    return false;
}

bool gfx_movemem_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3dex2(C0(0, 8), C0(8, 8) * 8, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3d(C0(16, 8), 0, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    const uint8_t index = C1(24, 8);
    const uint8_t offset = C1(16, 8);
    const uint8_t hasOffset = C1(8, 8);

    (*cmd0)++;

    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    if (ucode_handler_index == ucode_f3dex2) {
        gfx->GfxSpMovememF3dex2(index, offset, sResourceManager->GetResourceRawPointer(hash));
    } else {
        auto light = (Fast::LightEntry*)sResourceManager->GetResourceRawPointer(hash);
        uintptr_t data = (uintptr_t)&light->Ambient;
        gfx->GfxSpMovememF3d(index, offset, (void*)(data + (hasOffset == 1 ? 0x8 : 0)));
    }
    return false;
}

bool gfx_push_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* path = (const char*)gfx->SegAddr(cmd->words.w1);

    if (!gfx_check_image_signature(path)) {
        SPDLOG_ERROR("G_PUSH_SHADER: Shader is not a valid OTR resource name, unable to register push shader");
        return false;
    }

    path = &path[7];

    size_t shaderId = static_cast<size_t>(-1);
    for (const auto& shader : gfx->mShaders) {
        if (strcmp(shader.second, path) == 0) {
            shaderId = shader.first;
            break;
        }
    }

    if (shaderId == static_cast<size_t>(-1)) {
        shaderId = gfx->mShadersIndex++;
        gfx->mShaders[shaderId] = path;
    }

    gfx->mShaderStack.push(shaderId);

    return false;
}

bool gfx_pop_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mShaderStack.pop();

    return false;
}

const char* gfx_get_shader(int16_t id) {
    Interpreter* gfx = mInstance.lock().get();

    for (const std::pair<size_t, const char*>& shader : gfx->mShaders) {
        if (shader.first == id) {
            return shader.second;
        }
    }

    return nullptr; // Use no shader
}

bool gfx_moveword_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3dex2(C0(16, 8), C0(0, 16), cmd->words.w1);

    return false;
}

bool gfx_moveword_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3d(C0(0, 8), C0(8, 16), cmd->words.w1);

    return false;
}

bool gfx_texture_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(1, 7));

    return false;
}

// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_texture_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(0, 8));

    return false;
}

// Almost all versions of the microcode have their own version of this opcode
bool gfx_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    const F3DVtx* vtx = (const F3DVtx*)gfx->SegAddr(cmd->words.w1);
    uintptr_t va = (uintptr_t)vtx;
    if (va < 0x100000000ULL || va >= 0x1000000000000ULL) { // PORT: garbage/unresolved vertex address (user space is 48-bit)
        static int dumped = 0;
        if (dumped++ < 4) {
            fprintf(stderr, "[badvtx] w1=%016lx -> %p, cmd=%p, context:\n", (unsigned long)cmd->words.w1, (void*)vtx,
                    (void*)cmd);
            for (int k = -6; k <= 2; k++)
                fprintf(stderr, "  %+d: %016lx %016lx\n", k, (unsigned long)(cmd + k)->words.w0,
                        (unsigned long)(cmd + k)->words.w1);
        }
        return false;
    }
    gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), vtx);

    return false;
}

bool gfx_vtx_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpVertex(C0(10, 6), C0(17, 7), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex((C0(0, 16)) / sizeof(F3DVtx), C0(16, 4), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_hash_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    // Offset added to the start of the vertices
    const uintptr_t offset = (*cmd0)->words.w1;
    // This is a two-part display list command, so increment the instruction pointer so we can get the CRC64
    // hash from the second
    (*cmd0)++;
    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    // We need to know if the offset is a cached pointer or not. An offset greater than one million is not a
    // real offset, so it must be a real pointer
    if (offset > 0xFFFFF) {
        (*cmd0)--;
        F3DGfx* cmd = *cmd0;
        gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (F3DVtx*)offset);
        (*cmd0)++;
    } else {
        F3DVtx* vtx = (F3DVtx*)sResourceManager->GetResourceRawPointer(hash);

        if (vtx != NULL) {
            vtx = (F3DVtx*)((char*)vtx + offset);

            (*cmd0)--;
            F3DGfx* cmd = *cmd0;

            // TODO: WTF??
            cmd->words.w1 = (uintptr_t)vtx;

            gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), vtx);
            (*cmd0)++;
        }
    }
    return false;
}

bool gfx_vtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    (*cmd0)++;
    cmd = *cmd0;
    size_t vtxCnt = cmd->words.w0;
    size_t vtxIdxOff = cmd->words.w1 >> 16;
    size_t vtxDataOff = cmd->words.w1 & 0xFFFF;
    F3DVtx* vtx = (F3DVtx*)sResourceManager->GetResourceRawPointer((const char*)fileName);
    vtx += vtxDataOff;

    gfx->GfxSpVertex(vtxCnt, vtxIdxOff, vtx);
    return false;
}

bool gfx_dl_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    F3DGfx* nDL = (F3DGfx*)sResourceManager->GetResourceRawPointer((const char*)fileName);

    if (C0(16, 1) == 0 && nDL != nullptr) {
        g_exec_stack.call(*cmd0, nDL);
    } else {
        if (nDL != nullptr) {
            (*cmd0) = nDL;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        } else {
            assert(0 && "???");
            // gfx_path.pop_back();
            // cmd = cmd_stack.top();
            // cmd_stack.pop();
        }
    }
    return false;
}

// The original F3D microcode doesn't seem to have this opcode. Glide handles it as part of moveword
bool gfx_modify_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpModifyVertex(C0(1, 15), C0(16, 8), (uint32_t)cmd->words.w1);
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
// PORT debug: the last display-list jumps (from, raw w1, resolved target), printed with a [badop] dump.
static struct { const void* from; uintptr_t w1; const void* to; } sDlTrail[16];
static unsigned sDlTrailPos;

bool gfx_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(cmd->words.w1);
    sDlTrail[sDlTrailPos++ & 15] = { cmd, (uintptr_t)cmd->words.w1, subGFX };
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call(*cmd0, subGFX);
        }
    } else {
        if (subGFX == nullptr) {
            // PORT PATCH: a branch to NULL is an unpatched placeholder in game-built lists; end this list.
            g_exec_stack.ret();
            return true;
        }
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

bool gfx_dl_otr_hash_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    if (C0(16, 1) == 0) {
        // Push return address
        (*cmd0)++;

        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)sResourceManager->GetResourceRawPointer(hash);

        if (gfx != 0) {
            g_exec_stack.call(cmd, gfx);
        }
    } else {
        Interpreter* gfx = mInstance.lock().get();
        assert(0 && "????");
        (*cmd0) = (F3DGfx*)gfx->SegAddr((*cmd0)->words.w1);
        return true;
    }
    return false;
}
bool gfx_dl_index_handler(F3DGfx** cmd0) {
    // Compute seg addr by converting an index value to an offset value
    // handling 32 vs 64 bit size differences for Gfx
    // adding 1 to trigger the segaddr flow
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);
    uint8_t segNum = (uint8_t)(cmd->words.w1 >> 24);
    uint32_t index = (uint32_t)(cmd->words.w1 & 0x00FFFFFF);
    uintptr_t segAddr = (segNum << 24) | (index * sizeof(F3DGfx)) + 1;

    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(segAddr);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call((*cmd0), subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_pushcd_handler_custom(F3DGfx** cmd0) {
    gfx_push_current_dir((char*)(*cmd0)->words.w1);
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_branch_z_otr_handler_f3dex2(F3DGfx** cmd0) {
    // Push return address
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);

    uint8_t vbidx = (uint8_t)((*cmd0)->words.w0 & 0x00000FFF);
    uint32_t zval = (uint32_t)((*cmd0)->words.w1);

    (*cmd0)++;

    if (gfx->mRsp->loaded_vertices[vbidx].z <= zval ||
        (gfx->mRsp->extra_geometry_mode & G_EX_ALWAYS_EXECUTE_BRANCH) != 0) {
        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)sResourceManager->GetResourceRawPointer(hash);

        if (gfx != 0) {
            (*cmd0) = gfx;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        }
    }
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_end_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mMarkerOn = false;
    g_exec_stack.ret();
    return true;
}

bool gfx_set_prim_depth_handler_rdp(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    uint32_t w1 = (*cmd)->words.w1;
    gfx->mRdp->prim_depth = (uint16_t)((w1 >> 16) & 0x7FFF); // Mask to 15 bits
    return false;
}

// Only on F3DEX2
bool gfx_geometry_mode_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_tri1_otr_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();

    F3DGfx* cmd = *cmd0;
    uint8_t v00 = (uint8_t)(cmd->words.w0 & 0x0000FFFF);
    uint8_t v01 = (uint8_t)(cmd->words.w1 >> 16);
    uint8_t v02 = (uint8_t)(cmd->words.w1 & 0x0000FFFF);
    gfx->GfxSpTri1(v00, v01, v02, false);

    return false;
}

bool gfx_tri1_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);

    return false;
}

bool gfx_tri1_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);

    return false;
}

bool gfx_tri1_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 10, C1(8, 8) / 10, C1(0, 8) / 10, false);

    return false;
}

// F3DEX, and F3DEX2 share a tri2 function, however F3DEX has a different quad function.
bool gfx_tri2_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(17, 7), C0(9, 7), C0(1, 7), false);
    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);
    return false;
}

bool gfx_quad_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    return false;
}

bool gfx_quad_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(0, 8) / 2, C1(24, 8) / 2, false);
    return false;
}

bool gfx_othermode_l_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(31 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, cmd->words.w1);

    return false;
}

bool gfx_othermode_l_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8), C0(0, 8), cmd->words.w1);

    return false;
}

bool gfx_othermode_h_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(63 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, (uint64_t)cmd->words.w1 << 32);

    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_othermode_h_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8) + 32, C0(0, 8), (uint64_t)cmd->words.w1 << 32);

    return false;
}

// A resolved address still in the N64 segmented range (<= 0x0FFFFFFF) usually means SegAddr
// failed to resolve it (segment not set up). Keep it only if it belongs to a loaded module,
// where it's a real low pointer (e.g. a static TLUT) rather than an unresolved segment addr.
static bool IsValidResolvedAddress(uintptr_t addr) {
    if (addr > 0x0FFFFFFF) {
        return true;
    }

    // Still in the N64 segmented range, but might be a false positive (a real low pointer).
#ifdef _WIN32
    // For Windows, check whether the address belongs to a dll.
    HMODULE module = nullptr;
    return GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCSTR>(addr), &module) != 0;
#else
    // For non-Windows platforms, check whether the address belongs to a loaded object.
    Dl_info info;
    return dladdr(reinterpret_cast<void*>(addr), &info) != 0;
#endif
}

bool gfx_set_timg_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t i = (uintptr_t)gfx->SegAddr(cmd->words.w1);
    sLastTimgCmd = cmd;
    if ((i >> 32) == 0xFFFFFFFFull && (uint32_t)i >= 0xA0000000u) { // PORT: a host pointer cut to 32 bits by the game
        static int n = 0;
        if (n++ < 4) {
            fprintf(stderr, "[badtimg] w1=0x%lx list %p:\n", (unsigned long)cmd->words.w1, (void*)cmd);
            for (int k = -6; k <= 4; k++)
                fprintf(stderr, "    %+3d: %08x %016lx\n", k, (uint32_t)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
        }
    }
    static const bool sTimgI16 = getenv("PORT_TILE1LOG") != nullptr;
    if (sTimgI16 && ((((uint32_t)cmd->words.w0) >> 19) & 0x3F) == ((4 << 2) | 2)) { // fmt I, 16b: the detail load
        static int n = 0;
        if (n++ < 3) {
            fprintf(stderr, "[detail] settimg w1=0x%lx -> %p list %p\n", (unsigned long)cmd->words.w1, (void*)i, (void*)cmd);
            for (int k = -8; k <= 8; k++)
                fprintf(stderr, "    %+3d: %08x %016lx\n", k, (uint32_t)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
        }
    }
    static const bool sTimgI8 = getenv("PORT_TIMGLOG_I8") != nullptr;
    if (sTimgI8 && ((((uint32_t)cmd->words.w0) >> 19) & 0x3F) == ((4 << 2) | 1)) {
        static int n = 0;
        if (n++ < 40) fprintf(stderr, "[timg8] w1=0x%lx -> %p list %p\n", (unsigned long)cmd->words.w1, (void*)i, (void*)cmd);
    }
    if (gPortDrawLog) { // PORT debug: raw texture image addresses of the logged frame
        uint32_t w0 = (uint32_t)cmd->words.w0;
        fprintf(stderr, "[timg] w1=0x%lx -> %p fmt=%u siz=%u from list %p\n", (unsigned long)cmd->words.w1, (void*)i,
                (w0 >> 21) & 7, (w0 >> 19) & 3, (void*)cmd);
        if (getenv("PORT_DRAWLOG_SEQ"))
            for (int k = 1; k <= 7; k++) {
                fprintf(stderr, "    +%d: %08x %016lx\n", k, (uint32_t)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
                if (((uint32_t)(cmd + k)->words.w0 >> 24) == 0xDE && (cmd + k)->words.w1 > 0x100000000ull)
                    for (int q = 0; q < 14; q++) {
                        F3DGfx* sub = (F3DGfx*)(cmd + k)->words.w1;
                        fprintf(stderr, "        %2d: %08x %016lx\n", q, (uint32_t)sub[q].words.w0, (unsigned long)sub[q].words.w1);
                        if (((uint32_t)sub[q].words.w0 >> 24) == 0xDF) break;
                    }
            }
        static int dumpedCtx = 0;
        if (dumpedCtx++ < 1) {
            for (int k = 1; k <= 4; k += 3) { // the material's sub-lists (load sequence, geometry)
                F3DGfx* sub = (F3DGfx*)gfx->SegAddr((cmd + k)->words.w1);
                fprintf(stderr, "  sub %+d -> %p\n", k, (void*)sub);
                for (int q = 0; sub && (uintptr_t)sub >= 0x100000000ull && q < 24; q++) {
                    fprintf(stderr, "    %2d: %08x %016lx\n", q, (uint32_t)sub[q].words.w0, (unsigned long)sub[q].words.w1);
                    if (((uint32_t)sub[q].words.w0 >> 24) == 0xDF) break;
                }
            }
        }
    }

    if (i == 0) {
        return false; // PORT: unresolved N64 address
    }
    if (i < 0x100000000ull && PortBadTimgCtx()) { // PORT debug: the commands around a bad SETTIMG
        static int n = 0;
        if (n++ < 2)
            for (int k = -12; k <= 4; k++)
                fprintf(stderr, "  [badctx] %+3d: %08x %016lx\n", k, (uint32_t)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
    }
    if (i < 0x100000000ull) {
        // PORT PATCH: not a host address; a stale or unresolved N64 address must not be dereferenced as a path.
        static int shownT = 0;
        if (shownT++ < 5) fprintf(stderr, "[timg] bad address 0x%lx from w1=0x%lx\n", (unsigned long)i, (unsigned long)cmd->words.w1);
        return false;
    }
    char* imgData = (char*)i;
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetdata = {};
    // Default scale factors to 1 for raw N64 textures. OTR textures set these
    // from the resource, but raw textures would leave them at 0.
    rawTexMetdata.h_byte_scale = 1;
    rawTexMetdata.v_pixel_scale = 1;

    if ((i & 1) != 1) {
        if (gfx_check_image_signature(imgData) == 1) {
            std::shared_ptr<Fast::Texture> tex =
                std::static_pointer_cast<Fast::Texture>(gfx->ResolveResourceCached(imgData));

            if (tex == nullptr) {
                (*cmd0)++;
                return false;
            }

            i = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
            texFlags = tex->Flags;
            rawTexMetdata.width = tex->Width;
            rawTexMetdata.height = tex->Height;
            rawTexMetdata.h_byte_scale = tex->HByteScale;
            rawTexMetdata.v_pixel_scale = tex->VPixelScale;
            rawTexMetdata.type = tex->Type;
            rawTexMetdata.resource = tex;
        }
    }

    if (!IsValidResolvedAddress(i)) {
        return false;
    }

    gfx->GfxDpSetTextureImage(C0(21, 3), C0(19, 2), C0(0, 12) + 1, imgData, texFlags, rawTexMetdata, (void*)i);

    return false;
}

bool gfx_set_timg_otr_hash_handler_custom(F3DGfx** cmd0) {
    uintptr_t addr = (*cmd0)->words.w1;
    (*cmd0)++;
    uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (uint64_t)(*cmd0)->words.w1;

    const char* fileName = sResourceManager->GetArchiveManager()->HashToCString(hash);
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if (fileName == nullptr) {
        (*cmd0)++;
        return false;
    }

    std::shared_ptr<Fast::Texture> texture = std::static_pointer_cast<Fast::Texture>(
        sResourceManager->LoadResourceProcess(sResourceManager->GetArchiveManager()->HashToCString(hash)));
    if (texture != nullptr) {
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        // OTRTODO: We have disabled caching for now to fix a texture corruption issue with HD texture
        // support. In doing so, there is a potential performance hit since we are not caching lookups. We
        // need to do proper profiling to see whether or not it is worth it to keep the caching system.

        char* tex = reinterpret_cast<char*>(texture->ImageData);

        if (tex != nullptr) {
            (*cmd0)--;
            uintptr_t oldData = (*cmd0)->words.w1;
            // TODO: wtf??
            (*cmd0)->words.w1 = (uintptr_t)tex;

            // if (ourHash != (uint64_t)-1) {
            //     auto res = ResourceLoad(ourHash);
            // }

            (*cmd0)++;
        }

        (*cmd0)--;
        F3DGfx* cmd = (*cmd0);
        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        if (tex != NULL) {
            Interpreter* gfx = mInstance.lock().get();
            gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata, tex);
        }
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_HASH: Texture is null");
    }

    (*cmd0)++;
    return false;
}

bool gfx_set_timg_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* fileName = (char*)cmd->words.w1;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    std::shared_ptr<Fast::Texture> texture =
        std::static_pointer_cast<Fast::Texture>(sResourceManager->LoadResourceProcess(fileName));
    if (texture != nullptr) {
        Interpreter* gfx = mInstance.lock().get();
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata,
                                  reinterpret_cast<char*>(texture->ImageData));
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_FILEPATH: Texture is null");
    }
    return false;
}

bool gfx_set_fb_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();

    if (cmd->words.w1) {
        gfx->SetFrameBuffer((int32_t)cmd->words.w1, 1.0f);
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.find((int32_t)cmd->words.w1);
        gfx->mFbActive = true;
    } else {
        gfx->ResetFrameBuffer();
        gfx->mFbActive = false;
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    }
    return false;
}

bool gfx_reset_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();
    gfx->mFbActive = false;
    gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    gfx->mRapi->StartDrawToFramebuffer(gfx->mRendersToFb ? gfx->mGameFb : 0,
                                       (float)gfx->mCurDimensions.height / gfx->mNativeDimensions.height);
    // Force viewport and scissor to reapply against the main framebuffer, in case a previous smaller
    // framebuffer truncated the values
    gfx->mRdp->viewport_or_scissor_changed = true;
    gfx->mRenderingState.viewport = {};
    gfx->mRenderingState.scissor = {};
    return false;
}

bool gfx_copy_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    bool* hasCopiedPtr = (bool*)cmd->words.w1;

    gfx->Flush();
    gfx->CopyFrameBuffer(C0(11, 11), C0(0, 11), (bool)C0(22, 1), hasCopiedPtr);
    return false;
}

bool gfx_read_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    int32_t width, height;
    [[maybe_unused]] int32_t ulx, uly;
    uint16_t* rgba16Buffer = (uint16_t*)cmd->words.w1;
    int fbId = C0(0, 8);
    bool bswap = C0(8, 1);
    ++(*cmd0);
    cmd = *cmd0;
    // Specifying the upper left origin value is unused and unsupported at the renderer level
    ulx = C0(0, 16);
    uly = C0(16, 16);
    width = C1(0, 16);
    height = C1(16, 16);

    gfx->Flush();
    gfx->mRapi->ReadFramebufferToCPU(fbId, width, height, rgba16Buffer);

#ifndef IS_BIGENDIAN
    // byteswap the output to BE
    if (bswap) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            rgba16Buffer[i] = BE16SWAP(rgba16Buffer[i]);
        }
    }
#endif

    return false;
}

bool gfx_register_blended_texture_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // Flush incase we are replacing a previous blended texture that hasn't been finialized to the GPU
    gfx->Flush();

    char* timg = (char*)cmd->words.w1;

    ++(*cmd0);
    cmd = *cmd0;

    uint8_t* mask = (uint8_t*)cmd->words.w0;
    uint8_t* replacementTex = (uint8_t*)cmd->words.w1;

    if (!gfx_check_image_signature(timg)) {
        SPDLOG_ERROR(
            "OTR_G_REGBLENDEDTEX: Texture is not a valid OTR resource name, unable to register blended texture");
        return false;
    }

    // With no mask, we should clear the blended texture
    if (mask == nullptr) {
        gfx->UnregisterBlendedTexture(timg);
    } else {
        gfx->RegisterBlendedTexture(timg, mask, replacementTex);
    }

    return false;
}

bool gfx_set_timg_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->Flush();
    gfx->mRapi->SelectTextureFb((uint32_t)cmd->words.w1);
    gfx->mRdp->textures_changed[0] = false;
    gfx->mRdp->textures_changed[1] = false;
    return false;
}

bool gfx_set_grayscale_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->grayscale = cmd->words.w1;
    return false;
}

bool gfx_load_block_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadBlock(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_load_block_wide_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    uint32_t tile = cmd->words.w0 & 0x7;
    uint32_t lrs = cmd->words.w1;

    (*cmd0)++;
    cmd = *cmd0;

    uint32_t uls = (cmd->words.w0 >> 16) & 0xFFFF;
    uint32_t ult = (cmd->words.w0 >> 0) & 0xFFFF;
    uint32_t dxt = (cmd->words.w1 >> 0) & 0xFFF;

    gfx->GfxDpLoadBlock(tile, uls, ult, lrs, dxt);
    return false;
}

bool gfx_load_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTile(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTile(C0(21, 3), C0(19, 2), C0(9, 9), C0(0, 9), C1(24, 3), C1(20, 4), C1(18, 2), C1(14, 4), C1(10, 4),
                      C1(8, 2), C1(4, 4), C1(0, 4));
    return false;
}

bool gfx_set_tile_size_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_size_interp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    if (gfx->mInterpolationIndex == gfx->mInterpolationIndexTarget) {
        int tile = C1(24, 3);
        gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].uls, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].ult, &(*cmd0)->words.w1, sizeof(float));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].lrs, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].lrt, &(*cmd0)->words.w1, sizeof(float));
    } else {
        ++(*cmd0);
        ++(*cmd0);
    }

    return false;
}

bool gfx_set_tile_size_lerp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    int tile = C1(24, 3);
    float coords[8];
    for (int i = 0; i < 8; i += 2) {
        ++(*cmd0);
        memcpy(&coords[i], &(*cmd0)->words.w0, sizeof(float));
        memcpy(&coords[i + 1], &(*cmd0)->words.w1, sizeof(float));
    }

    float t = gfx->mInterpolationT;
    gfx->mRdp->texture_tile[tile].uls = coords[0] + t * (coords[4] - coords[0]);
    gfx->mRdp->texture_tile[tile].ult = coords[1] + t * (coords[5] - coords[1]);
    gfx->mRdp->texture_tile[tile].lrs = coords[2] + t * (coords[6] - coords[2]);
    gfx->mRdp->texture_tile[tile].lrt = coords[3] + t * (coords[7] - coords[3]);
    gfx->mRdp->textures_changed[0] = true;
    gfx->mRdp->textures_changed[1] = true;

    return false;
}

bool gfx_set_interpolation_index_target(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    gfx->mInterpolationIndexTarget = cmd->words.w1;
    return false;
}

bool gfx_load_tlut_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTlut(C1(24, 3), C1(14, 10));
    return false;
}

bool gfx_set_env_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetEnvColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_prim_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetPrimColor(C0(8, 8), C0(0, 8), C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fog_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFogColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

// CENTER/SCALE and K4/K5 are wired as combiner inputs, so the standard (A-B)*C+D
// shader path covers their common uses. TODO: chroma-key width/threshold
// gating from G_SETKEYR/GB (wR/wG/wB ignored) and the YUV->RGB matrix K0..K3
// applied during texture sampling.
// G_SETKEYR: w1 = [wR:12 | cR:8 | sR:8]
bool gfx_set_key_r_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.r = C1(8, 8);
    gfx->mRdp->key_scale.r = C1(0, 8);
    return false;
}

// G_SETKEYGB: w0 = [op:8 | wG:12 | _:4 | wB:12], w1 = [cG:8 | sG:8 | cB:8 | sB:8]
bool gfx_set_key_gb_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.g = C1(24, 8);
    gfx->mRdp->key_scale.g = C1(16, 8);
    gfx->mRdp->key_center.b = C1(8, 8);
    gfx->mRdp->key_scale.b = C1(0, 8);
    return false;
}

// G_SETCONVERT: w0 = [op:8 | k0:9 | k1:9 | k2_hi:4], w1 = [k2_lo:5 | k3:9 | k4:9 | k5:9]
// K0..K5 are signed 9-bit values; sign-extend after decoding.
bool gfx_set_convert_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->convert_k[0] = sign_extend_9(C0(13, 9));
    gfx->mRdp->convert_k[1] = sign_extend_9(C0(4, 9));
    // k2 is split across w0 and w1
    gfx->mRdp->convert_k[2] = sign_extend_9((C0(0, 4) << 5) | C1(27, 5));
    gfx->mRdp->convert_k[3] = sign_extend_9(C1(18, 9));
    gfx->mRdp->convert_k[4] = sign_extend_9(C1(9, 9));
    gfx->mRdp->convert_k[5] = sign_extend_9(C1(0, 9));
    return false;
}

bool gfx_set_blend_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetBlendColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fill_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFillColor((uint32_t)cmd->words.w1);
    return false;
}

bool gfx_set_intensity_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetGrayscaleColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_combine_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetCombineMode(
        color_comb(C0(20, 4), C1(28, 4), C0(15, 5), C1(15, 3)), alpha_comb(C0(12, 3), C1(12, 3), C0(9, 3), C1(9, 3)),
        color_comb(C0(5, 4), C1(24, 4), C0(0, 5), C1(6, 3)), alpha_comb(C1(21, 3), C1(3, 3), C1(18, 3), C1(0, 3)));
    return false;
}

bool gfx_tex_rect_and_flip_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = C0(12, 12);
    lry = C0(0, 12);
    tile = C1(24, 3);
    ulx = C1(12, 12);
    uly = C1(0, 12);
    // TODO make sure I don't need to increment cmd0
    ++(*cmd0);
    cmd = *cmd0;
    uls = C1(16, 16);
    ult = C1(0, 16);
    ++(*cmd0);
    cmd = *cmd0;
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);

    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_tex_rect_wide_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    lry = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    tile = C1(24, 3);
    ++(*cmd0);
    cmd = *cmd0;
    ulx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    uly = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    ++(*cmd0);
    cmd = *cmd0;
    uls = C0(16, 16);
    ult = C0(0, 16);
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);
    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_image_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int16_t tile, iw, ih;
    int16_t x0, y0, s0, t0;
    int16_t x1, y1, s1, t1;
    tile = C0(0, 3);
    iw = C1(16, 16);
    ih = C1(0, 16);
    cmd = ++(*cmd0);
    x0 = C0(16, 16);
    y0 = C0(0, 16);
    s0 = C1(16, 16);
    t0 = C1(0, 16);
    cmd = ++(*cmd0);
    x1 = C0(16, 16);
    y1 = C0(0, 16);
    s1 = C1(16, 16);
    t1 = C1(0, 16);
    gfx->GfxDpImageRectangle(tile, iw, ih, x0, y0, s0, t0, x1, y1, s1, t1);

    return false;
}

bool gfx_fill_rect_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpFillRectangle(C1(12, 12), C1(0, 12), C0(12, 12), C0(0, 12));
    return false;
}

bool gfx_fill_wide_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);
    int32_t lrx, lry, ulx, uly;

    lrx = (int32_t)(C0(0, 24) << 8) >> 8;
    lry = (int32_t)(C1(0, 24) << 8) >> 8;
    cmd = ++(*cmd0);
    ulx = (int32_t)(C0(0, 24) << 8) >> 8;
    uly = (int32_t)(C1(0, 24) << 8) >> 8;
    gfx->GfxDpFillRectangle(ulx, uly, lrx, lry);

    return false;
}

bool gfx_SetScissor_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetScissor(C1(24, 2), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_z_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetZImage(gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_set_c_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetColorImage(C0(21, 3), C0(19, 2), C0(0, 11), gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_rdp_set_other_mode_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetOtherMode(C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_bg_copy_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexBgCopy((F3DuObjBg*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_bg_1cyc_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->Gfxs2dexBg1cyc((F3DuObjBg*)cmd->words.w1);
    return false;
}

bool gfx_obj_rectangle_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexRecyCopy((F3DuObjSprite*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_extra_geometry_mode_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxSpExtraGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_stubbed_command_handler(F3DGfx** cmd0) {
    return false;
}

bool gfx_spnoop_command_handler_f3dex2(F3DGfx** cmd0) {
    return false;
}

class UcodeHandler {
  public:
    inline constexpr UcodeHandler(
        std::initializer_list<std::pair<int8_t, std::pair<const char*, GfxOpcodeHandlerFunc>>> initializer) {
        std::fill(std::begin(mHandlers), std::end(mHandlers),
                  std::pair<const char*, GfxOpcodeHandlerFunc>(nullptr, nullptr));

        for (const auto& [opcode, handler] : initializer) {
            mHandlers[static_cast<uint8_t>(opcode)] = handler;
        }
    }

    inline bool contains(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)].first != nullptr;
    }

    inline std::pair<const char*, GfxOpcodeHandlerFunc> at(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)];
    }

  private:
    std::pair<const char*, GfxOpcodeHandlerFunc> mHandlers[std::numeric_limits<uint8_t>::max() + 1];
};

static constexpr UcodeHandler rdpHandlers = {
    { RDP_G_SETTARGETINTERPINDEX,
      { "G_SETTARGETINTERPINDEX", gfx_set_interpolation_index_target } }, // G_SETTARGETINTERPINDEX
    { RDP_G_SETTILESIZE_INTERP,
      { "G_SETTILESIZE_INTERP", gfx_set_tile_size_interp_handler_rdp } },                     // G_SETTILESIZE_INTERP
    { RDP_G_SETTILESIZE_LERP, { "G_SETTILESIZE_LERP", gfx_set_tile_size_lerp_handler_rdp } }, // G_SETTILESIZE_LERP
    { RDP_G_TEXRECT, { "G_TEXRECT", gfx_tex_rect_and_flip_handler_rdp } },                    // G_TEXRECT (-28)
    { RDP_G_TEXRECTFLIP, { "G_TEXRECTFLIP", gfx_tex_rect_and_flip_handler_rdp } },            // G_TEXRECTFLIP (-27)
    { RDP_G_RDPLOADSYNC, { "mRdpLOADSYNC", gfx_stubbed_command_handler } },                   // mRdpLOADSYNC (-26)
    { RDP_G_RDPPIPESYNC, { "mRdpPIPESYNC", gfx_stubbed_command_handler } },                   // mRdpPIPESYNC (-25)
    { RDP_G_RDPTILESYNC, { "mRdpTILESYNC", gfx_stubbed_command_handler } },                   // mRdpPIPESYNC (-24)
    { RDP_G_RDPFULLSYNC, { "mRdpFULLSYNC", gfx_stubbed_command_handler } },                   // mRdpFULLSYNC (-23)
    { RDP_G_SETKEYGB, { "G_SETKEYGB", gfx_set_key_gb_handler_rdp } },                         // G_SETKEYGB (-22)
    { RDP_G_SETKEYR, { "G_SETKEYR", gfx_set_key_r_handler_rdp } },                            // G_SETKEYR (-21)
    { RDP_G_SETCONVERT, { "G_SETCONVERT", gfx_set_convert_handler_rdp } },                    // G_SETCONVERT (-20)
    { RDP_G_SETSCISSOR, { "G_SETSCISSOR", gfx_SetScissor_handler_rdp } },                     // G_SETSCISSOR (-19)
    { RDP_G_SETPRIMDEPTH, { "G_SETPRIMDEPTH", gfx_set_prim_depth_handler_rdp } },             // G_SETPRIMDEPTH (-18)
    { RDP_G_RDPSETOTHERMODE, { "mRdpSETOTHERMODE", gfx_rdp_set_other_mode_rdp } },            // mRdpSETOTHERMODE (-17)
    { RDP_G_LOADTLUT, { "G_LOADTLUT", gfx_load_tlut_handler_rdp } },                          // G_LOADTLUT (-16)
    { RDP_G_SETTILESIZE, { "G_SETTILESIZE", gfx_set_tile_size_handler_rdp } },                // G_SETTILESIZE (-14)
    { RDP_G_LOADBLOCK, { "G_LOADBLOCK", gfx_load_block_handler_rdp } },                       // G_LOADBLOCK (-13)
    { RDP_G_LOADTILE, { "G_LOADTILE", gfx_load_tile_handler_rdp } },                          // G_LOADTILE (-12)
    { RDP_G_SETTILE, { "G_SETTILE", gfx_set_tile_handler_rdp } },                             // G_SETTILE (-11)
    { RDP_G_FILLRECT, { "G_FILLRECT", gfx_fill_rect_handler_rdp } },                          // G_FILLRECT (-10)
    { RDP_G_SETFILLCOLOR, { "G_SETFILLCOLOR", gfx_set_fill_color_handler_rdp } },             // G_SETFILLCOLOR (-9)
    { RDP_G_SETFOGCOLOR, { "G_SETFOGCOLOR", gfx_set_fog_color_handler_rdp } },                // G_SETFOGCOLOR (-8)
    { RDP_G_SETBLENDCOLOR, { "G_SETBLENDCOLOR", gfx_set_blend_color_handler_rdp } },          // G_SETBLENDCOLOR (-7)
    { RDP_G_SETPRIMCOLOR, { "G_SETPRIMCOLOR", gfx_set_prim_color_handler_rdp } },             // G_SETPRIMCOLOR (-6)
    { RDP_G_SETENVCOLOR, { "G_SETENVCOLOR", gfx_set_env_color_handler_rdp } },                // G_SETENVCOLOR (-5)
    { RDP_G_SETCOMBINE, { "G_SETCOMBINE", gfx_set_combine_handler_rdp } },                    // G_SETCOMBINE (-4)
    { RDP_G_SETTIMG, { "G_SETTIMG", gfx_set_timg_handler_rdp } },                             // G_SETTIMG (-3)
    { RDP_G_SETZIMG, { "G_SETZIMG", gfx_set_z_img_handler_rdp } },                            // G_SETZIMG (-2)
    { RDP_G_SETCIMG, { "G_SETCIMG", gfx_set_c_img_handler_rdp } },                            // G_SETCIMG (-1)
};

static constexpr UcodeHandler otrHandlers = {
    { OTR_G_SETTIMG_OTR_HASH,
      { "G_SETTIMG_OTR_HASH", gfx_set_timg_otr_hash_handler_custom } },       // G_SETTIMG_OTR_HASH (0x20)
    { OTR_G_SETFB, { "G_SETFB", gfx_set_fb_handler_custom } },                // G_SETFB (0x21)
    { OTR_G_RESETFB, { "G_RESETFB", gfx_reset_fb_handler_custom } },          // G_RESETFB (0x22)
    { OTR_G_SETTIMG_FB, { "G_SETTIMG_FB", gfx_set_timg_fb_handler_custom } }, // G_SETTIMG_FB (0x23)
    { OTR_G_VTX_OTR_FILEPATH,
      { "G_VTX_OTR_FILEPATH", gfx_vtx_otr_filepath_handler_custom } }, // G_VTX_OTR_FILEPATH (0x24)
    { OTR_G_SETTIMG_OTR_FILEPATH,
      { "G_SETTIMG_OTR_FILEPATH", gfx_set_timg_otr_filepath_handler_custom } }, // G_SETTIMG_OTR_FILEPATH (0x25)
    { OTR_G_TRI1_OTR, { "G_TRI1_OTR", gfx_tri1_otr_handler_f3dex2 } },          // G_TRI1_OTR (0x26)
    { OTR_G_DL_OTR_FILEPATH, { "G_DL_OTR_FILEPATH", gfx_dl_otr_filepath_handler_custom } }, // G_DL_OTR_FILEPATH (0x27)
    { OTR_G_PUSHCD, { "G_PUSHCD", gfx_pushcd_handler_custom } },                            // G_PUSHCD (0x28)
    { OTR_G_MTX_OTR_FILEPATH,
      { "G_MTX_OTR_FILEPATH", gfx_mtx_otr_filepath_handler_custom } },          // G_MTX_OTR_FILEPATH (0x29)
    { OTR_G_DL_OTR_HASH, { "G_DL_OTR_HASH", gfx_dl_otr_hash_handler_custom } }, // G_DL_OTR_HASH (0x31)
    { OTR_G_VTX_OTR_HASH, { "G_VTX_OTR_HASH", gfx_vtx_hash_handler_custom } },  // G_VTX_OTR_HASH (0x32)
    { OTR_G_MARKER, { "G_MARKER", gfx_marker_handler_otr } },                   // G_MARKER (0X33)
    { OTR_G_INVALTEXCACHE, { "G_INVALTEXCACHE", gfx_invalidate_tex_cache_handler_f3dex2 } }, // G_INVALTEXCACHE (0X34)
    { OTR_G_BRANCH_Z_OTR, { "G_BRANCH_Z_OTR", gfx_branch_z_otr_handler_f3dex2 } },           // G_BRANCH_Z_OTR (0x35)
    { OTR_G_MTX_OTR, { "G_MTX_OTR", gfx_mtx_otr_handler_custom } },                          // G_MTX_OTR (0x36)
    { OTR_G_TEXRECT_WIDE, { "G_TEXRECT_WIDE", gfx_tex_rect_wide_handler_custom } },          // G_TEXRECT_WIDE (0x37)
    { OTR_G_FILLWIDERECT, { "G_FILLWIDERECT", gfx_fill_wide_rect_handler_custom } },         // G_FILLWIDERECT (0x38)
    { OTR_G_SETGRAYSCALE, { "G_SETGRAYSCALE", gfx_set_grayscale_handler_custom } },          // G_SETGRAYSCALE (0x39)
    { OTR_G_EXTRAGEOMETRYMODE,
      { "G_EXTRAGEOMETRYMODE", gfx_extra_geometry_mode_handler_custom } }, // G_EXTRAGEOMETRYMODE (0x3a)
    { OTR_G_COPYFB, { "G_COPYFB", gfx_copy_fb_handler_custom } },          // G_COPYFB (0x3b)
    { OTR_G_IMAGERECT, { "G_IMAGERECT", gfx_image_rect_handler_custom } }, // G_IMAGERECT (0x3c)
    { OTR_G_DL_INDEX, { "G_DL_INDEX", gfx_dl_index_handler } },            // G_DL_INDEX (0x3d)
    { OTR_G_READFB, { "G_READFB", gfx_read_fb_handler_custom } },          // G_READFB (0x3e)
    { OTR_G_REGBLENDEDTEX,
      { "G_REGBLENDEDTEX", gfx_register_blended_texture_handler_custom } },         // G_REGBLENDEDTEX (0x3f)
    { OTR_G_SETINTENSITY, { "G_SETINTENSITY", gfx_set_intensity_handler_custom } }, // G_SETINTENSITY (0x40)
    { OTR_G_MOVEMEM_HASH, { "OTR_G_MOVEMEM_HASH", gfx_movemem_handler_otr } },      // OTR_G_MOVEMEM_HASH
    { OTR_G_PUSH_SHADER, { "G_PUSH_SHADER", gfx_push_shader } },
    { OTR_G_POP_SHADER, { "G_POP_SHADER", gfx_pop_shader } },
    { RDP_G_LOADBLOCK_WIDE, { "G_LOADBLOCK_WIDE", gfx_load_block_wide_handler_rdp } }, // RDP_G_LOADBLOCK_WIDE (-15)
    { RDP_G_VTX_WIDE, { "G_VTX_WIDE", gfx_vtx_handler_f3dex2 } },                      // RDP_G_VTX_WIDE (-16)
    { RDP_G_TRI1_WIDE, { "G_TRI1_WIDE", gfx_tri1_handler_f3dex2 } },                   // RDP_G_TRI1_WIDE (-17)
};

static constexpr UcodeHandler f3dex2Handlers = {
    { F3DEX2_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_SPNOOP, { "G_SPNOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX2_G_MTX, { "G_MTX", gfx_mtx_handler_f3dex2 } },
    { F3DEX2_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3dex2 } },
    { F3DEX2_G_MOVEMEM, { "G_MOVEMEM", gfx_movemem_handler_f3dex2 } },
    { F3DEX2_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3dex2 } },
    { F3DEX2_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3dex2 } },
    { F3DEX2_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex2 } },
    { F3DEX2_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX2_G_GEOMETRYMODE, { "G_GEOMETRYMODE", gfx_geometry_mode_handler_f3dex2 } },
    { F3DEX2_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex2 } },
    { F3DEX2_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX2_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3dex2 } },
};

static constexpr UcodeHandler f3dexHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
    { F3DEX_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex } },
};

static constexpr UcodeHandler f3dHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3d } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3d } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
};

// LUSTODO: These S2DEX commands have different opcode numbers on F3DEX2 vs other ucodes. More research needs to be done
// to see if the implementations are different.
static constexpr UcodeHandler s2dexHandlers = {
    { F3DEX2_G_BG_COPY, { "G_BG_COPY", gfx_bg_copy_handler_s2dex } },
    { F3DEX2_G_BG_1CYC, { "G_BG_1CYC", gfx_bg_1cyc_handler_s2dex } },
    { F3DEX2_G_OBJ_RENDERMODE, { "G_OBJ_RENDERMODE", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE_R, { "G_OBJ_RECTANGLE_R", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE, { "G_OBJ_RECTANGLE", gfx_obj_rectangle_handler_s2dex } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
};

static constexpr std::array ucode_handlers = {
    &f3dHandlers,    // ucode_f3db
    &f3dHandlers,    // ucode_f3d
    &f3dexHandlers,  // ucode_f3dex
    &f3dexHandlers,  // ucode_f3dexb
    &f3dex2Handlers, // ucode_f3dex2
    &s2dexHandlers,  // ucode_s2dex
};

const char* GfxGetOpcodeName(int8_t opcode) {
    if (otrHandlers.contains(opcode)) {
        return otrHandlers.at(opcode).first;
    }

    if (rdpHandlers.contains(opcode)) {
        return rdpHandlers.at(opcode).first;
    }

    if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            return ucode_handlers[ucode_handler_index]->at(opcode).first;
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    return nullptr;
}

// TODO, implement a system where we can get the current opcode handler by writing to the GWords. If the powers that be
// are OK with that...
static void gfx_set_ucode_handler(UcodeHandlers ucode) {
    // Loaded ucode must be in range of the supported ucode_handlers
    assert(ucode < ucode_max);
    Interpreter* gfx = mInstance.lock().get();
    ucode_handler_index = ucode;

    // Reset some RSP state values upon ucode load to deal with hardware quirks discovered by emulators
    switch (ucode) {
        case ucode_f3d:
        case ucode_f3db:
        case ucode_f3dex:
        case ucode_f3dexb:
        case ucode_f3dex2:
            gfx->mRsp->fog_mul = 0;
            gfx->mRsp->fog_offset = 0;
            break;
        default:
            break;
    }
}

static void gfx_step() {
    auto& cmd = g_exec_stack.currCmd();
    auto cmd0 = cmd;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);

#ifdef USE_GBI_TRACE
    if (cmd->words.trace.valid && mConsoleVariable->GetInteger("gEnableGFXTrace", 0)) {
#define TRACE                                  \
    "\n====================================\n" \
    " - CMD: {:02X}\n"                         \
    " - Path: {}:{}\n"                         \
    " - W0: {:08X}\n"                          \
    " - W1: {:08X}\n"                          \
    "===================================="
        SPDLOG_INFO(TRACE, (uint8_t)opcode, cmd->words.trace.file, cmd->words.trace.idx, cmd->words.w0, cmd->words.w1);
    }
#endif

    if (opcode == F3DEX2_G_LOAD_UCODE) {
        gfx_set_ucode_handler((UcodeHandlers)(cmd->words.w0 & 0xFFFFFF));
        ++cmd;
        return;
        // Instead of having a handler for each ucode for switching ucode, just check for it early and return.
    }

    if (otrHandlers.contains(opcode)) {
        // OTR filepath handlers expect w1 to be a valid string pointer.
        // Guard against null or N64-segment addresses that would crash in strlen/strncmp.
        if (opcode == OTR_G_VTX_OTR_FILEPATH || opcode == OTR_G_SETTIMG_OTR_FILEPATH ||
            opcode == OTR_G_DL_OTR_FILEPATH || opcode == OTR_G_PUSHCD || opcode == OTR_G_MTX_OTR_FILEPATH) {
            uintptr_t w1 = (uintptr_t)cmd->words.w1;
            if (w1 < 0x10000
#if UINTPTR_MAX > 0xFFFFFFFFu
                // On 64-bit: filter kernel/sentinel addresses.
                || w1 > 0x0000FFFFFFFFFFFFull
#endif
            ) {
                ++g_exec_stack.currCmd();
                return;
            }
        }
        if (otrHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (rdpHandlers.contains(opcode)) {
        if (rdpHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            if (ucode_handlers[ucode_handler_index]->at(opcode).second(&cmd)) {
                return;
            }
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
            static int dumped = 0;
            if (dumped++ < 2) { // PORT: show the commands around the garbage to identify the list that produced it
                fprintf(stderr, "[badop] cmd=%p, context:\n", (void*)cmd);
                for (int k = -6; k <= 2; k++) fprintf(stderr, "  %+d: %016lx %016lx\n", k, (unsigned long)(cmd + k)->words.w0, (unsigned long)(cmd + k)->words.w1);
                for (unsigned k = 0; k < 16; k++) {
                    auto& e = sDlTrail[(sDlTrailPos + k) & 15];
                    if (e.from) fprintf(stderr, "  dl %p: w1=0x%lx -> %p\n", e.from, (unsigned long)e.w1, e.to);
                }
            }
            // PORT: an opcode the microcode does not have means this list is not a display list (e.g. a freed buffer
            // already reused for asset data); end it instead of interpreting the data.
            g_exec_stack.ret();
            return;
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    ++cmd;
}

void Interpreter::SpReset() {
    while (!mShaderStack.empty()) {
        mShaderStack.pop();
    }
    mRsp->modelview_matrix_stack_size = 1;
    mRsp->current_num_lights = 2;
    mRsp->lights_changed = true;
    mRsp->lookat[0].dir[0] = 0;
    mRsp->lookat[0].dir[1] = 127;
    mRsp->lookat[0].dir[2] = 0;
    mRsp->lookat[1].dir[0] = 127;
    mRsp->lookat[1].dir[1] = 0;
    mRsp->lookat[1].dir[2] = 0;
    CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
    CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
}

void Interpreter::RegisterFbTexture(const void* cpuAddr, int fbId) {
    mFbTextures[(uintptr_t)cpuAddr] = fbId;
}

void Interpreter::UnregisterFbTexture(const void* cpuAddr) {
    mFbTextures.erase((uintptr_t)cpuAddr);
}

void Interpreter::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    mWapi->GetDimensions(width, height, posX, posY);
}

void Interpreter::Init(class GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name,
                       bool start_in_fullscreen, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY,
                       std::shared_ptr<Ship::ConsoleVariable> consoleVariable,
                       std::shared_ptr<Ship::ResourceManager> resourceManager) {
    mWapi = wapi;
    mRapi = rapi;
    mResourceManager = std::move(resourceManager);
    sResourceManager = mResourceManager;
    mConsoleVariable = std::move(consoleVariable);
    mWapi->Init(game_name, rapi->GetName(), start_in_fullscreen, width, height, posX, posY);
    mRapi->Init();
    mRapi->UpdateFramebufferParameters(0, width, height, 1, false, true, true, true);
    mCurDimensions.internal_mul = mConsoleVariable->GetFloat(CVAR_INTERNAL_RESOLUTION, 1);
    mMsaaLevel = mConsoleVariable->GetInteger(CVAR_MSAA_VALUE, 1);

    mCurDimensions.width = width;
    mCurDimensions.height = height;

    mGameFb = mRapi->CreateFramebuffer();
    mGameFbMsaaResolved = mRapi->CreateFramebuffer();

    mNativeDimensions.width = getenv("PORT_NATIVE_W") ? atoi(getenv("PORT_NATIVE_W")) : SCREEN_WIDTH;   // PORT: Stadium draws in a 640x480 space
    mNativeDimensions.height = getenv("PORT_NATIVE_H") ? atoi(getenv("PORT_NATIVE_H")) : SCREEN_HEIGHT;

    for (int i = 0; i < MAX_SEGMENT_POINTERS; i++) {
        mSegmentPointers[i] = 0;
    }

    if (mTexUploadBuffer == nullptr) {
        // We cap texture max to 8k, because why would you need more?
        int max_tex_size = std::min(8192, mRapi->GetMaxTextureSize());
        mTexUploadBuffer = (uint8_t*)malloc(max_tex_size * max_tex_size * 4);
    }

    ucode_handler_index = UcodeHandlers::ucode_f3dex2;

    // Pre-allocate texture cache buckets to prevent rehash-induced iterator invalidation.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
}

void Interpreter::Destroy() {
    // TODO: should also destroy rapi, and any other resources acquired in fast3d
    free(mTexUploadBuffer);
    mWapi->Destroy();

    // Texture cache and loaded textures store references to Resources which need to be unreferenced.
    TextureCacheClear();
    mRdp->texture_to_load.raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[0].raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[1].raw_tex_metadata.resource = nullptr;
}

GfxRenderingAPI* Interpreter::GetCurrentRenderingAPI() {
    return mRapi;
}

void Interpreter::SetGfxDebugger(std::shared_ptr<GfxDebugger> debugger) {
    mGfxDebugger = std::move(debugger);
}

std::shared_ptr<GfxDebugger> Interpreter::GetGfxDebugger() const {
    return mGfxDebugger;
}

void Interpreter::SetFast3dWindow(std::shared_ptr<Fast3dWindow> window) {
    mFast3dWindow = std::move(window);
}

std::shared_ptr<Fast3dWindow> Interpreter::GetCurrentWindow() {
    if (auto inst = mInstance.lock()) {
        return inst->mFast3dWindow.lock();
    }
    return nullptr;
}

void Interpreter::HandleWindowEvents() {
    mWapi->HandleEvents();
}

bool Interpreter::IsFrameReady() {
    return mWapi->IsFrameReady();
}

bool Interpreter::ViewportMatchesRendererResolution() {
#ifdef __APPLE__
    // Always treat the viewport as not matching the render resolution on mac
    // to avoid issues with retina scaling.
    return false;
#else
    if (mCurDimensions.width == mGameWindowViewport.width && mCurDimensions.height == mGameWindowViewport.height) {
        return true;
    }
    return false;
#endif
}

void Interpreter::StartFrame() {
    mWapi->GetDimensions(&mGfxCurrentWindowDimensions.width, &mGfxCurrentWindowDimensions.height, &mCurWindowPosX,
                         &mCurWindowPosY);
    if (mCurDimensions.height == 0) {
        // Avoid division by zero
        mCurDimensions.height = 1;
    }
    mCurDimensions.aspect_ratio = (float)mCurDimensions.width / (float)mCurDimensions.height;

    // Update the framebuffer sizes when the viewport or native dimension changes
    if (mCurDimensions.width != mPrvDimensions.width || mCurDimensions.height != mPrvDimensions.height ||
        mNativeDimensions.width != mPrevNativeDimensions.width ||
        mNativeDimensions.height != mPrevNativeDimensions.height) {

        for (auto& fb : mFrameBuffers) {
            uint32_t width = fb.second.orig_width, height = fb.second.orig_height;
            if (fb.second.resize) {
                AdjustWidthHeightForScale(width, height, fb.second.native_width, fb.second.native_height);
            }
            if (width != fb.second.applied_width || height != fb.second.applied_height) {
                mRapi->UpdateFramebufferParameters(fb.first, width, height, 1, true, true, true, true);
                fb.second.applied_width = width;
                fb.second.applied_height = height;
            }
        }
    }

    mPrvDimensions = mCurDimensions;
    mPrevNativeDimensions = mNativeDimensions;
    // PORT: always render into the game framebuffer: like the N64's, it keeps its picture between frames (menus draw
    // their static background once and then only what changes; see Run).
    if (true || !ViewportMatchesRendererResolution() || mMsaaLevel > 1) {
        mRendersToFb = true;
        std::vector<int> fbs{ mGameFb };
        for (auto& e : sMainFbs) {
            if (e.second != mGameFb) {
                fbs.push_back(e.second);
            }
        }
        for (auto& e : sSnapFbs) { // (frame interpolation's copies of them)
            fbs.push_back(e.second.first);
            e.second.second = mMsaaLevel;
        }
        for (int fb : fbs) {
            if (!ViewportMatchesRendererResolution()) {
                mRapi->UpdateFramebufferParameters(fb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel, true,
                                                   true, true, true);
            } else {
                // MSAA framebuffer needs to be resolved to an equally sized target when complete, which must therefore
                // match the window size
                // PORT: Y-inverted when shown as a texture (see above): only an MSAA resolve goes into the window
                mRapi->UpdateFramebufferParameters(fb, mGfxCurrentWindowDimensions.width,
                                                   mGfxCurrentWindowDimensions.height, mMsaaLevel, mMsaaLevel <= 1, true,
                                                   true, true);
            }
        }
        if (mMsaaLevel > 1 && !ViewportMatchesRendererResolution()) {
            mRapi->UpdateFramebufferParameters(mGameFbMsaaResolved, mCurDimensions.width, mCurDimensions.height, 1,
                                               false, false, false, false);
        }
    } else {
        mRendersToFb = false;
    }

    mFbActive = false;
}

GfxExecStack g_exec_stack = {};

void Interpreter::RunGuiOnly() {
    SpReset();

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(true, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    Flush();
    mGfxFrameBuffer = 0;
    mGfxFrameBufferFb = -1;

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution()) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
                mGfxFrameBufferFb = mGameFbMsaaResolved;
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
            mGfxFrameBufferFb = mGameFb;
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

static bool sOpLog = getenv("PORT_OPLOG") != nullptr;
static int sOpHist[256];
static int sOpFrames = 0;

static F3DGfx* sLastCmd = nullptr;
bool Interpreter::TrackSegmentsOnly(Gfx* commands) {
    static const bool sFillLog = getenv("PORT_TRACE_FILL") != nullptr; // debug: fill rectangles seen by the state pass
    if (PortTile1Log()) fprintf(stderr, "[tile1] -- track begin\n");
    // PORT: state pass for a frame that is not drawn. The N64 keeps the RSP segment table and the RDP's TMEM/tile
    // state across tasks, and the game relies on that (e.g. a texture loaded once when a scene starts). This walks the
    // display-list tree following G_DL / G_ENDDL and applies segment writes and texture loads through the normal
    // handlers. Returns true if the frame must be drawn after all: it renders into an off-screen color image, or it
    // draws over the previous picture instead of repainting the whole screen (no full-screen fill rectangle; the frame
    // buffer keeps its colour between frames, see Run).
    std::vector<F3DGfx*> stack;
    F3DGfx* cmd = (F3DGfx*)commands;
    uintptr_t cimg = 0, zimg = 0;
    uint32_t cimgWidth = 320;
    bool fullFill = false;
    uint32_t draws = 0;
    static uint32_t sLastDraws = 0;
    for (int budget = 0; cmd != nullptr && budget < 400000; budget++) {
        if ((uintptr_t)cmd < 0x100000000ull) {
            break; // unresolved address: stop this branch
        }
        uint32_t w0 = (uint32_t)cmd->words.w0;
        uintptr_t w1 = cmd->words.w1;
        uint8_t op = (uint8_t)(w0 >> 24);
        bool invalid = op > 0x07 && op < 0xD3; // not an F3DEX2 opcode: not a display list (stale/reused buffer)
        if (op == 0xDF /* G_ENDDL */ || invalid) {
            if (stack.empty()) {
                break;
            }
            cmd = stack.back();
            stack.pop_back();
            continue;
        }
        if (op == 0xDE /* G_DL */) {
            F3DGfx* sub = (F3DGfx*)SegAddr(w1);
            if (((w0 >> 16) & 0xFF) == 0) { // call
                if (sub != nullptr && stack.size() < 64) {
                    stack.push_back(cmd + 1);
                    cmd = sub;
                    continue;
                }
            } else { // branch
                if (sub == nullptr) {
                    if (stack.empty()) {
                        break;
                    }
                    cmd = stack.back();
                    stack.pop_back();
                } else {
                    cmd = sub;
                }
                continue;
            }
        } else if (op == 0xDB /* G_MOVEWORD */ && ((w0 >> 16) & 0xFF) == G_MW_SEGMENT) {
            int seg = (int)((w0 & 0xFFFF) / 4);
            if (seg >= 0 && seg < MAX_SEGMENT_POINTERS) {
                mSegmentPointers[seg] = w1;
            }
        } else if (op == 0xFF /* G_SETCIMG */) {
            uint32_t width = (w0 & 0xFFF) + 1;
            if (width != 320 && width != 640 && w1 != 0) {
                return true; // render-to-texture in this frame
            }
            cimg = (uintptr_t)SegAddr(w1);
            if (IsUnshownScreenImage(width, (const void*)cimg)) {
                // PORT: a screen-sized render target (the Bonus stickers sheet gets one thumbnail per frame; skipped
                // frames lost theirs, and the sheet's white fill)
                return true;
            }
            cimgWidth = width;
            sMainFbDrawStamp[cimg] = ++sDrawStamp; // the RDP renders this frame buffer (even if not drawn here)
        } else if (op == 0xFE /* G_SETZIMG */) {
            zimg = (uintptr_t)SegAddr(w1);
        } else if (op == 0x05 || op == 0x06 || op == 0x07 || op == 0xE4 || op == 0xE5) {
            draws++; // G_TRI1, G_TRI2, G_QUAD, G_TEXRECT, G_TEXRECTFLIP
        } else if (op == 0xF6 /* G_FILLRECT */) {
            draws++;
            uint32_t lrx = (w0 >> 14) & 0x3FF, lry = (w0 >> 2) & 0x3FF;
            uint32_t ulx = ((uint32_t)w1 >> 14) & 0x3FF, uly = ((uint32_t)w1 >> 2) & 0x3FF;
            uint32_t h = cimgWidth > 400 ? 480 : 240;
            if (sFillLog) fprintf(stderr, "[fill] %u,%u-%u,%u cimgW=%u z=%d\n", ulx, uly, lrx, lry, cimgWidth, (int)(cimg == zimg));
            if (cimg != zimg && ulx == 0 && uly == 0 && lrx + 1 >= cimgWidth && lry + 1 >= h) {
                fullFill = true;
            }
        } else if (op == 0xFD || op == 0xF5 || op == 0xF2 || op == 0xF3 || op == 0xF4 || op == 0xF0) {
            // G_SETTIMG, G_SETTILE, G_SETTILESIZE, G_LOADBLOCK, G_LOADTILE, G_LOADTLUT: TMEM / tile state
            int8_t sop = (int8_t)op;
            if (rdpHandlers.contains(sop)) {
                F3DGfx* c = cmd;
                rdpHandlers.at(sop).second(&c);
            }
        }
        cmd++;
    }
    static const bool sSkipAll = getenv("PORT_SKIP_INCREMENTAL") != nullptr; // debug: skip such frames anyway
    // A frame that draws over the previous picture loses something only if it draws what later frames do not repeat:
    // screens paint their static layers in the first frames (or on every frame of a fade) and then only what changes,
    // so the amount of drawing changes there. Steady animation (the same amount every frame) is still skipped.
    // Tracked per N64 frame buffer (each has its own persistent target): a frame whose buffer last got the same amount
    // of drawing repaints the same things (steady animation) and can be skipped.
    static std::unordered_map<uintptr_t, uint32_t> sLastDrawsPerBuffer;
    uint32_t& last = sLastDrawsPerBuffer[cimg];
    bool mustDraw = !fullFill && draws > 0 && draws != last && !sSkipAll;
    last = draws;
    if (PortTraceFill()) fprintf(stderr, "[fill] frame full=%d draws=%u must=%d\n", (int)fullFill, draws, (int)mustDraw);
    sLastDraws = draws;
    return mustDraw;
}

// PORT: frame interpolation (see ReplayBegin in interpreter.h)
namespace {
struct ReplayState {
    bool begun = false, haveEnd = false;
    RSP rspStart, rspEnd;
    RDP rdpStart, rdpEnd;
    uintptr_t segStart[MAX_SEGMENT_POINTERS], segEnd[MAX_SEGMENT_POINTERS];
    XYWidthHeight nativeStart, nativeEnd;
    std::list<TmemRegion> tmemStart, tmemEnd;
};
ReplayState* sReplay = nullptr;
// A copy of the TMEM model, with the RDP's loaded-texture pointers into the old list moved to the copy.
std::list<TmemRegion> CopyTmem(const std::list<TmemRegion>& from, RDP& rdp) {
    std::list<TmemRegion> to = from;
    for (auto& lt : rdp.loaded_texture) {
        auto d = to.begin();
        for (auto s = from.begin(); s != from.end(); ++s, ++d) {
            const uint8_t* b = s->data.data();
            if (lt.addr >= b && lt.addr < b + s->data.size()) {
                lt.addr = d->data.data() + (lt.addr - b);
                break;
            }
        }
    }
    return to;
}
} // namespace

void Interpreter::Interp2DPrepare(Gfx* commands) {
    sRectMove.clear();
    static const bool sOff = getenv("PORT_INTERP_2D") != nullptr && atoi(getenv("PORT_INTERP_2D")) == 0;
    if (sOff || sRectLast.empty()) return;
    // the frame's rectangles: a walk like TrackSegmentsOnly's (texture state through the normal handlers), state
    // restored afterwards
    RSP savedRsp = *mRsp;
    RDP savedRdp = *mRdp;
    uintptr_t savedSegments[MAX_SEGMENT_POINTERS];
    memcpy(savedSegments, mSegmentPointers, sizeof(savedSegments));
    std::list<TmemRegion> savedTmem = CopyTmem(sTmemRegions, savedRdp);
    Interp2DRects cur;
    std::vector<F3DGfx*> stack;
    F3DGfx* cmd = (F3DGfx*)commands;
    bool main = true;
    uintptr_t cimg = 0, zimg = 0;
    for (int budget = 0; cmd != nullptr && budget < 400000; budget++) {
        if ((uintptr_t)cmd < 0x100000000ull) break;
        uint32_t w0 = (uint32_t)cmd->words.w0;
        uintptr_t w1 = cmd->words.w1;
        uint8_t op = (uint8_t)(w0 >> 24);
        if (op == 0xDF /* G_ENDDL */ || (op > 0x07 && op < 0xD3)) {
            if (stack.empty()) break;
            cmd = stack.back();
            stack.pop_back();
            continue;
        }
        if (op == 0xDE /* G_DL */) {
            F3DGfx* sub = (F3DGfx*)SegAddr(w1);
            if (((w0 >> 16) & 0xFF) == 0) {
                if (sub != nullptr && stack.size() < 64) {
                    stack.push_back(cmd + 1);
                    cmd = sub;
                    continue;
                }
            } else {
                if (sub == nullptr) {
                    if (stack.empty()) break;
                    cmd = stack.back();
                    stack.pop_back();
                } else {
                    cmd = sub;
                }
                continue;
            }
        } else if (op == 0xDB /* G_MOVEWORD */ && ((w0 >> 16) & 0xFF) == G_MW_SEGMENT) {
            int seg = (int)((w0 & 0xFFFF) / 4);
            if (seg >= 0 && seg < MAX_SEGMENT_POINTERS) mSegmentPointers[seg] = w1;
        } else if (op == 0xFF /* G_SETCIMG */) {
            uint32_t width = (w0 & 0xFFF) + 1;
            cimg = (uintptr_t)SegAddr(w1);
            main = width == 320 || width == 640;
        } else if (op == 0xFE /* G_SETZIMG */) {
            zimg = (uintptr_t)SegAddr(w1);
        } else if (op == 0xE4 || op == 0xE5 /* G_TEXRECT(FLIP) */) {
            const int32_t lrx = (w0 >> 12) & 0xFFF, lry = w0 & 0xFFF;
            const uint8_t tile = (uint8_t)((w1 >> 24) & 7);
            const int32_t ulx = ((uint32_t)w1 >> 12) & 0xFFF, uly = (uint32_t)w1 & 0xFFF;
            const uint32_t st = (uint32_t)cmd[1].words.w1;
            if (main) {
                const TmemRegion* r = TmemFind(mRdp->texture_tile[tile].tmem);
                const int16_t uls = (int16_t)(st >> 16), ult = (int16_t)(st & 0xFFFF);
                cur[{ r ? (const void*)r->raw : nullptr, lrx - ulx, lry - uly,
                      ((uint32_t)(uint16_t)uls << 16) | (uint16_t)ult }]
                    .push_back({ ulx, uly });
            }
            cmd += 3;
            continue;
        } else if (op == 0xF6 /* G_FILLRECT */) {
            const int32_t lrx = (w0 >> 12) & 0xFFF, lry = w0 & 0xFFF;
            const int32_t ulx = ((uint32_t)w1 >> 12) & 0xFFF, uly = (uint32_t)w1 & 0xFFF;
            if (main && cimg != zimg) {
                cur[{ nullptr, lrx - ulx, lry - uly,
                      (uint32_t)mRdp->fill_color.r << 24 | mRdp->fill_color.g << 16 | mRdp->fill_color.b << 8 |
                          mRdp->fill_color.a }]
                    .push_back({ ulx, uly });
            }
        } else if (op == 0xFD || op == 0xF5 || op == 0xF2 || op == 0xF3 || op == 0xF4 || op == 0xF0 || op == 0xF7) {
            // G_SETTIMG, G_SETTILE, G_SETTILESIZE, G_LOADBLOCK, G_LOADTILE, G_LOADTLUT, G_SETFILLCOLOR
            int8_t sop = (int8_t)op;
            if (rdpHandlers.contains(sop)) {
                F3DGfx* c = cmd;
                rdpHandlers.at(sop).second(&c);
            }
        }
        cmd++;
    }
    *mRsp = savedRsp;
    *mRdp = savedRdp;
    memcpy(mSegmentPointers, savedSegments, sizeof(savedSegments));
    sTmemRegions = std::move(savedTmem);

    // pairing: not moving first, then the most common moves
    constexpr int32_t kMaxMove = 48 * 4; // (10.2 fixed point)
    struct Side {
        std::vector<std::pair<int32_t, int32_t>> pos;
        std::vector<bool> used;
    };
    std::unordered_map<Interp2DKey, Side, Interp2DKeyHash> curSide, lastSide;
    std::map<std::pair<int32_t, int32_t>, int> votes;
    std::set<std::pair<int32_t, int32_t>> uniqueMoves;
    for (auto& [key, cv] : cur) {
        auto lit = sRectLast.find(key);
        if (lit == sRectLast.end()) continue;
        Side& c = curSide[key];
        Side& l = lastSide[key];
        c.pos = cv;
        c.used.assign(cv.size(), false);
        l.pos = lit->second;
        l.used.assign(l.pos.size(), false);
        sRectMove[key].assign(cv.size(), { 0, 0 });
        for (size_t a = 0; a < c.pos.size(); a++) {
            for (size_t b = 0; b < l.pos.size(); b++) {
                if (!l.used[b] && l.pos[b] == c.pos[a]) {
                    c.used[a] = l.used[b] = true;
                    break;
                }
            }
        }
        if (c.pos.size() == 1 && l.pos.size() == 1 && !c.used[0]) {
            const int32_t dx = c.pos[0].first - l.pos[0].first, dy = c.pos[0].second - l.pos[0].second;
            if (std::abs(dx) <= kMaxMove && std::abs(dy) <= kMaxMove) uniqueMoves.insert({ dx, dy });
        }
        if (c.pos.size() > 64 || l.pos.size() > 64) continue;
        for (size_t a = 0; a < c.pos.size(); a++) {
            if (c.used[a]) continue;
            for (size_t b = 0; b < l.pos.size(); b++) {
                if (l.used[b]) continue;
                const int32_t dx = c.pos[a].first - l.pos[b].first, dy = c.pos[a].second - l.pos[b].second;
                if (std::abs(dx) <= kMaxMove && std::abs(dy) <= kMaxMove) votes[{ dx, dy }]++;
            }
        }
    }
    std::vector<std::pair<int, std::pair<int32_t, int32_t>>> order;
    for (auto& [d, count] : votes) {
        if (count >= 4 || uniqueMoves.count(d)) order.push_back({ uniqueMoves.count(d) ? count + 1000000 : count, d });
    }
    std::sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    for (auto& [count, d] : order) {
        for (auto& [key, c] : curSide) {
            Side& l = lastSide[key];
            auto& moves = sRectMove[key];
            for (size_t a = 0; a < c.pos.size(); a++) {
                if (c.used[a]) continue;
                for (size_t b = 0; b < l.pos.size(); b++) {
                    if (!l.used[b] && c.pos[a].first - l.pos[b].first == d.first &&
                        c.pos[a].second - l.pos[b].second == d.second) {
                        c.used[a] = l.used[b] = true;
                        moves[a] = d;
                        break;
                    }
                }
            }
        }
    }
    // tear guard: a moving rectangle that overlaps a new one (not in the last drawing, e.g. the clipped edge of a
    // panel sliding in from off screen, or text being typed) stays put, and so does what overlaps it in turn: a panel
    // moves as a whole or not at all
    struct Box {
        int32_t x0, y0, x1, y1;
        bool moving, blocked;
        std::pair<int32_t, int32_t>* move;
    };
    std::vector<Box> boxes;
    for (auto& [key, cv] : cur) {
        auto sit = curSide.find(key);
        auto mit = sRectMove.find(key);
        for (size_t a = 0; a < cv.size(); a++) {
            const bool matched = sit != curSide.end() && sit->second.used[a];
            std::pair<int32_t, int32_t>* mv = mit != sRectMove.end() ? &mit->second[a] : nullptr;
            const bool moving = mv != nullptr && (mv->first != 0 || mv->second != 0);
            boxes.push_back({ cv[a].first, cv[a].second, cv[a].first + key.w, cv[a].second + key.h, moving, !matched, mv });
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& m : boxes) {
            if (!m.moving) continue;
            for (const auto& o : boxes) {
                if (o.blocked && &o != &m && o.x0 <= m.x1 && m.x0 <= o.x1 && o.y0 <= m.y1 && m.y0 <= o.y1) { // (touching too)
                    *m.move = { 0, 0 };
                    m.moving = false;
                    m.blocked = true;
                    changed = true;
                    break;
                }
            }
        }
    }
}

bool Interpreter::CanRestorePicture() const {
    static const bool sOff = getenv("PORT_INTERP_INCREMENTAL") != nullptr && atoi(getenv("PORT_INTERP_INCREMENTAL")) == 0;
    return !sOff && mRendersToFb;
}

void Interpreter::ReplayBegin(bool restorePicture) {
    sReplayRestore = restorePicture;
    if (sReplay == nullptr) sReplay = new ReplayState();
    ReplayState& r = *sReplay;
    r.rspStart = *mRsp;
    r.rdpStart = *mRdp;
    memcpy(r.segStart, mSegmentPointers, sizeof(r.segStart));
    r.nativeStart = mNativeDimensions;
    r.tmemStart = CopyTmem(sTmemRegions, r.rdpStart);
    r.begun = true;
    r.haveEnd = false;
}

void Interpreter::ReplayPassStart() {
    if (sReplay == nullptr || !sReplay->begun) return;
    ReplayState& r = *sReplay;
    if (!r.haveEnd) { // the state the first drawing left (its TMEM list kept as it is: the RDP points into it)
        r.rspEnd = *mRsp;
        r.rdpEnd = *mRdp;
        memcpy(r.segEnd, mSegmentPointers, sizeof(r.segEnd));
        r.nativeEnd = mNativeDimensions;
        r.tmemEnd = std::move(sTmemRegions);
        r.haveEnd = true;
    }
    *mRsp = r.rspStart;
    *mRdp = r.rdpStart;
    memcpy(mSegmentPointers, r.segStart, sizeof(r.segStart));
    mNativeDimensions = r.nativeStart;
    sTmemRegions = CopyTmem(r.tmemStart, *mRdp);
}

void Interpreter::ReplayEnd() {
    if (sReplay == nullptr || !sReplay->begun) return;
    ReplayState& r = *sReplay;
    if (r.haveEnd) {
        *mRsp = r.rspEnd;
        *mRdp = r.rdpEnd;
        memcpy(mSegmentPointers, r.segEnd, sizeof(r.segEnd));
        mNativeDimensions = r.nativeEnd;
        sTmemRegions = std::move(r.tmemEnd);
        r.tmemEnd.clear();
    }
    r.tmemStart.clear();
    r.begun = false;
    r.haveEnd = false;
    sReplayRestore = false;
}

bool Interpreter::FrameRepaintsScreen(Gfx* commands) {
    // Same walk as TrackSegmentsOnly, read-only: the segment table is restored afterwards.
    uintptr_t savedSegments[MAX_SEGMENT_POINTERS];
    memcpy(savedSegments, mSegmentPointers, sizeof(savedSegments));
    std::vector<F3DGfx*> stack;
    F3DGfx* cmd = (F3DGfx*)commands;
    uintptr_t cimg = 0, zimg = 0;
    uint32_t cimgWidth = 320;
    bool offscreen = false, fullFill = false;
    uint32_t otherModeH = 0; // cycle type: a fill rectangle in 1/2-cycle mode can be translucent (a fade)
    for (int budget = 0; cmd != nullptr && budget < 400000; budget++) {
        if ((uintptr_t)cmd < 0x100000000ull) {
            break;
        }
        uint32_t w0 = (uint32_t)cmd->words.w0;
        uintptr_t w1 = cmd->words.w1;
        uint8_t op = (uint8_t)(w0 >> 24);
        if (op == 0xDF /* G_ENDDL */ || (op > 0x07 && op < 0xD3)) {
            if (stack.empty()) {
                break;
            }
            cmd = stack.back();
            stack.pop_back();
            continue;
        }
        if (op == 0xDE /* G_DL */) {
            F3DGfx* sub = (F3DGfx*)SegAddr(w1);
            if (((w0 >> 16) & 0xFF) == 0) {
                if (sub != nullptr && stack.size() < 64) {
                    stack.push_back(cmd + 1);
                    cmd = sub;
                    continue;
                }
            } else {
                if (sub == nullptr) {
                    if (stack.empty()) {
                        break;
                    }
                    cmd = stack.back();
                    stack.pop_back();
                } else {
                    cmd = sub;
                }
                continue;
            }
        } else if (op == 0xDB /* G_MOVEWORD */ && ((w0 >> 16) & 0xFF) == G_MW_SEGMENT) {
            int seg = (int)((w0 & 0xFFFF) / 4);
            if (seg >= 0 && seg < MAX_SEGMENT_POINTERS) {
                mSegmentPointers[seg] = w1;
            }
        } else if (op == 0xFF /* G_SETCIMG */) {
            uint32_t width = (w0 & 0xFFF) + 1;
            cimg = (uintptr_t)SegAddr(w1);
            offscreen = (width != 320 && width != 640 && w1 != 0) || IsUnshownScreenImage(width, (const void*)cimg);
            cimgWidth = width;
        } else if (op == 0xFE /* G_SETZIMG */) {
            zimg = (uintptr_t)SegAddr(w1);
        } else if (op == 0xE3 /* G_SETOTHERMODE_H */) {
            uint32_t len = (w0 & 0xFF) + 1, shift = 32 - ((w0 >> 8) & 0xFF) - len;
            uint32_t mask = (len >= 32 ? 0xFFFFFFFFu : ((1u << len) - 1)) << shift;
            otherModeH = (otherModeH & ~mask) | ((uint32_t)w1 & mask);
        } else if (op == 0xEF /* G_RDPSETOTHERMODE */) {
            otherModeH = w0 & 0xFFFFFF;
        } else if (op == 0xF6 /* G_FILLRECT */ && !offscreen && ((otherModeH >> G_MDSFT_CYCLETYPE) & 3) == 3) {
            uint32_t lrx = (w0 >> 14) & 0x3FF, lry = (w0 >> 2) & 0x3FF;
            uint32_t ulx = ((uint32_t)w1 >> 14) & 0x3FF, uly = ((uint32_t)w1 >> 2) & 0x3FF;
            uint32_t h = cimgWidth > 400 ? 480 : 240;
            if (cimg != zimg && ulx == 0 && uly == 0 && lrx + 1 >= cimgWidth && lry + 1 >= h) {
                fullFill = true;
            }
        }
        cmd++;
    }
    memcpy(mSegmentPointers, savedSegments, sizeof(savedSegments));
    return fullFill;
}

void Interpreter::Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements) {
    const bool traceTiming = TraceRenderTiming();
    const auto runStart = traceTiming ? RenderClock::now() : RenderClock::time_point{};
    static RenderClock::time_point lastRunStart{};
    const double gapMs = traceTiming && lastRunStart.time_since_epoch().count() != 0
                             ? std::chrono::duration<double, std::milli>(runStart - lastRunStart).count()
                             : 0;
    if (traceTiming) {
        lastRunStart = runStart;
        sTriangleCount = 0;
        sUploadTiming = {};
        sDrawTiming = {};
        sShaderTiming = {};
    }
    sMarginsClearedThisRun = false;
    // (the margin clear's edge band, GfxDpSetColorImage: only on frames that repaint the whole screen)
    sBandClearThisRun = mCurDimensions.height > 0 && mCurDimensions.width * 3 > mCurDimensions.height * 4 * 1.01f &&
                        FrameRepaintsScreen(commands);
    s2DRectArea = 0.0f;
    sWideRects = false;
    sAnchor = 0;
    sSnapDoneThisRun.clear();
    // 2D in-between frames: the last exact drawing's rectangles become the reference; an exact drawing records anew
    if (sRectLastRunExact) {
        sRectLast.swap(sRectRec);
    }
    sRectRec.clear();
    sRectSeen.clear();
    sRectLastRunExact = mReplayT >= 0.999f;
    sPreserveNextColorImage = false;
    sSeen3DThisRun = false;
    sLock43ThisRun = false;
    sFilledWideThisRun = false;
    if (PortTile1Log()) fprintf(stderr, "[tile1] -- run begin\n");
    SpReset();

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mCurMtxReplacements = &mtx_replacements;

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    WarmupStep();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    // PORT: the N64 never clears its frame buffer by itself; games clear it with a fill rectangle when they need to.
    // Keep the colour (menus draw their static background only in the first frames of a screen).
    static const bool sClearEach = getenv("PORT_CLEAR_EACH_FRAME") != nullptr; // debug: old behaviour
    mRapi->ClearFramebuffer(!mRendersToFb || sClearEach, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    const auto listStart = traceTiming ? RenderClock::now() : RenderClock::time_point{};
    auto dbg = mGfxDebugger;
    g_exec_stack.start((F3DGfx*)commands);
    while (!g_exec_stack.cmd_stack.empty()) {
        auto cmd = g_exec_stack.cmd_stack.top();
        if (cmd == nullptr) {
            // PORT PATCH: a branch to NULL left an empty frame; drop it instead of dereferencing.
            g_exec_stack.cmd_stack.pop();
            continue;
        }

        if ((uintptr_t)cmd < 0x100000000ull) {
            // PORT PATCH: a list pointer that is not a host address (unresolved N64 address); drop the frame.
            static int shown = 0;
            if (shown++ < 5) {
                fprintf(stderr, "[badcmd] cmd=%p depth=%zu prev op=%08x\n", (void*)cmd, g_exec_stack.cmd_stack.size(), (unsigned)(sLastCmd ? sLastCmd->words.w0 >> 32 : 0));
                if (sLastCmd) fprintf(stderr, "[badcmd] last cmd %p w0=%lx w1=%lx\n", (void*)sLastCmd, (unsigned long)sLastCmd->words.w0, (unsigned long)sLastCmd->words.w1);
            }
            g_exec_stack.cmd_stack.pop();
            continue;
        }
        sLastCmd = cmd;
        if (dbg->IsDebugging()) {
            g_exec_stack.gfx_path.push_back(cmd);
            if (dbg->HasBreakPoint(g_exec_stack.gfx_path)) {
                // On a breakpoint with the active framebuffer still set, we need to reset back to prevent
                // soft locking the renderer
                if (mFbActive) {
                    mFbActive = 0;
                    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, 1);
                }

                break;
            }
            g_exec_stack.gfx_path.pop_back();
        }
        if (sOpLog) sOpHist[(uint8_t)(cmd->words.w0 >> 24)]++;
        gfx_step();
    }
    if (sOpLog && (++sOpFrames % 20) == 0) {
        fprintf(stderr, "[ops] frame %d:", sOpFrames);
        for (int k = 0; k < 256; k++) if (sOpHist[k]) fprintf(stderr, " %02x=%d", k, sOpHist[k]);
        fputc(10, stderr);
        memset(sOpHist, 0, sizeof sOpHist);
    }

    Flush();
    if (traceTiming) {
        const auto setupMs = std::chrono::duration<double, std::milli>(listStart - runStart).count();
        const auto listMs = std::chrono::duration<double, std::milli>(RenderClock::now() - listStart).count();
        if (TraceRenderFrames()) {
            fprintf(stderr, "[render] frame %llu setup=%.2f list=%.2f upload=%.2f/%u draw=%.2f/%u shader=%.2f/%u\n",
                    (unsigned long long)Port_ViTick(), setupMs, listMs, sUploadTiming.ms, sUploadTiming.calls,
                    sDrawTiming.ms, sDrawTiming.calls, sShaderTiming.ms, sShaderTiming.calls);
        }
    }
    mGfxFrameBuffer = 0;
    mGfxFrameBufferFb = -1;
    currentDir = std::stack<std::string>();

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution()) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
                mGfxFrameBufferFb = mGameFbMsaaResolved;
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
            mGfxFrameBufferFb = mGameFb;
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
    if (TraceRenderSpikes()) {
        const double renderMs = std::chrono::duration<double, std::milli>(RenderClock::now() - runStart).count();
        if (gapMs > RenderSpikeLimit() || renderMs > RenderSpikeLimit()) {
            fprintf(stderr, "[render-spike] frame %llu gap=%.2f render=%.2f upload=%.2f/%u draw=%.2f/%u "
                            "shader=%.2f/%u triangles=%zu\n",
                    (unsigned long long)Port_ViTick(), gapMs, renderMs, sUploadTiming.ms, sUploadTiming.calls,
                    sDrawTiming.ms, sDrawTiming.calls, sShaderTiming.ms, sShaderTiming.calls, sTriangleCount);
        }
    }

}

void Interpreter::EndFrame() {
    const bool traceTiming = TraceRenderTiming();
    const auto start = traceTiming ? RenderClock::now() : RenderClock::time_point{};
    gPortReplayFrame = mReplayPass && !mReplayReadback;
    mRapi->EndFrame();
    const auto submitEnd = traceTiming ? RenderClock::now() : RenderClock::time_point{};
    mWapi->SwapBuffersBegin();
    mRapi->FinishRender();
    const auto presentEnd = traceTiming ? RenderClock::now() : RenderClock::time_point{};
    mWapi->SwapBuffersEnd();
    if (traceTiming) {
        const auto submitMs = std::chrono::duration<double, std::milli>(submitEnd - start).count();
        const auto presentMs = std::chrono::duration<double, std::milli>(RenderClock::now() - submitEnd).count();
        const auto waitMs = std::chrono::duration<double, std::milli>(RenderClock::now() - presentEnd).count();
        if (TraceRenderFrames()) {
            fprintf(stderr, "[render-end] frame %llu submit=%.2f present=%.2f swap_wait=%.2f\n",
                    (unsigned long long)Port_ViTick(), submitMs, presentMs, waitMs);
        }
        if (TraceRenderSpikes() && submitMs + presentMs > RenderSpikeLimit()) {
            fprintf(stderr, "[render-present-spike] frame %llu submit=%.2f present=%.2f swap_wait=%.2f\n",
                    (unsigned long long)Port_ViTick(), submitMs, presentMs, waitMs);
        }
    }
}

void gfx_set_target_ucode(UcodeHandlers ucode) {
    ucode_handler_index = ucode;
}

int Interpreter::GetTargetFps() {
    return mWapi->GetTargetFps();
}

void Interpreter::SetTargetFps(int fps) {
    mWapi->SetTargetFps(fps);
}

void Interpreter::SetMaxFrameLatency(int latency) {
    mWapi->SetMaxFrameLatency(latency);
}

int Interpreter::CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                   uint8_t resize, bool forceFixedAspect) {
    uint32_t orig_width = width, orig_height = height;
    if (resize) {
        AdjustWidthHeightForScale(width, height, native_width, native_height);
    }

    int fb = mRapi->CreateFramebuffer();
    mRapi->UpdateFramebufferParameters(fb, width, height, 1, true, true, true, true);

    mFrameBuffers[fb] = {
        orig_width, orig_height, width, height, native_width, native_height, static_cast<bool>(resize), forceFixedAspect
    };
    return fb;
}

void Interpreter::SetFrameBuffer(int fb, float noiseScale) {
    mRapi->StartDrawToFramebuffer(fb, noiseScale);
    mRapi->ClearFramebuffer(false, true);
}

void Interpreter::CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr) {
    // Do not copy again if we have already copied before
    if (copyOnce && hasCopiedPtr != nullptr && *hasCopiedPtr) {
        return;
    }

    if (fb_src_id == 0 && mRendersToFb) {
        // read from the framebuffer we've been rendering to
        fb_src_id = mGameFb;
    }

    int srcX0, srcY0, srcX1, srcY1;
    int dstX0, dstY0, dstX1, dstY1;

    // When rendering to the main window buffer or MSAA is enabled with a buffer size equal to the view port,
    // then the source coordinates must account for any docked ImGui elements
    if (fb_src_id == 0 || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                           mCurDimensions.height == mGameWindowViewport.height)) {
        srcX0 = mGameWindowViewport.x;
        srcY0 = mGameWindowViewport.y;
        srcX1 = mGameWindowViewport.x + mGameWindowViewport.width;
        srcY1 = mGameWindowViewport.y + mGameWindowViewport.height;
    } else {
        srcX0 = 0;
        srcY0 = 0;
        srcX1 = mCurDimensions.width;
        srcY1 = mCurDimensions.height;
    }

    dstX0 = 0;
    dstY0 = 0;
    dstX1 = mCurDimensions.width;
    dstY1 = mCurDimensions.height;

    mRapi->CopyFramebuffer(fb_dst_id, fb_src_id, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);

    // Set the copied pointer if we have one
    if (hasCopiedPtr != nullptr) {
        *hasCopiedPtr = true;
    }
}

void Interpreter::ResetFrameBuffer() {
    mRapi->StartDrawToFramebuffer(0, (float)mCurDimensions.height / mNativeDimensions.height);
}

void Interpreter::AdjustPixelDepthCoordinates(float& x, float& y) {
    x = x * RATIO_X(mActiveFrameBuffer, mCurDimensions) -
        (mNativeDimensions.width * RATIO_X(mActiveFrameBuffer, mCurDimensions) - mCurDimensions.width) / 2;
    y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
    if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                          mCurDimensions.height == mGameWindowViewport.height)) {
        x += mGameWindowViewport.x;
        y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
    }
}

void Interpreter::GetPixelDepthPrepare(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);
    mGetPixelDepthPending.emplace(x, y);
}

uint16_t Interpreter::GetPixelDepth(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);

    if (auto it = mGetPixelDepthCached.find(std::make_pair(x, y)); it != mGetPixelDepthCached.end()) {
        return it->second;
    }

    mGetPixelDepthPending.emplace(x, y);

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res =
        mRapi->GetPixelDepth(mRendersToFb ? mGameFb : 0, mGetPixelDepthPending);
    mGetPixelDepthCached.merge(res);
    mGetPixelDepthPending.clear();

    return mGetPixelDepthCached.find(std::make_pair(x, y))->second;
}

void gfx_push_current_dir(char* path) {
    if (gfx_check_image_signature(path) == 1)
        path = &path[7];

    currentDir.push(GetPathWithoutFileName(path));
}

int32_t gfx_check_image_signature(const char* imgData) {
    uintptr_t i = (uintptr_t)(imgData);

    if ((i & 1) == 1) {
        return 0;
    }

    // Filter addresses that are obviously not valid string pointers before
    // attempting to dereference for the "__OTR__" check.
    if (i == 0 || i < 0x10000) {
        return 0;
    }
#if UINTPTR_MAX > 0xFFFFFFFFu
    // On 64-bit: filter kernel/sentinel addresses. Upper bound covers all
    // user-space layouts (x86_64 47-bit canonical, ARM64 48-bit VA, etc.).
    if (i > 0x0000FFFFFFFFFFFFull) {
        return 0;
    }
#endif

    return sResourceManager->OtrSignatureCheck(imgData);
}

void Interpreter::RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    if (gfx_check_image_signature(reinterpret_cast<char*>(replacement))) {
        Fast::Texture* tex = std::static_pointer_cast<Fast::Texture>(
                                 mResourceManager->LoadResourceProcess(reinterpret_cast<char*>(replacement)))
                                 .get();

        replacement = tex->ImageData;
    }

    mMaskedTextures[name] = MaskedTextureEntry{ mask, replacement };
}

void Interpreter::UnregisterBlendedTexture(const char* name) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    mMaskedTextures.erase(name);
}

// New getters and setters
void Interpreter::SetNativeDimensions(float width, float height) {
    mNativeDimensions.width = width;
    mNativeDimensions.height = height;
}

void Interpreter::SetResolutionMultiplier(float multiplier) {
    mCurDimensions.internal_mul = multiplier;
}

void Interpreter::SetMsaaLevel(uint32_t level) {
    mMsaaLevel = level;
}

void Interpreter::GetCurDimensions(uint32_t* width, uint32_t* height) {
    *width = mCurDimensions.width;
    *height = mCurDimensions.height;
}

} // namespace Fast

void gfx_cc_get_features(uint64_t shader_id0, uint64_t shader_id1, struct CCFeatures* cc_features) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                cc_features->c[i][j][k] = shader_id0 >> i * 32 + j * 16 + k * 4 & 0xf;
            }
        }
    }

    cc_features->opt_alpha = (shader_id1 & SHADER_OPT(ALPHA)) != 0;
    cc_features->opt_fog = (shader_id1 & SHADER_OPT(FOG)) != 0;
    cc_features->opt_texture_edge = (shader_id1 & SHADER_OPT(TEXTURE_EDGE)) != 0;
    cc_features->opt_noise = (shader_id1 & SHADER_OPT(NOISE)) != 0;
    cc_features->opt_2cyc = (shader_id1 & SHADER_OPT(_2CYC)) != 0;
    cc_features->opt_alpha_threshold = (shader_id1 & SHADER_OPT(ALPHA_THRESHOLD)) != 0;
    cc_features->opt_invisible = (shader_id1 & SHADER_OPT(INVISIBLE)) != 0;
    cc_features->opt_grayscale = (shader_id1 & SHADER_OPT(GRAYSCALE)) != 0;
    cc_features->opt_prim_depth = (shader_id1 & SHADER_OPT(PRIM_DEPTH)) != 0;

    cc_features->clamp[0][0] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_S);
    cc_features->clamp[0][1] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_T);
    cc_features->clamp[1][0] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_S);
    cc_features->clamp[1][1] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_T);

    cc_features->usedTextures[0] = false;
    cc_features->usedTextures[1] = false;
    cc_features->used_masks[0] = false;
    cc_features->used_masks[1] = false;
    cc_features->used_blend[0] = false;
    cc_features->used_blend[1] = false;
    cc_features->numInputs = 0;

    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                if (cc_features->c[c][i][j] >= SHADER_INPUT_1 && cc_features->c[c][i][j] <= SHADER_INPUT_7) {
                    if (cc_features->c[c][i][j] > cc_features->numInputs) {
                        cc_features->numInputs = cc_features->c[c][i][j];
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL0 || cc_features->c[c][i][j] == SHADER_TEXEL0A) {
                    cc_features->usedTextures[0] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[1] = true;
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL1 || cc_features->c[c][i][j] == SHADER_TEXEL1A) {
                    cc_features->usedTextures[1] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[0] = true;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 2; c++) {
        cc_features->do_single[c][0] = cc_features->c[c][0][2] == SHADER_0;
        cc_features->do_single[c][1] = cc_features->c[c][1][2] == SHADER_0;
        cc_features->do_multiply[c][0] = cc_features->c[c][0][1] == SHADER_0 && cc_features->c[c][0][3] == SHADER_0;
        cc_features->do_multiply[c][1] = cc_features->c[c][1][1] == SHADER_0 && cc_features->c[c][1][3] == SHADER_0;
        cc_features->do_mix[c][0] = cc_features->c[c][0][1] == cc_features->c[c][0][3];
        cc_features->do_mix[c][1] = cc_features->c[c][1][1] == cc_features->c[c][1][3];
        cc_features->color_alpha_same[c] = (shader_id0 >> c * 32 & 0xffff) == (shader_id0 >> c * 32 + 16 & 0xffff);
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_MASK)) {
        cc_features->used_masks[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_MASK)) {
        cc_features->used_masks[1] = true;
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_BLEND)) {
        cc_features->used_blend[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_BLEND)) {
        cc_features->used_blend[1] = true;
    }

    cc_features->shader_id = Fast::ShaderIdUnmask(shader_id1);
}

extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize, bool forceFixedAspect) {
    return Fast::mInstance.lock().get()->CreateFrameBuffer(width, height, native_width, native_height, resize,
                                                           forceFixedAspect);
}

extern "C" void gfx_texture_cache_clear() {
    Fast::mInstance.lock().get()->TextureCacheClear();
}

extern "C" void gfx_shader_cache_clear() {
    auto instance = Fast::mInstance.lock().get();
    instance->mColorCombinerPool.clear();
    instance->mPrevCombiner = Fast::mInstance.lock().get()->mColorCombinerPool.end();
    instance->mRenderingState.mShaderProgram = nullptr;
    instance->mRapi->ClearShaderCache();
}

extern "C" void gfx_register_fb_texture(const void* cpuAddr, int fbId) {
    Fast::mInstance.lock().get()->RegisterFbTexture(cpuAddr, fbId);
}

extern "C" void gfx_unregister_fb_texture(const void* cpuAddr) {
    Fast::mInstance.lock().get()->UnregisterFbTexture(cpuAddr);
}
