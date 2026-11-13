#include "fast/Fast3dWindow.h"

#include "ship/core/Context.h"
#include "ship/config/Config.h"
#include "ship/controller/controldeck/ControlDeck.h"
#include "ship/config/ConsoleVariable.h"
#include "ship/window/FileDrop.h"
#include "ship/resource/ResourceManager.h"
#include "fast/interpreter.h"
#include "fast/backends/gfx_sdl.h"
#include "fast/backends/gfx_dxgi.h"
#include "fast/backends/gfx_opengl.h"
#include "fast/backends/gfx_metal.h"
#include "fast/backends/gfx_vulkan.h"
#include "fast/backends/gfx_direct3d_common.h"
#include "fast/backends/gfx_direct3d11.h"
#include "fast/backends/gfx_window_manager_api.h"

#include "fast/Fast3dGui.h"
#include "libultraship/window/gui/GfxDebuggerWindow.h"

#include <fstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace Fast {

extern void GfxSetInstance(std::shared_ptr<Interpreter> gfx);

Fast3dWindow::Fast3dWindow(std::shared_ptr<Ship::Gui> gui, std::shared_ptr<FastMouseStateManager> mouseStateManager,
                           std::shared_ptr<Ship::Config> config,
                           std::shared_ptr<Ship::ConsoleVariable> consoleVariables,
                           std::shared_ptr<Ship::ControlDeck> controlDeck)
    : Ship::Window(gui, mouseStateManager, std::move(config)), mConsoleVariables(std::move(consoleVariables)),
      mControlDeck(std::move(controlDeck)) {
    mWindowManagerApi = nullptr;
    mRenderingApi = nullptr;
    mInterpreter = std::make_shared<Interpreter>();
    GfxSetInstance(mInterpreter);

#ifdef _WIN32
    AddAvailableWindowBackend(WindowBackend::FAST3D_DXGI_DX11);
#endif
#ifdef __APPLE__
    if (Metal_IsSupported()) {
        AddAvailableWindowBackend(WindowBackend::FAST3D_SDL_METAL);
    }
#endif
    AddAvailableWindowBackend(WindowBackend::FAST3D_SDL_OPENGL);
}

Fast3dWindow::Fast3dWindow(std::shared_ptr<Ship::Gui> gui, std::shared_ptr<Ship::Config> config,
                           std::shared_ptr<Ship::ConsoleVariable> consoleVariables,
                           std::shared_ptr<Ship::ControlDeck> controlDeck)
    : Fast3dWindow(gui, std::make_shared<FastMouseStateManager>(), std::move(config), std::move(consoleVariables),
                   std::move(controlDeck)) {
}

Fast3dWindow::Fast3dWindow(std::vector<std::shared_ptr<Ship::GuiWindow>> guiWindows,
                           std::shared_ptr<Ship::Config> config,
                           std::shared_ptr<Ship::ConsoleVariable> consoleVariables,
                           std::shared_ptr<Ship::ControlDeck> controlDeck)
    : Fast3dWindow(std::make_shared<Fast3dGui>(guiWindows), std::move(config), std::move(consoleVariables),
                   std::move(controlDeck)) {
}

Fast3dWindow::Fast3dWindow(std::shared_ptr<Ship::Config> config,
                           std::shared_ptr<Ship::ConsoleVariable> consoleVariables,
                           std::shared_ptr<Ship::ControlDeck> controlDeck)
    : Fast3dWindow(std::vector<std::shared_ptr<Ship::GuiWindow>>(), std::move(config), std::move(consoleVariables),
                   std::move(controlDeck)) {
}

Fast3dWindow::~Fast3dWindow() {
    SPDLOG_DEBUG("destruct fast3dwindow");
    mInterpreter->Destroy();
    delete mRenderingApi;
    delete mWindowManagerApi;
}

void Fast3dWindow::OnInit(const nlohmann::json& initArgs) {
    Window::OnInit(initArgs);
    auto self = std::dynamic_pointer_cast<Fast3dWindow>(GetSharedComponent());
    mInterpreter->SetFast3dWindow(self);
    bool gameMode = false;

#ifdef __linux__
    std::ifstream osReleaseFile("/etc/os-release");
    if (osReleaseFile.is_open()) {
        std::string line;
        while (std::getline(osReleaseFile, line)) {
            if (line.find("VARIANT_ID") != std::string::npos) {
                if (line.find("steamdeck") != std::string::npos) {
                    gameMode = std::getenv("XDG_CURRENT_DESKTOP") != nullptr &&
                               std::string(std::getenv("XDG_CURRENT_DESKTOP")) == "gamescope";
                }
                break;
            }
        }
    }
#elif defined(__ANDROID__) || defined(__IOS__)
    gameMode = true;
#endif

    bool isFullscreen;
    uint32_t width, height;
    int32_t posX, posY;

    isFullscreen = GetConfig()->GetBool("Window.Fullscreen.Enabled", false) || gameMode;
    posX = GetConfig()->GetInt("Window.PositionX", 100);
    posY = GetConfig()->GetInt("Window.PositionY", 100);

    if (isFullscreen) {
        width = GetConfig()->GetInt("Window.Fullscreen.Width", gameMode ? 1280 : 1920);
        height = GetConfig()->GetInt("Window.Fullscreen.Height", gameMode ? 800 : 1080);
    } else {
        width = GetConfig()->GetInt("Window.Width", 640);
        height = GetConfig()->GetInt("Window.Height", 480);
    }
    SetFullscreenScancode(GetConfig()->GetInt("Shortcuts.Fullscreen", Ship::KbScancode::LUS_KB_F11));
    SetMouseCaptureScancode(GetConfig()->GetInt("Shortcuts.MouseCapture", Ship::KbScancode::LUS_KB_F2));

    InitWindowManager();
    mGfxDebugger = std::make_shared<GfxDebugger>();
    mInterpreter->SetGfxDebugger(mGfxDebugger);
    const std::string& windowTitle = GetContext() != nullptr ? GetContext()->GetShortName() : GetName();
    mInterpreter->Init(mWindowManagerApi, mRenderingApi, windowTitle.c_str(), isFullscreen, width, height, posX, posY,
                       GetConsoleVariables(), GetContext()->GetChildren().GetFirst<Ship::ResourceManager>());
    mWindowManagerApi->SetFullscreenChangedCallback(OnFullscreenChanged);
    mWindowManagerApi->SetKeyboardCallbacks(KeyDown, KeyUp, AllKeysUp);
    mWindowManagerApi->SetMouseCallbacks(MouseButtonDown, MouseButtonUp);

    SetTextureFilter((FilteringMode)GetConsoleVariables()->GetInteger(CVAR_TEXTURE_FILTER, FILTER_THREE_POINT));

    // Register the GFX Debugger window if not already present.
    if (GetGui() && GetGui()->GetGuiWindow("Gfx Debugger") == nullptr) {
        auto gfxDebugger = mGfxDebugger;
        auto resourceManager =
            RequireDependency(GetContext()->GetChildren().GetFirst<Ship::ResourceManager>(), "ResourceManager");
        GetGui()->AddGuiWindow(std::make_shared<LUS::GfxDebuggerWindow>(CVAR_GFX_DEBUGGER_WINDOW_OPEN, "Gfx Debugger",
                                                                        self, gfxDebugger, resourceManager));
    }
}

int32_t Fast3dWindow::GetTargetFps() {
    return mInterpreter->GetTargetFps();
}

void Fast3dWindow::SetTargetFps(int32_t fps) {
    mInterpreter->SetTargetFps(fps);
}

void Fast3dWindow::SetMaximumFrameLatency(int32_t latency) {
    mInterpreter->SetMaxFrameLatency(latency);
}

void Fast3dWindow::GetPixelDepthPrepare(float x, float y) {
    mInterpreter->GetPixelDepthPrepare(x, y);
}

uint16_t Fast3dWindow::GetPixelDepth(float x, float y) {
    return mInterpreter->GetPixelDepth(x, y);
}

void Fast3dWindow::InitWindowManager() {
#ifdef ENABLE_VULKAN
    // PORT: Vulkan is used only when the config asks for it (Window.Backend.Id 4) and a driver with a device is
    // installed; otherwise the default backend. Probed only then: the probe (SDL video + a Vulkan instance before the
    // window exists) stalled the OpenGL backend on macOS.
    if (GetConfig()->GetInt("Window.Backend.Id", -1) == WindowBackend::FAST3D_SDL_VULKAN && Vulkan_IsSupported()) {
        AddAvailableWindowBackend(WindowBackend::FAST3D_SDL_VULKAN);
    }
#endif
    SetWindowBackend(GetSavedWindowBackend());

    switch (GetWindowBackend()) {
#ifdef ENABLE_DX11
        case WindowBackend::FAST3D_DXGI_DX11:
            mWindowManagerApi =
                new GfxWindowBackendDXGI(GetConfig(), GetContext()->GetChildren().GetFirst<Ship::FileDrop>(),
                                         GetConsoleVariables(), std::dynamic_pointer_cast<Fast::Fast3dGui>(GetGui()));
            mRenderingApi =
                new GfxRenderingAPIDX11(static_cast<GfxWindowBackendDXGI*>(mWindowManagerApi), GetConsoleVariables(),
                                        GetContext()->GetChildren().GetFirst<Ship::ResourceManager>());
            break;
#endif
#ifdef ENABLE_OPENGL
        case WindowBackend::FAST3D_SDL_OPENGL:
            mWindowManagerApi =
                new GfxWindowBackendSDL(GetConfig(), GetContext()->GetChildren().GetFirst<Ship::FileDrop>(),
                                        GetConsoleVariables(), std::dynamic_pointer_cast<Fast::Fast3dGui>(GetGui()));
            mRenderingApi = new GfxRenderingAPIOGL(GetConsoleVariables(),
                                                   GetContext()->GetChildren().GetFirst<Ship::ResourceManager>());
            break;
#endif
#ifdef __APPLE__
        case WindowBackend::FAST3D_SDL_METAL:
            mWindowManagerApi =
                new GfxWindowBackendSDL(GetConfig(), GetContext()->GetChildren().GetFirst<Ship::FileDrop>(),
                                        GetConsoleVariables(), std::dynamic_pointer_cast<Fast::Fast3dGui>(GetGui()));
            mRenderingApi = new GfxRenderingAPIMetal(GetConsoleVariables(),
                                                     GetContext()->GetChildren().GetFirst<Ship::ResourceManager>());
            break;
#endif
#ifdef ENABLE_VULKAN
        case WindowBackend::FAST3D_SDL_VULKAN:
            mWindowManagerApi =
                new GfxWindowBackendSDL(GetConfig(), GetContext()->GetChildren().GetFirst<Ship::FileDrop>(),
                                        GetConsoleVariables(), std::dynamic_pointer_cast<Fast::Fast3dGui>(GetGui()));
            mRenderingApi = new GfxRenderingAPIVK(GetConsoleVariables(),
                                                  GetContext()->GetChildren().GetFirst<Ship::ResourceManager>());
            break;
#endif
        default:
            SPDLOG_ERROR("Could not load the correct rendering backend");
            break;
    }
}

void Fast3dWindow::SetTextureFilter(FilteringMode filteringMode) {
    mInterpreter->GetCurrentRenderingAPI()->SetTextureFilter(filteringMode);
}

void Fast3dWindow::EnableSRGBMode() {
    mInterpreter->mRapi->SetSrgbMode();
}

void Fast3dWindow::SetRendererUCode(UcodeHandlers ucode) {
    gfx_set_target_ucode(ucode);
}

void Fast3dWindow::Close() {
    mWindowManagerApi->Close();
}

void Fast3dWindow::RunGuiOnly() {
    mInterpreter->RunGuiOnly();
}

void Fast3dWindow::StartFrame() {
    mInterpreter->StartFrame();
}

void Fast3dWindow::EndFrame() {
    mInterpreter->EndFrame();
}

bool Fast3dWindow::IsFrameReady() {
    // PORT: never drop a frame. N64 frame buffers keep their picture and the game draws some things only once (menu
    // panels), so a frame dropped by the DXGI backend's wall-clock pacing lost them for good on a slow host.
    static const bool sAllowDrop = getenv("PORT_ALLOW_FRAME_DROP") != nullptr;
    return !sAllowDrop || mWindowManagerApi->IsFrameReady();
}

// PORT: PORT_TRACE_FRAMETIME=1 prints, every 2 s, what a drawn picture costs: the wait for a drawable (StartDraw:
// the display's swapchain), the interpreter (Run), the GUI and the game picture into the window (EndDraw), submit
// (EndFrame), and the GPU's busy time (command buffers' GPUStartTime..GPUEndTime, Metal only) per picture.
std::atomic<uint64_t> gPortGpuBusyNs{ 0 };
std::atomic<uint64_t> gPortGpuBuffers{ 0 };

// GPU time per command buffer label (PORT_TRACE_FRAMETIME=2: printed with the statistics)
static std::mutex sGpuLabelLock;
static std::map<std::string, double> sGpuByLabel;
void PortGpuByLabel(const char* label, double seconds) {
    std::lock_guard<std::mutex> g(sGpuLabelLock);
    sGpuByLabel[label] += seconds;
}

static bool TraceFrameTime() {
    static const bool on = getenv("PORT_TRACE_FRAMETIME") != nullptr;
    return on;
}

namespace {
struct FrameTimeStats {
    std::vector<float> seg[5]; // wait, run, gui, submit, cpu (run + gui + submit)
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    uint64_t gpuNs0 = 0, gpuBuf0 = 0;

    void Add(double wait, double run, double gui, double submit) {
        seg[0].push_back((float)wait);
        seg[1].push_back((float)run);
        seg[2].push_back((float)gui);
        seg[3].push_back((float)submit);
        seg[4].push_back((float)(run + gui + submit));
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - since).count();
        if (dt < 2.0) {
            return;
        }
        size_t n = seg[0].size();
        uint64_t gpuNs = gPortGpuBusyNs.load(), gpuBuf = gPortGpuBuffers.load();
        double gpuMs = (gpuNs - gpuNs0) / 1e6;
        char line[512];
        int len = snprintf(line, sizeof line, "[frametime] %5.1f pictures/s", n / dt);
        static const char* names[5] = { "wait", "run", "gui", "submit", "cpu" };
        for (int k = 0; k < 5; k++) {
            std::vector<float> v = seg[k];
            std::sort(v.begin(), v.end());
            double sum = 0;
            for (float x : v) sum += x;
            len += snprintf(line + len, sizeof line - len, "  %s %.2f/%.2f", names[k], sum / n,
                            v[std::min(n - 1, (size_t)(0.95 * (n - 1) + 0.5))]);
        }
        len += snprintf(line + len, sizeof line - len, "  gpu %.2f ms/picture (%.0f%% busy, %.1f cmdbufs)",
                        gpuMs / n, 100.0 * gpuMs / (dt * 1000), (double)(gpuBuf - gpuBuf0) / n);
        fprintf(stderr, "%s   (avg/p95 ms)\n", line);
        static const bool sByLabel = strcmp(getenv("PORT_TRACE_FRAMETIME"), "2") == 0;
        if (sByLabel) {
            std::lock_guard<std::mutex> g(sGpuLabelLock);
            for (auto& [name, sec] : sGpuByLabel) {
                fprintf(stderr, "[frametime]    gpu %-36s %.2f ms/picture\n", name.c_str(), sec * 1000 / n);
            }
            sGpuByLabel.clear();
        }
        for (auto& v : seg) v.clear();
        since = now;
        gpuNs0 = gpuNs;
        gpuBuf0 = gpuBuf;
    }
};
} // namespace

bool Fast3dWindow::DrawAndRunGraphicsCommands(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtxReplacements) {
    // Skip dropped frames
    if (!IsFrameReady()) {
        // PORT: a dropped frame still updates the persistent RSP/RDP state; one that renders to a texture is drawn
        if (!mInterpreter->TrackSegmentsOnly(commands)) {
            return false;
        }
    }

    using Clock = std::chrono::steady_clock;
    const bool trace = TraceFrameTime();
    const auto t0 = trace ? Clock::now() : Clock::time_point{};
    auto gui = GetGui();
    // Setup mouse state manager
    GetMouseStateManager()->StartFrame();
    // Setup of the backend frames and draw initial Window and GUI menus
    gui->StartDraw();
    const auto t1 = trace ? Clock::now() : Clock::time_point{};
    // Setup game framebuffers to match available window space
    mInterpreter->StartFrame();
    // Execute the games gfx commands
    mInterpreter->Run(commands, mtxReplacements);
    const auto t2 = trace ? Clock::now() : Clock::time_point{};
    // Renders the game frame buffer to the final window and finishes the GUI
    gui->EndDraw();
    const auto t3 = trace ? Clock::now() : Clock::time_point{};
    // Finalize swap buffers
    mInterpreter->EndFrame();
    if (trace) {
        static FrameTimeStats sStats;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        sStats.Add(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, Clock::now()));
    }

    return true;
}

void Fast3dWindow::HandleEvents() {
    mWindowManagerApi->HandleEvents();
}

void Fast3dWindow::SetCursorVisibility(bool visible) {
    mWindowManagerApi->SetCursorVisibility(visible);
}

uint32_t Fast3dWindow::GetWidth() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return width;
}

uint32_t Fast3dWindow::GetHeight() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return height;
}

float Fast3dWindow::GetAspectRatio() {
    return mInterpreter->mCurDimensions.aspect_ratio;
}

int32_t Fast3dWindow::GetPosX() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return posX;
}

int32_t Fast3dWindow::GetPosY() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return posY;
}

void Fast3dWindow::SetMousePos(Ship::Coords pos) {
    mWindowManagerApi->SetMousePos(pos.x, pos.y);
}

Ship::Coords Fast3dWindow::GetMousePos() {
    int32_t x, y;
    mWindowManagerApi->GetMousePos(&x, &y);
    return { x, y };
}

Ship::Coords Fast3dWindow::GetMouseDelta() {
    int32_t x, y;
    mWindowManagerApi->GetMouseDelta(&x, &y);
    return { x, y };
}

Ship::CoordsF Fast3dWindow::GetMouseWheel() {
    float x, y;
    mWindowManagerApi->GetMouseWheel(&x, &y);
    return { x, y };
}

bool Fast3dWindow::GetMouseState(Ship::MouseBtn btn) {
    return mWindowManagerApi->GetMouseState(static_cast<uint32_t>(btn));
}

void Fast3dWindow::SetMouseCapture(bool capture) {
    mWindowManagerApi->SetMouseCapture(capture);
}

bool Fast3dWindow::IsMouseCaptured() {
    return mWindowManagerApi->IsMouseCaptured();
}

uint32_t Fast3dWindow::GetCurrentRefreshRate() {
    uint32_t refreshRate;
    mWindowManagerApi->GetActiveWindowRefreshRate(&refreshRate);
    return refreshRate;
}

bool Fast3dWindow::SupportsWindowedFullscreen() {
#ifdef __APPLE__
    return false;
#endif

    if (GetWindowBackend() == WindowBackend::FAST3D_SDL_OPENGL) {
        return true;
    }

    return false;
}

bool Fast3dWindow::CanDisableVerticalSync() {
    return mWindowManagerApi->CanDisableVsync();
}

void Fast3dWindow::SetResolutionMultiplier(float multiplier) {
    mInterpreter->SetResolutionMultiplier(multiplier);
}

void Fast3dWindow::SetMsaaLevel(uint32_t value) {
    mInterpreter->SetMsaaLevel(value);
}

void Fast3dWindow::SetFullscreen(bool isFullscreen) {
    // Save current window position before fullscreening
    SaveWindowToConfig();
    mWindowManagerApi->SetFullscreen(isFullscreen);
}

bool Fast3dWindow::IsFullscreen() {
    return mWindowManagerApi->IsFullscreen();
}

bool Fast3dWindow::IsRunning() {
    return mWindowManagerApi->IsRunning();
}

uintptr_t Fast3dWindow::GetGfxFrameBuffer() {
    return mInterpreter->mGfxFrameBuffer;
}

const char* Fast3dWindow::GetKeyName(int32_t scancode) {
    return mWindowManagerApi->GetKeyName(scancode);
}

bool Fast3dWindow::KeyUp(int32_t scancode) {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return false;
    }

    if (scancode == wnd->GetFullscreenScancode()) {
        wnd->ToggleFullscreen();
    }

    if (scancode == wnd->GetMouseCaptureScancode()) {
        wnd->GetMouseStateManager()->ToggleMouseCaptureOverride();
    }

    wnd->SetLastScancode(-1);
    return wnd->GetControlDeck()->ProcessKeyboardEvent(Ship::KbEventType::LUS_KB_EVENT_KEY_UP,
                                                       static_cast<Ship::KbScancode>(scancode));
}

bool Fast3dWindow::KeyDown(int32_t scancode) {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return false;
    }

    bool isProcessed = wnd->GetControlDeck()->ProcessKeyboardEvent(Ship::KbEventType::LUS_KB_EVENT_KEY_DOWN,
                                                                   static_cast<Ship::KbScancode>(scancode));
    wnd->SetLastScancode(scancode);
    return isProcessed;
}

void Fast3dWindow::AllKeysUp() {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return;
    }

    wnd->GetControlDeck()->ProcessKeyboardEvent(Ship::KbEventType::LUS_KB_EVENT_ALL_KEYS_UP,
                                                Ship::KbScancode::LUS_KB_UNKNOWN);
}

bool Fast3dWindow::MouseButtonUp(int button) {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return false;
    }

    return wnd->GetControlDeck()->ProcessMouseButtonEvent(false, static_cast<Ship::MouseBtn>(button));
}

bool Fast3dWindow::MouseButtonDown(int button) {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return false;
    }

    return wnd->GetControlDeck()->ProcessMouseButtonEvent(true, static_cast<Ship::MouseBtn>(button));
}

void Fast3dWindow::OnFullscreenChanged(bool isNowFullscreen) {
    auto wnd = Interpreter::GetCurrentWindow();
    if (!wnd) {
        return;
    }

    wnd->GetConfig()->SetBool("Window.Fullscreen.Enabled", isNowFullscreen);
}

std::shared_ptr<GfxDebugger> Fast3dWindow::GetGfxDebugger() const {
    return mGfxDebugger;
}

std::string Fast3dWindow::GetWindowBackendName() {
    switch (GetWindowBackend()) {
#ifdef ENABLE_DX11
        case WindowBackend::FAST3D_DXGI_DX11:
            return "DirectX 11";
#endif
#ifdef ENABLE_OPENGL
        case WindowBackend::FAST3D_SDL_OPENGL:
            return "OpenGL";
#endif
#ifdef __APPLE__
        case WindowBackend::FAST3D_SDL_METAL:
            return "Metal";
#endif
#ifdef ENABLE_VULKAN
        case WindowBackend::FAST3D_SDL_VULKAN:
            return "Vulkan";
#endif
        default:
            return "";
    }
}

void Fast3dWindow::SetCurrentDimensions(uint32_t width, uint32_t height) {
    mWindowManagerApi->SetDimensions(width, height, GetPosX(), GetPosY());
}

void Fast3dWindow::SetCurrentDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) {
    mWindowManagerApi->SetDimensions(width, height, posX, posY);
}

void Fast3dWindow::SetCurrentDimensions(bool isFullscreen, uint32_t width, uint32_t height) {
    SetFullscreen(isFullscreen);
    mWindowManagerApi->SetDimensions(width, height, GetPosX(), GetPosY());
}

void Fast3dWindow::SetCurrentDimensions(bool isFullscreen, uint32_t width, uint32_t height, int32_t posX,
                                        int32_t posY) {
    SetFullscreen(isFullscreen);
    mWindowManagerApi->SetDimensions(width, height, posX, posY);
}

Ship::WindowRect Fast3dWindow::GetPrimaryMonitorRect() {
    return mWindowManagerApi->GetPrimaryMonitorRect();
}

std::weak_ptr<Interpreter> Fast3dWindow::GetInterpreterWeak() const {
    return mInterpreter;
}

std::shared_ptr<Ship::ConsoleVariable> Fast3dWindow::GetConsoleVariables() const {
    return RequireDependency(mConsoleVariables, "ConsoleVariables");
}

std::shared_ptr<Ship::ControlDeck> Fast3dWindow::GetControlDeck() const {
    if (mControlDeck == nullptr) {
        if (auto context = GetContext()) {
            mControlDeck = context->GetChildren().GetFirst<Ship::ControlDeck>();
        }
    }
    if (mControlDeck == nullptr) {
        throw std::runtime_error("Component 'Window' requires dependency 'ControlDeck' to exist before use");
    }
    return mControlDeck;
}

} // namespace Fast
