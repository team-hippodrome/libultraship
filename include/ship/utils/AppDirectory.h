#pragma once

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace Ship {
// Desktop data follows the launcher artifact, never its caller's working directory or an AppImage mount.
// Sandboxed platforms supply their writable files directory instead of the installed library/APK directory.
inline std::filesystem::path ResolveAppDirectory(const std::filesystem::path& executableDirectory,
                                                const std::string& overrideDirectory,
                                                const std::filesystem::path& userHome,
                                                const std::filesystem::path& appImage,
                                                const std::filesystem::path& sandboxFiles,
                                                bool macBundle) {
    namespace fs = std::filesystem;
    if (!overrideDirectory.empty()) {
        auto path = fs::u8path(overrideDirectory);
        if ((overrideDirectory == "~" || overrideDirectory.rfind("~/", 0) == 0) && !userHome.empty()) {
            path = overrideDirectory == "~" ? userHome : userHome / fs::u8path(overrideDirectory.substr(2));
        }
        return fs::absolute(path).lexically_normal();
    }
    if (!sandboxFiles.empty()) return fs::absolute(sandboxFiles).lexically_normal();
    if (!appImage.empty()) return fs::absolute(appImage).lexically_normal().parent_path();
    if (executableDirectory.empty()) throw std::runtime_error("Cannot locate the launcher folder; set SHIP_HOME.");
    auto directory = fs::absolute(executableDirectory).lexically_normal();
    if (macBundle) {
        for (auto bundle = directory; !bundle.empty(); bundle = bundle.parent_path()) {
            auto extension = bundle.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
                return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
            });
            if (extension == ".app") return bundle.parent_path();
            if (bundle == bundle.parent_path()) break;
        }
    }
    return directory;
}
} // namespace Ship
