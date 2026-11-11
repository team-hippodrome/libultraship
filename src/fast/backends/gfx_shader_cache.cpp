// Adapted from Kenix3/libultraship 37daaa66 (Ghostship).
#include "fast/backends/gfx_shader_cache.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "ship/core/Context.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Fast {
namespace {
constexpr size_t kMaxBlobSize = 32 * 1024 * 1024;
constexpr std::array<char, 8> kMagic = { 'S', 'T', 'D', 'S', 'H', 'D', 'R', '1' };

bool TraceCache() {
    static const bool enabled = getenv("PORT_TRACE_SHADER_CACHE") != nullptr;
    return enabled;
}
} // namespace

uint64_t ShaderDiskCache::Hash(std::initializer_list<std::string_view> parts) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (auto part : parts) {
        // A separator length prevents different divisions of the same text colliding.
        uint64_t size = part.size();
        for (unsigned i = 0; i < 8; ++i) {
            hash = (hash ^ ((size >> (i * 8)) & 255)) * 0x100000001b3ULL;
        }
        for (unsigned char byte : part) hash = (hash ^ byte) * 0x100000001b3ULL;
    }
    return hash;
}

void ShaderDiskCache::Open(const char* backend) {
    mDir.clear();
    if (getenv("PORT_NO_SHADER_DISK_CACHE")) return;
    auto path = std::filesystem::u8path(Ship::Context::GetPathRelativeToAppDirectory("shadercache")) / backend;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (!ec) mDir = std::move(path);
}

std::filesystem::path ShaderDiskCache::PathFor(uint64_t key) const {
    char name[32];
    snprintf(name, sizeof(name), "%016llx.bin", static_cast<unsigned long long>(key));
    return mDir / name;
}

std::vector<char> ShaderDiskCache::Load(uint64_t key) const {
    if (mDir.empty()) return {};
    std::ifstream in(PathFor(key), std::ios::binary | std::ios::ate);
    const auto length = in.tellg();
    constexpr size_t headerSize = kMagic.size() + sizeof(uint64_t) * 2;
    if (length <= static_cast<std::streamoff>(headerSize) ||
        length > static_cast<std::streamoff>(headerSize + kMaxBlobSize)) return {};
    in.seekg(0);
    std::array<char, 8> magic{};
    uint64_t storedKey = 0, checksum = 0;
    in.read(magic.data(), magic.size());
    in.read(reinterpret_cast<char*>(&storedKey), sizeof(storedKey));
    in.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    std::vector<char> data(static_cast<size_t>(length) - headerSize);
    in.read(data.data(), data.size());
    if (!in || magic != kMagic || storedKey != key ||
        Hash({ std::string_view(data.data(), data.size()) }) != checksum) return {};
    if (TraceCache()) fprintf(stderr, "[shader-cache] %s hit %016llx\n",
                              mDir.filename().string().c_str(), static_cast<unsigned long long>(key));
    return data;
}

void ShaderDiskCache::Store(uint64_t key, const void* data, size_t size) const {
    if (mDir.empty() || size == 0 || size > kMaxBlobSize) return;
    static std::atomic<uint64_t> sequence{0};
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    const auto path = PathFor(key);
    auto tmp = path;
    tmp += "." + std::to_string(pid) + "." + std::to_string(sequence.fetch_add(1)) + ".tmp";
    const uint64_t checksum = Hash({ std::string_view(static_cast<const char*>(data), size) });
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(kMagic.data(), kMagic.size());
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));
        out.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
        out.write(static_cast<const char*>(data), size);
        out.close();
        if (!out) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    // Unique temporary names also allow multiple warm-up workers or game instances.
#ifdef _WIN32
    const bool saved = MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    const bool saved = !ec;
#endif
    if (!saved) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    } else if (TraceCache()) {
        fprintf(stderr, "[shader-cache] %s stored %016llx (%zu bytes)\n",
                mDir.filename().string().c_str(), static_cast<unsigned long long>(key), size);
    }
}

} // namespace Fast
