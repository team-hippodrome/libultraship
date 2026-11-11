#pragma once

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace Fast {

// Source-keyed compiler output. Backends validate driver compatibility before use.
class ShaderDiskCache {
  public:
    void Open(const char* backend);
    std::vector<char> Load(uint64_t key) const;
    void Store(uint64_t key, const void* data, size_t size) const;
    static uint64_t Hash(std::initializer_list<std::string_view> parts);

  private:
    std::filesystem::path PathFor(uint64_t key) const;
    std::filesystem::path mDir;
};

} // namespace Fast
