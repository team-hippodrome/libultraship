#include "ship/config/Config.h"

#include <fstream>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <any>
#include <cstdio>
#include <mutex>
#include <spdlog/spdlog.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include "ship/utils/StringHelper.h"
#include "ship/window/Window.h"

namespace fs = std::filesystem;

namespace Ship {
Config::Config(const std::string& path, std::shared_ptr<Window> window)
    : Component("Config"), mPath(path), mIsNewInstance(false), mWindow(std::move(window)) {
    Reload();
    MarkInitialized();
}

Config::~Config() {
    SPDLOG_TRACE("destruct config");
}

const std::string& Config::GetPath() const {
    return mPath;
}

std::string Config::FormatNestedKey(const std::string& key) {
    std::vector<std::string> dots = StringHelper::Split(key, ".");

    std::string tmp;
    if (dots.size() > 1) {
        for (const auto& dot : dots) {
            tmp += "/" + dot;
        }
    } else {
        tmp = "/" + dots[0];
    }

    return tmp;
}

nlohmann::json Config::Nested(const std::string& key) {
    std::vector<std::string> dots = StringHelper::Split(key, ".");
    if (!mFlattenedJson.is_object()) {
        return mFlattenedJson;
    }

    nlohmann::json gjson = mFlattenedJson.unflatten();

    if (dots.size() > 1) {
        for (auto& dot : dots) {
            if (dot == "*" || gjson.contains(dot)) {
                gjson = gjson[dot];
            }
        }
        return gjson;
    }

    return gjson[key];
}

std::string Config::GetString(const std::string& key, const std::string& defaultValue) {
    nlohmann::json n = Nested(key);
    if (n.is_string() && !n.get<std::string>().empty()) {
        return n;
    }

    return defaultValue;
}

float Config::GetFloat(const std::string& key, float defaultValue) {
    nlohmann::json n = Nested(key);
    if (n.is_number_float()) {
        return n;
    }

    return defaultValue;
}

bool Config::GetBool(const std::string& key, bool defaultValue) {
    nlohmann::json n = Nested(key);
    if (n.is_boolean()) {
        return n;
    }

    return defaultValue;
}

int32_t Config::GetInt(const std::string& key, int32_t defaultValue) {
    nlohmann::json n = Nested(key);
    if (n.is_number_integer()) {
        return n;
    }

    return defaultValue;
}

uint32_t Config::GetUInt(const std::string& key, uint32_t defaultValue) {
    nlohmann::json n = Nested(key);
    if (n.is_number_unsigned()) {
        return n;
    }

    return defaultValue;
}

bool Config::Contains(const std::string& key) {
    return !Nested(key).is_null();
}

void Config::SetString(const std::string& key, const std::string& value) {
    mFlattenedJson[FormatNestedKey(key)] = value;
}

void Config::SetFloat(const std::string& key, float value) {
    mFlattenedJson[FormatNestedKey(key)] = value;
}

void Config::SetBool(const std::string& key, bool value) {
    mFlattenedJson[FormatNestedKey(key)] = value;
}

void Config::SetInt(const std::string& key, int32_t value) {
    mFlattenedJson[FormatNestedKey(key)] = value;
}

void Config::SetUInt(const std::string& key, uint32_t value) {
    mFlattenedJson[FormatNestedKey(key)] = value;
}

void Config::Erase(const std::string& key) {
    mFlattenedJson.erase(FormatNestedKey(key));
}

void Config::SetBlock(const std::string& key, nlohmann::json block) {
    nlohmann::json gjson = mFlattenedJson.unflatten();
    if (key.find(".") != std::string::npos) {
        nlohmann::json* gjson2 = &gjson;
        std::vector<std::string> dots = StringHelper::Split(key, ".");
        if (dots.size() > 1) {
            size_t curDot = 0;
            for (auto& dot : dots) {
                if (curDot == dots.size() - 1) {
                    if (gjson2->contains(dot)) {
                        gjson2->at(dot) = block;
                        break;
                    } else {
                        gjson2->emplace(dot, block);
                        break;
                    }
                } else if (gjson2->contains(dot)) {
                    gjson2 = &gjson2->at(dot);
                    curDot++;
                }
            }
        }
    } else {
        gjson[key] = block;
    }
    mFlattenedJson = gjson.flatten();
    Save();
}

void Config::EraseBlock(const std::string& key) {
    nlohmann::json gjson = mFlattenedJson.unflatten();
    if (key.find(".") != std::string::npos) {
        nlohmann::json* gjson2 = &gjson;
        std::vector<std::string> dots = StringHelper::Split(key, ".");
        if (dots.size() > 1) {
            size_t curDot = 0;
            for (auto& dot : dots) {
                if (gjson2->contains(dot)) {
                    if (curDot == dots.size() - 1) {
                        gjson2->at(dot).clear();
                        gjson2->erase(dot);
                    } else {
                        gjson2 = &gjson2->at(dot);
                        curDot++;
                    }
                }
            }
        }
    } else {
        if (gjson.contains(key)) {
            gjson.erase(key);
        }
    }
    mFlattenedJson = gjson.flatten();
    Save();
}

void Config::Copy(const std::string& fromKey, const std::string& toKey) {
    auto nestedFromKey = FormatNestedKey(fromKey);
    auto nestedToKey = FormatNestedKey(toKey);
    if (mFlattenedJson.contains(nestedFromKey)) {
        mFlattenedJson[nestedToKey] = mFlattenedJson[nestedFromKey];
    }
}

void Config::Reload() {
    if (mPath == "None" || !fs::exists(fs::u8path(mPath)) || !fs::is_regular_file(fs::u8path(mPath))) {
        mIsNewInstance = true;
        mFlattenedJson = nlohmann::json::object();
        return;
    }
    std::ifstream ifs(fs::u8path(mPath));

    mNestedJson = nlohmann::json::object();
    mFlattenedJson = nlohmann::json::object();
    try {
        mNestedJson = nlohmann::json::parse(ifs);
        mFlattenedJson = mNestedJson.flatten();
    } catch (const nlohmann::json::exception& e) {
        SPDLOG_ERROR("Failed to parse config file {}: {}", mPath, e.what());
    } catch (const std::exception& e) { SPDLOG_ERROR("Unexpected error loading config file {}: {}", mPath, e.what()); }
}

// The file is written next to the config under a temporary name and then renamed over it, so the config on disk is
// always either the old or the new file. Writing in place truncated it first: a process that ended during the write
// (a crash, a terminate, _exit on another thread) left an empty or cut-off config behind.
void Config::Save() {
    static std::mutex saveMutex; // one temporary file per config path: two saves at once must not share it
    std::lock_guard<std::mutex> lock(saveMutex);
    mNestedJson = mFlattenedJson.unflatten();
    if (mPath == "None") {
        return;
    }
    const std::string text = mNestedJson.dump(4);
    const fs::path path = fs::u8path(mPath);
    const fs::path temporary = fs::u8path(mPath + ".tmp");
#ifdef _WIN32
    FILE* file = _wfopen(temporary.c_str(), L"wb");
#else
    FILE* file = fopen(temporary.c_str(), "wb");
#endif
    if (file == nullptr) {
        SPDLOG_ERROR("Failed to save config file {}: cannot create {}", mPath, temporary.string());
        return;
    }
    bool ok = fwrite(text.data(), 1, text.size(), file) == text.size() && fflush(file) == 0;
#ifdef _WIN32
    ok = ok && _commit(_fileno(file)) == 0;
#else
    ok = ok && fsync(fileno(file)) == 0;
#endif
    ok = fclose(file) == 0 && ok;
    std::error_code ec;
    if (!ok) {
        fs::remove(temporary, ec);
        SPDLOG_ERROR("Failed to save config file {}: write failed; the previous file was kept", mPath);
        return;
    }
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long error = GetLastError();
        fs::remove(temporary, ec);
        SPDLOG_ERROR("Failed to save config file {}: cannot replace it (error {}); the previous file was kept", mPath,
                     error);
    }
#else
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        fs::remove(temporary, ec);
        SPDLOG_ERROR("Failed to save config file {}: cannot replace it; the previous file was kept", mPath);
    }
#endif
}

template <typename T> std::vector<T> Config::GetArray(const std::string& key) {
    if (nlohmann::json tmp = Nested(key); tmp.is_array()) {
        return tmp.get<std::vector<T>>();
    }
    return std::vector<T>();
};

template <typename T> void Config::SetArray(const std::string& key, std::vector<T> array) {
    mFlattenedJson[FormatNestedKey(key)] = nlohmann::json(array);
}

nlohmann::json Config::GetNestedJson() {
    return mNestedJson;
}

bool Config::RegisterVersionUpdater(std::shared_ptr<ConfigVersionUpdater> versionUpdater) {
    auto [_, emplaced] = mVersionUpdaters.emplace(versionUpdater->GetVersion(), versionUpdater);
    return emplaced;
}

void Config::RunVersionUpdates() {
    for (auto [_, versionUpdater] : mVersionUpdaters) {
        uint32_t version = GetUInt("ConfigVersion", 0);
        if (version < versionUpdater->GetVersion()) {
            versionUpdater->Update(this);
            SetUInt("ConfigVersion", versionUpdater->GetVersion());
        }
    }
    Save();
}

ConfigVersionUpdater::ConfigVersionUpdater(uint32_t toVersion) : mVersion(toVersion) {
}

uint32_t ConfigVersionUpdater::GetVersion() {
    return mVersion;
}

} // namespace Ship
