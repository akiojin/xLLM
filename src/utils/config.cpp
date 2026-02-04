#include "utils/config.h"
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <spdlog/spdlog.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "utils/allowlist.h"

namespace xllm {

namespace {

std::optional<std::string> getEnvValue(const char* name) {
    if (!name || !*name) {
        return std::nullopt;
    }
#ifdef _WIN32
    if (const char* v = std::getenv(name)) {
        return std::string(v);
    }
    DWORD size = GetEnvironmentVariableA(name, nullptr, 0);
    if (size == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        return std::string();
    }
    std::string value(size, '\0');
    DWORD copied = GetEnvironmentVariableA(name, value.data(), size);
    if (copied == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
        return std::nullopt;
    }
    value.resize(copied);
    return value;
#else
    if (const char* v = std::getenv(name)) {
        return std::string(v);
    }
    return std::nullopt;
#endif
}

/// Get environment variable with fallback to deprecated name
/// Logs a warning if the deprecated name is used
std::optional<std::string> getEnvWithFallback(const char* new_name, const char* old_name) {
    if (auto v = getEnvValue(new_name)) {
        return v;
    }
    if (auto v = getEnvValue(old_name)) {
        spdlog::warn("Environment variable '{}' is deprecated, use '{}' instead", old_name, new_name);
        return v;
    }
    return std::nullopt;
}

std::optional<bool> parseBoolEnv(const std::string& value) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    return std::nullopt;
}

}  // namespace

DownloadConfig loadDownloadConfig() {
    auto info = loadDownloadConfigWithLog();
    return info.first;
}

std::pair<DownloadConfig, std::string> loadDownloadConfigWithLog() {
    DownloadConfig cfg;
    std::ostringstream log;
    bool used_env = false;

    if (auto env = getEnvValue("LLM_DL_MAX_RETRIES")) {
        try {
            int v = std::stoi(*env);
            if (v >= 0) cfg.max_retries = v;
            log << "env:MAX_RETRIES=" << v << " ";
            used_env = true;
        } catch (...) {}
    }

    if (auto env = getEnvValue("LLM_DL_BACKOFF_MS")) {
        try {
            long long ms = std::stoll(*env);
            if (ms >= 0) cfg.backoff = std::chrono::milliseconds(ms);
            log << "env:BACKOFF_MS=" << ms << " ";
            used_env = true;
        } catch (...) {}
    }

    if (auto env = getEnvValue("LLM_DL_CONCURRENCY")) {
        try {
            long long v = std::stoll(*env);
            if (v > 0 && v < 64) cfg.max_concurrency = static_cast<size_t>(v);
            log << "env:CONCURRENCY=" << v << " ";
            used_env = true;
        } catch (...) {}
    }

    if (auto env = getEnvValue("LLM_DL_MAX_BPS")) {
        try {
            long long v = std::stoll(*env);
            if (v > 0) cfg.max_bytes_per_sec = static_cast<size_t>(v);
            log << "env:MAX_BPS=" << v << " ";
            used_env = true;
        } catch (...) {}
    }

    if (auto env = getEnvValue("LLM_DL_CHUNK")) {
        try {
            long long v = std::stoll(*env);
            if (v > 0 && v <= 1 << 20) cfg.chunk_size = static_cast<size_t>(v);
            log << "env:CHUNK=" << v << " ";
            used_env = true;
        } catch (...) {}
    }

    if (auto env = getEnvValue("LLM_DL_TIMEOUT_MS")) {
        try {
            long long v = std::stoll(*env);
            if (v > 0) cfg.timeout = std::chrono::milliseconds(v);
            log << "env:TIMEOUT_MS=" << v << " ";
            used_env = true;
        } catch (...) {}
    }

    if (log.tellp() > 0) log << "|";
    log << "sources=";
    if (used_env) log << "env";
    if (!used_env) log << "default";

    return {cfg, log.str()};
}

namespace {

std::filesystem::path defaultModelsDir() {
    try {
        std::filesystem::path home = getEnvValue("HOME").value_or("");
        if (!home.empty()) return home / ".xllm/models";
    } catch (...) {
    }
    return std::filesystem::path();
}

}  // namespace

std::pair<NodeConfig, std::string> loadNodeConfigWithLog() {
    NodeConfig cfg;
    cfg.bind_address = "0.0.0.0";
    cfg.require_gpu = true;
    std::ostringstream log;
    bool used_env = false;

    // defaults: ~/.xllm/models/
    cfg.models_dir = defaultModelsDir().empty()
                         ? ".xllm/models"
                         : defaultModelsDir().string();
    cfg.origin_allowlist = splitAllowlistCsv("huggingface.co/*,cdn-lfs.huggingface.co/*");

    // env overrides with fallback to deprecated names
    // New names: XLLM_*
    // Deprecated: LLM_* without NODE prefix

    if (auto v = getEnvWithFallback("XLLM_MODELS_DIR", "LLM_MODELS_DIR")) {
        cfg.models_dir = *v;
        log << "env:MODELS_DIR=" << *v << " ";
        used_env = true;
    }
    if (auto v = getEnvWithFallback("XLLM_PORT", "LLM_PORT")) {
        try {
            cfg.node_port = std::stoi(*v);
            log << "env:NODE_PORT=" << cfg.node_port << " ";
            used_env = true;
        } catch (...) {}
    }
    if (auto v = getEnvWithFallback("XLLM_BIND_ADDRESS", "LLM_BIND_ADDRESS")) {
        cfg.bind_address = *v;
        log << "env:BIND_ADDRESS=" << *v << " ";
        used_env = true;
    }
    if (auto v = getEnvWithFallback("XLLM_ORIGIN_ALLOWLIST", "LLM_ORIGIN_ALLOWLIST")) {
        auto list = splitAllowlistCsv(*v);
        if (!list.empty()) {
            cfg.origin_allowlist = std::move(list);
            log << "env:ORIGIN_ALLOWLIST=" << *v << " ";
            used_env = true;
        }
    }

    if (auto v = getEnvWithFallback("XLLM_CORS_ENABLED", "LLM_CORS_ENABLED")) {
        if (auto parsed = parseBoolEnv(*v)) {
            cfg.cors_enabled = *parsed;
            log << "env:CORS_ENABLED=" << (*parsed ? "true" : "false") << " ";
            used_env = true;
        }
    }
    if (auto v = getEnvWithFallback("XLLM_CORS_ALLOW_ORIGIN", "LLM_CORS_ALLOW_ORIGIN")) {
        cfg.cors_allow_origin = *v;
        log << "env:CORS_ALLOW_ORIGIN=" << *v << " ";
        used_env = true;
    }
    if (auto v = getEnvWithFallback("XLLM_CORS_ALLOW_METHODS", "LLM_CORS_ALLOW_METHODS")) {
        cfg.cors_allow_methods = *v;
        log << "env:CORS_ALLOW_METHODS=" << *v << " ";
        used_env = true;
    }
    if (auto v = getEnvWithFallback("XLLM_CORS_ALLOW_HEADERS", "LLM_CORS_ALLOW_HEADERS")) {
        cfg.cors_allow_headers = *v;
        log << "env:CORS_ALLOW_HEADERS=" << *v << " ";
        used_env = true;
    }

    if (auto v = getEnvValue("LLM_DEFAULT_EMBEDDING_MODEL")) {
        cfg.default_embedding_model = *v;
        log << "env:DEFAULT_EMBEDDING_MODEL=" << *v << " ";
        used_env = true;
    }

    if (log.tellp() > 0) log << "|";
    log << "sources=";
    if (used_env) log << "env";
    if (!used_env) log << "default";

    return {cfg, log.str()};
}

NodeConfig loadNodeConfig() {
    auto info = loadNodeConfigWithLog();
    return info.first;
}

}  // namespace xllm
