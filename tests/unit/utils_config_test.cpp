#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <unordered_map>

#include "utils/config.h"

using namespace xllm;
namespace fs = std::filesystem;

class EnvGuard {
public:
    EnvGuard(const std::vector<std::string>& keys) : keys_(keys) {
        for (const auto& k : keys_) {
            const char* v = std::getenv(k.c_str());
            if (v) saved_[k] = v;
        }
    }
    ~EnvGuard() {
        for (const auto& k : keys_) {
            if (auto it = saved_.find(k); it != saved_.end()) {
                setenv(k.c_str(), it->second.c_str(), 1);
            } else {
                unsetenv(k.c_str());
            }
        }
    }
private:
    std::vector<std::string> keys_;
    std::unordered_map<std::string, std::string> saved_;
};

TEST(UtilsConfigTest, EnvOverridesNodeConfig) {
    EnvGuard guard({"LLM_MODELS_DIR", "XLLM_PORT", "XLLM_MODELS_DIR"});

    // Test with deprecated env var names (fallback)
    setenv("LLM_MODELS_DIR", "/env/models", 1);
    setenv("XLLM_PORT", "19000", 1);
    auto cfg = loadNodeConfig();
    EXPECT_EQ(cfg.models_dir, "/env/models");
    EXPECT_EQ(cfg.node_port, 19000);
    EXPECT_TRUE(cfg.require_gpu);  // env flags are ignored, GPU required
}

TEST(UtilsConfigTest, NewEnvVarsTakePriorityOverDeprecated) {
    EnvGuard guard({"XLLM_MODELS_DIR", "LLM_MODELS_DIR",
                    "XLLM_PORT"});

    // Set both new and deprecated env vars
    setenv("XLLM_MODELS_DIR", "/new/models", 1);
    setenv("LLM_MODELS_DIR", "/old/models", 1);  // Should be ignored
    auto cfg = loadNodeConfig();
    EXPECT_EQ(cfg.models_dir, "/new/models");
    EXPECT_TRUE(cfg.require_gpu);  // GPU requirement cannot be disabled
}

TEST(UtilsConfigTest, DefaultModelsDirIsXllmModels) {
    // Verify the default models_dir is ~/.xllm/models (not ~/.llmlb/)
    EnvGuard guard({"HOME", "XLLM_MODELS_DIR", "LLM_MODELS_DIR"});
    unsetenv("XLLM_MODELS_DIR");
    unsetenv("LLM_MODELS_DIR");

    fs::path tmp_home = fs::temp_directory_path() / "test_xllm_models";
    fs::create_directories(tmp_home);
    setenv("HOME", tmp_home.string().c_str(), 1);

    auto cfg = loadNodeConfig();

    // Default models_dir should be under .xllm
    EXPECT_NE(cfg.models_dir.find(".xllm"), std::string::npos);
    EXPECT_NE(cfg.models_dir.find("models"), std::string::npos);

    fs::remove_all(tmp_home);
}

TEST(UtilsConfigTest, EnvOverridesDownloadConfig) {
    EnvGuard guard({"LLM_DL_MAX_RETRIES",
                    "LLM_DL_BACKOFF_MS",
                    "LLM_DL_CONCURRENCY",
                    "LLM_DL_MAX_BPS",
                    "LLM_DL_CHUNK",
                    "LLM_DL_TIMEOUT_MS"});

    setenv("LLM_DL_MAX_RETRIES", "5", 1);
    setenv("LLM_DL_BACKOFF_MS", "900", 1);
    setenv("LLM_DL_CONCURRENCY", "7", 1);
    setenv("LLM_DL_MAX_BPS", "1024", 1);
    setenv("LLM_DL_CHUNK", "8192", 1);
    setenv("LLM_DL_TIMEOUT_MS", "120000", 1);

    auto cfg = loadDownloadConfig();
    EXPECT_EQ(cfg.max_retries, 5);
    EXPECT_EQ(cfg.backoff.count(), 900);
    EXPECT_EQ(cfg.max_concurrency, 7u);
    EXPECT_EQ(cfg.max_bytes_per_sec, 1024u);
    EXPECT_EQ(cfg.chunk_size, 8192u);
    EXPECT_EQ(cfg.timeout.count(), 120000);
}
