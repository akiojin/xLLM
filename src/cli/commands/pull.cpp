// SPEC-58378000: node pull command
// Downloads a model from HuggingFace

#include "utils/cli.h"
#include "cli/cli_client.h"
#include "cli/progress_renderer.h"
#include "models/model_downloader.h"
#include "models/model_sync.h"
#include "utils/config.h"
#include "utils/allowlist.h"
#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <optional>
#include <cctype>

namespace xllm {
namespace cli {
namespace commands {

namespace {

struct HfModelRef {
    std::string repo;
    std::string filename;
};

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional<std::string> hostFromUrl(const std::string& url) {
    auto scheme_pos = url.find("://");
    if (scheme_pos == std::string::npos) return std::nullopt;
    std::string rest = url.substr(scheme_pos + 3);
    auto slash = rest.find('/');
    std::string host = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    if (host.empty()) return std::nullopt;
    return host;
}

bool isHfHost(const std::string& host) {
    const std::string lower = toLowerAscii(host);
    if (lower == "huggingface.co" || endsWith(lower, ".huggingface.co")) {
        return true;
    }
    if (const char* base = std::getenv("HF_BASE_URL")) {
        std::string base_url(base);
        auto base_host = hostFromUrl(base_url);
        if (base_host && toLowerAscii(*base_host) == lower) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> splitPath(const std::string& path) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < path.size()) {
        size_t pos = path.find('/', start);
        std::string segment = (pos == std::string::npos)
            ? path.substr(start)
            : path.substr(start, pos - start);
        if (!segment.empty()) {
            parts.push_back(segment);
        }
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return parts;
}

std::string joinPath(const std::vector<std::string>& parts, size_t start_index) {
    std::string out;
    for (size_t i = start_index; i < parts.size(); ++i) {
        if (!out.empty()) out.push_back('/');
        out.append(parts[i]);
    }
    return out;
}

std::optional<HfModelRef> parseHfModelRef(const std::string& input) {
    if (input.empty()) return std::nullopt;

    if (input.find("://") == std::string::npos) {
        std::string repo = input;
        auto qpos = repo.find_first_of("?#");
        if (qpos != std::string::npos) {
            repo = repo.substr(0, qpos);
        }
        if (!repo.empty() && repo.back() == '/') {
            repo.pop_back();
        }
        auto parts = splitPath(repo);
        if (parts.size() < 2) return std::nullopt;
        HfModelRef ref;
        ref.repo = parts[0] + "/" + parts[1];
        if (parts.size() > 2) {
            ref.filename = joinPath(parts, 2);
        }
        return ref;
    }

    auto host = hostFromUrl(input);
    if (!host || !isHfHost(*host)) return std::nullopt;

    auto scheme_pos = input.find("://");
    std::string rest = input.substr(scheme_pos + 3);
    auto slash = rest.find('/');
    if (slash == std::string::npos) return std::nullopt;
    std::string path = rest.substr(slash + 1);

    auto qpos = path.find_first_of("?#");
    if (qpos != std::string::npos) {
        path = path.substr(0, qpos);
    }

    auto parts = splitPath(path);
    if (parts.size() < 2) return std::nullopt;

    HfModelRef ref;
    ref.repo = parts[0] + "/" + parts[1];
    if (parts.size() > 2) {
        const std::string& marker = parts[2];
        if ((marker == "resolve" || marker == "blob" || marker == "raw") && parts.size() >= 5) {
            ref.filename = joinPath(parts, 4);
        } else if (marker != "tree") {
            ref.filename = joinPath(parts, 2);
        }
    }

    return ref;
}

}  // namespace

static int pullDirect(const PullOptions& options) {
    auto ref = parseHfModelRef(options.model);
    if (!ref) {
        std::cerr << "Error: invalid HuggingFace model reference (expected owner/model or HuggingFace URL)" << std::endl;
        return 1;
    }

    auto cfg = loadNodeConfig();
    ModelSync sync("", cfg.models_dir);
    if (!cfg.origin_allowlist.empty()) {
        sync.setOriginAllowlist(cfg.origin_allowlist);
    }
    ModelDownloader downloader("", cfg.models_dir);

    ProgressRenderer progress;
    progress.setPhase("pulling manifest");

    struct FileProgress {
        uint64_t downloaded{0};
        uint64_t total{0};
    };

    std::mutex mu;
    std::unordered_map<std::string, FileProgress> files;
    auto last_ts = std::chrono::steady_clock::now();
    uint64_t last_bytes = 0;

    DownloadCallbacks callbacks;
    callbacks.on_manifest = [&](const std::vector<std::string>& names) {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto& name : names) {
            files.emplace(name, FileProgress{});
        }
        progress.setPhase("downloading");
    };
    callbacks.on_progress = [&](const std::string& file, size_t downloaded, size_t total) {
        std::lock_guard<std::mutex> lock(mu);
        auto& fp = files[file];
        fp.downloaded = downloaded;
        fp.total = total;

        uint64_t sum_downloaded = 0;
        uint64_t sum_total = 0;
        for (const auto& kv : files) {
            sum_downloaded += kv.second.downloaded;
            sum_total += kv.second.total;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_ts).count();
        double speed = 0.0;
        if (elapsed > 0.0 && sum_downloaded >= last_bytes) {
            speed = static_cast<double>(sum_downloaded - last_bytes) / elapsed;
        }
        last_ts = now;
        last_bytes = sum_downloaded;
        progress.update(sum_downloaded, sum_total, speed);
    };
    callbacks.on_complete = [&](const std::string& file, bool success) {
        if (!success) return;
        std::lock_guard<std::mutex> lock(mu);
        auto& fp = files[file];
        if (fp.total == 0 && fp.downloaded > 0) {
            fp.total = fp.downloaded;
        }
    };

    bool ok = sync.downloadModel(downloader, ref->repo, callbacks, ref->filename);
    if (!ok) {
        progress.fail(downloader.getLastError());
        return 1;
    }
    progress.complete();
    return 0;
}

/// Execute the 'node pull' command
/// @param options Pull options (model name/URL)
/// @return Exit code (0=success, 1=error, 2=connection error)
int pull(const PullOptions& options) {
    if (options.model.empty()) {
        std::cerr << "Error: model name or HuggingFace URL required" << std::endl;
        return 1;
    }

    if (options.direct) {
        return pullDirect(options);
    }

    // Check for HF_TOKEN if this might be a gated model
    const char* hf_token = std::getenv("HF_TOKEN");

    // Create CLI client
    auto client = std::make_shared<CliClient>();

    // Check server connection
    if (!client->isServerRunning()) {
        std::cerr << "Warning: Could not connect to xllm server, falling back to direct download" << std::endl;
        return pullDirect(options);
    }

    std::cout << "pulling " << options.model << std::endl;

    // Create progress renderer
    ProgressRenderer progress;
    progress.setPhase("pulling manifest");

    // Pull model with progress callback
    auto result = client->pullModel(options.model, [&progress](uint64_t downloaded, uint64_t total, double speed) {
        progress.update(downloaded, total, speed);
    });

    if (!result.ok()) {
        progress.fail(result.error_message);

        // Check if it's a gated model error
        if (result.error_message.find("gated") != std::string::npos ||
            result.error_message.find("403") != std::string::npos) {
            if (!hf_token) {
                std::cerr << "This model requires authentication. Set HF_TOKEN environment variable." << std::endl;
            }
        }

        return 1;
    }

    progress.complete();
    return 0;
}

}  // namespace commands
}  // namespace cli
}  // namespace xllm
