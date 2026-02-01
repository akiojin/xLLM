/**
 * @file tokenizer.cpp
 * @brief Tokenizer implementation with BPE support
 */

#include "safetensors_internal.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <regex>

#include <vector>

// Provided by llama.cpp (linked via xLLM). Keep declaration local to avoid fragile includes.
std::vector<std::string> unicode_regex_split(const std::string& text,
                                             const std::vector<std::string>& regex_exprs);

namespace stcpp {

static bool stcpp_debug_enabled() {
    const char* env = std::getenv("STCPP_DEBUG");
    return env && *env && std::strcmp(env, "0") != 0;
}

#ifndef STCPP_DEBUG_LOG
#define STCPP_DEBUG_LOG(fmt, ...) \
    do { \
        if (stcpp_debug_enabled()) { \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while (0)
#endif

namespace json_parser {
extern void skip_ws(const char*& p, const char* end);
extern std::string parse_string(const char*& p, const char* end);
extern int64_t parse_int(const char*& p, const char* end);
extern void skip_value(const char*& p, const char* end);
}  // namespace json_parser

static std::string merge_pair_key(const std::string& first, const std::string& second);

static std::string normalize_pretokenizer_regex(std::string regex) {
    const std::string needle = "(?i:'s|'t|'re|'ve|'m|'ll|'d)";
    const std::string replacement = "'([sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])";
    size_t pos = 0;
    while ((pos = regex.find(needle, pos)) != std::string::npos) {
        regex.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
    return regex;
}

static bool extract_pretokenizer_regex(const std::string& content, std::string& regex) {
    const char* p = content.data();
    const char* end = p + content.size();
    const char* key = "\"Regex\"";
    const char* found = std::strstr(p, key);
    while (found) {
        const char* q = found + std::strlen(key);
        json_parser::skip_ws(q, end);
        if (q < end && *q == ':') {
            p = q + 1;
            break;
        }
        found = std::strstr(found + 1, key);
    }
    if (!found) return false;
    json_parser::skip_ws(p, end);
    if (p >= end || *p != '"') return false;

    regex = json_parser::parse_string(p, end);
    return !regex.empty();
}

static bool extract_config_token_id(const std::string& content, const char* key, int32_t& out) {
    std::string needle = std::string("\"") + key + "\"";
    const char* p = content.data();
    const char* end = p + content.size();
    const char* found = std::strstr(p, needle.c_str());
    if (!found) {
        return false;
    }
    p = found + needle.size();
    while (p < end && *p != ':') ++p;
    if (p >= end) {
        return false;
    }
    ++p;
    json_parser::skip_ws(p, end);
    if (p >= end) {
        return false;
    }
    if (*p == '[') {
        ++p;
        json_parser::skip_ws(p, end);
        if (p < end && ((*p >= '0' && *p <= '9') || *p == '-')) {
            out = static_cast<int32_t>(json_parser::parse_int(p, end));
            return true;
        }
        return false;
    }
    if ((*p >= '0' && *p <= '9') || *p == '-') {
        out = static_cast<int32_t>(json_parser::parse_int(p, end));
        return true;
    }
    return false;
}

static bool detect_pretokenizer_byte_encoded(const std::string& content) {
    // Heuristic: if the pre_tokenizer includes a Split stage, the regex runs
    // on raw text and ByteLevel encoding should happen after splitting.
    const auto pre_pos = content.find("\"pre_tokenizer\"");
    if (pre_pos == std::string::npos) {
        return false;
    }
    const std::string_view slice(content.data() + pre_pos,
                                 content.size() - pre_pos);
    const bool has_split = (slice.find("\"type\":\"Split\"") != std::string_view::npos) ||
                           (slice.find("\"type\": \"Split\"") != std::string_view::npos);
    if (has_split) {
        return false;
    }
    const auto byte_pos = slice.find("\"ByteLevel\"");
    if (byte_pos == std::string_view::npos) {
        return false;
    }
    const auto use_pos = slice.find("\"use_regex\"", byte_pos);
    if (use_pos == std::string_view::npos) {
        // Default ByteLevel behavior applies regex on byte-encoded text.
        return true;
    }
    const auto true_pos = slice.find("true", use_pos);
    if (true_pos != std::string_view::npos && true_pos - use_pos < 16) {
        return true;
    }
    return false;
}

/* Parse vocab from JSON object */
static bool parse_vocab(
    const char*& p,
    const char* end,
    TokenizerImpl& tokenizer,
    std::string& error
) {
    json_parser::skip_ws(p, end);
    if (p >= end || *p != '{') {
        error = "Expected '{' for vocab";
        return false;
    }
    ++p;

    while (p < end) {
        json_parser::skip_ws(p, end);
        if (p >= end || *p == '}') break;

        const char* before = p;
        std::string token = json_parser::parse_string(p, end);
        if (p == before) {
            error = "Invalid tokenizer.json: expected string key in vocab";
            return false;
        }
        json_parser::skip_ws(p, end);
        if (p < end && *p == ':') ++p;
        json_parser::skip_ws(p, end);

        int32_t id = static_cast<int32_t>(json_parser::parse_int(p, end));
        if (id >= 0) {
            if (id >= static_cast<int32_t>(tokenizer.vocab.size())) {
                tokenizer.vocab.resize(static_cast<size_t>(id) + 1);
            }
            tokenizer.vocab[static_cast<size_t>(id)] = token;
            tokenizer.vocab_to_id[token] = id;
        }

        json_parser::skip_ws(p, end);
        if (p < end && *p == ',') ++p;
    }
    if (p < end && *p == '}') ++p;

    return true;
}

/* Parse merge rules from JSON array */
static bool parse_merges(
    const char*& p,
    const char* end,
    TokenizerImpl& tokenizer,
    std::string& error
) {
    json_parser::skip_ws(p, end);
    if (p >= end || *p != '[') {
        error = "Expected '[' for merges";
        return false;
    }
    ++p;

    while (p < end) {
        json_parser::skip_ws(p, end);
        if (p >= end || *p == ']') break;

        if (*p == '[') {
            ++p;
            json_parser::skip_ws(p, end);
            const char* before_first = p;
            std::string first = json_parser::parse_string(p, end);
            if (p == before_first) {
                error = "Invalid tokenizer.json: expected string in merge pair";
                return false;
            }
            json_parser::skip_ws(p, end);
            if (p < end && *p == ',') ++p;
            json_parser::skip_ws(p, end);
            const char* before_second = p;
            std::string second = json_parser::parse_string(p, end);
            if (p == before_second) {
                error = "Invalid tokenizer.json: expected second string in merge pair";
                return false;
            }
            // Skip any trailing elements in the pair if present.
            while (p < end && *p != ']') {
                if (*p == ',') {
                    ++p;
                    json_parser::skip_value(p, end);
                } else {
                    ++p;
                }
            }
            if (p < end && *p == ']') ++p;
            tokenizer.merges.push_back({first, second});
        } else {
            const char* before = p;
            std::string merge = json_parser::parse_string(p, end);
            if (p == before) {
                error = "Invalid tokenizer.json: expected string entry in merges";
                return false;
            }
            // Parse "token1 token2" format
            size_t space = merge.find(' ');
            if (space != std::string::npos) {
                std::string first = merge.substr(0, space);
                std::string second = merge.substr(space + 1);
                tokenizer.merges.push_back({first, second});
            }
        }

        json_parser::skip_ws(p, end);
        if (p < end && *p == ',') ++p;
    }
    if (p < end && *p == ']') ++p;

    return true;
}

/* Parse added_tokens array for special tokens */
static bool parse_added_tokens(
    const char*& p,
    const char* end,
    TokenizerImpl& tokenizer,
    std::string& /*error*/
) {
    json_parser::skip_ws(p, end);
    if (p >= end || *p != '[') {
        return true;  // Optional field
    }
    ++p;

    while (p < end) {
        json_parser::skip_ws(p, end);
        if (p >= end || *p == ']') break;

        if (*p == '{') {
            ++p;
            int32_t id = -1;
            std::string content;
            bool is_special = false;

            while (p < end && *p != '}') {
                json_parser::skip_ws(p, end);
                std::string key = json_parser::parse_string(p, end);
                json_parser::skip_ws(p, end);
                if (p < end && *p == ':') ++p;
                json_parser::skip_ws(p, end);

                if (key == "id") {
                    id = static_cast<int32_t>(json_parser::parse_int(p, end));
                } else if (key == "content") {
                    content = json_parser::parse_string(p, end);
                } else if (key == "special") {
                    // Parse boolean
                    if (p + 4 <= end && strncmp(p, "true", 4) == 0) {
                        is_special = true;
                        p += 4;
                    } else if (p + 5 <= end && strncmp(p, "false", 5) == 0) {
                        is_special = false;
                        p += 5;
                    } else {
                        json_parser::skip_value(p, end);
                    }
                } else {
                    json_parser::skip_value(p, end);
                }

                json_parser::skip_ws(p, end);
                if (p < end && *p == ',') ++p;
            }
            if (p < end && *p == '}') ++p;

            // Add all added_tokens to vocab (critical for special token tokenization)
            if (id >= 0 && !content.empty()) {
                // Ensure vocab is large enough
                if (id >= static_cast<int32_t>(tokenizer.vocab.size())) {
                    tokenizer.vocab.resize(id + 1);
                }
                tokenizer.vocab[id] = content;
                tokenizer.vocab_to_id[content] = id;

                // Mark special tokens
                if (is_special) {
                    tokenizer.special_tokens.insert(content);
                    STCPP_DEBUG_LOG("[DEBUG] Special token added: '%s' id=%d\n", content.c_str(), id);
                }

                // Identify bos/eos/pad tokens by content
                if (is_special) {
                    if (content == "<s>" || content == "<|begin_of_text|>" ||
                        content == "[CLS]" || content == "<bos>" || content == "<|im_start|>") {
                        tokenizer.bos_token_id = id;
                    } else if (content == "</s>" || content == "<|end_of_text|>" ||
                               content == "[SEP]" || content == "<eos>" ||
                               content == "<|endoftext|>" || content == "<|im_end|>") {
                        tokenizer.eos_token_id = id;
                    } else if (content == "<pad>" || content == "[PAD]" ||
                               content == "<|pad|>") {
                        tokenizer.pad_token_id = id;
                    }
                }
            }
        } else {
            json_parser::skip_value(p, end);
        }

        json_parser::skip_ws(p, end);
        if (p < end && *p == ',') ++p;
    }
    if (p < end && *p == ']') ++p;

    return true;
}

/* Load tokenizer from tokenizer.json */
bool load_tokenizer(
    const std::string& model_dir,
    TokenizerImpl& tokenizer,
    std::string& error
) {
    std::filesystem::path tokenizer_path = std::filesystem::path(model_dir) / "tokenizer.json";

    std::ifstream file(tokenizer_path);
    if (!file.is_open()) {
        error = "Failed to open tokenizer.json: " + tokenizer_path.string();
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    const char* p = content.data();
    const char* end = p + content.size();

    json_parser::skip_ws(p, end);
    if (p >= end || *p != '{') {
        error = "Invalid tokenizer.json: expected '{'";
        return false;
    }
    ++p;

    while (p < end) {
        json_parser::skip_ws(p, end);
        if (p >= end || *p == '}') break;

        std::string key = json_parser::parse_string(p, end);
        json_parser::skip_ws(p, end);
        if (p < end && *p == ':') ++p;
        json_parser::skip_ws(p, end);

        if (key == "model") {
            // Parse model object containing vocab and merges
            if (p < end && *p == '{') {
                ++p;
                while (p < end && *p != '}') {
                    json_parser::skip_ws(p, end);
                    std::string model_key = json_parser::parse_string(p, end);
                    json_parser::skip_ws(p, end);
                    if (p < end && *p == ':') ++p;
                    json_parser::skip_ws(p, end);

                    if (model_key == "vocab") {
                        if (!parse_vocab(p, end, tokenizer, error)) {
                            return false;
                        }
                    } else if (model_key == "merges") {
                        if (!parse_merges(p, end, tokenizer, error)) {
                            return false;
                        }
                    } else if (model_key == "ignore_merges") {
                        if (p + 4 <= end && std::strncmp(p, "true", 4) == 0) {
                            tokenizer.ignore_merges = true;
                            p += 4;
                        } else if (p + 5 <= end && std::strncmp(p, "false", 5) == 0) {
                            tokenizer.ignore_merges = false;
                            p += 5;
                        } else {
                            json_parser::skip_value(p, end);
                        }
                    } else {
                        json_parser::skip_value(p, end);
                    }

                    json_parser::skip_ws(p, end);
                    if (p < end && *p == ',') ++p;
                }
                if (p < end && *p == '}') ++p;
            }
        } else if (key == "added_tokens") {
            if (!parse_added_tokens(p, end, tokenizer, error)) {
                return false;
            }
        } else {
            json_parser::skip_value(p, end);
        }

        json_parser::skip_ws(p, end);
        if (p < end && *p == ',') ++p;
    }

    // Extract pre_tokenizer regex (if present)
    std::string pretokenizer_regex;
    if (extract_pretokenizer_regex(content, pretokenizer_regex)) {
        tokenizer.pretokenizer_regex = normalize_pretokenizer_regex(pretokenizer_regex);
        tokenizer.pretokenizer_byte_encoded = detect_pretokenizer_byte_encoded(content);
        STCPP_DEBUG_LOG("[DEBUG] load_tokenizer: pretokenizer regex loaded (len=%zu, byte_encoded=%s)\n",
                        tokenizer.pretokenizer_regex.size(),
                        tokenizer.pretokenizer_byte_encoded ? "true" : "false");
    }

    STCPP_DEBUG_LOG("[DEBUG] load_tokenizer: vocab loaded, vocab_size=%zu, merges=%zu, ignore_merges=%s\n",
                    tokenizer.vocab.size(), tokenizer.merges.size(),
                    tokenizer.ignore_merges ? "true" : "false");

    if (tokenizer.ignore_merges && !tokenizer.merges.empty()) {
        const auto has_token = [&tokenizer](const std::string& tok) {
            return tokenizer.special_tokens.count(tok) > 0;
        };
        if (has_token("<|start|>") && has_token("<|message|>") && has_token("<|channel|>")) {
            tokenizer.ignore_merges = false;
            STCPP_DEBUG_LOG("[DEBUG] load_tokenizer: override ignore_merges=false for GPT-OSS-style tokenizer\n");
        }
    }

    if (!tokenizer.ignore_merges && !tokenizer.merges.empty()) {
        tokenizer.merge_ranks.clear();
        tokenizer.merge_ranks.reserve(tokenizer.merges.size());
        for (size_t i = 0; i < tokenizer.merges.size(); ++i) {
            const auto& pair = tokenizer.merges[i];
            tokenizer.merge_ranks.emplace(merge_pair_key(pair.first, pair.second),
                                          static_cast<int32_t>(i));
        }
    } else {
        tokenizer.merge_ranks.clear();
    }

    // Load tokenizer_config.json for additional settings
    std::filesystem::path config_path = std::filesystem::path(model_dir) / "tokenizer_config.json";
    std::ifstream config_file(config_path);
    if (config_file.is_open()) {
        std::string config_content((std::istreambuf_iterator<char>(config_file)),
                                    std::istreambuf_iterator<char>());

        const char* cp = config_content.data();
        const char* cend = cp + config_content.size();

        json_parser::skip_ws(cp, cend);
        if (cp < cend && *cp == '{') {
            ++cp;
            while (cp < cend) {
                json_parser::skip_ws(cp, cend);
                if (cp >= cend || *cp == '}') break;

                std::string cfg_key = json_parser::parse_string(cp, cend);
                json_parser::skip_ws(cp, cend);
                if (cp < cend && *cp == ':') ++cp;
                json_parser::skip_ws(cp, cend);

                if (cfg_key == "bos_token") {
                    std::string token = json_parser::parse_string(cp, cend);
                    auto it = tokenizer.vocab_to_id.find(token);
                    if (it != tokenizer.vocab_to_id.end()) {
                        tokenizer.bos_token_id = it->second;
                    }
                } else if (cfg_key == "eos_token") {
                    std::string token = json_parser::parse_string(cp, cend);
                    auto it = tokenizer.vocab_to_id.find(token);
                    if (it != tokenizer.vocab_to_id.end()) {
                        tokenizer.eos_token_id = it->second;
                    }
                } else if (cfg_key == "pad_token") {
                    std::string token = json_parser::parse_string(cp, cend);
                    auto it = tokenizer.vocab_to_id.find(token);
                    if (it != tokenizer.vocab_to_id.end()) {
                        tokenizer.pad_token_id = it->second;
                    }
                } else if (cfg_key == "chat_template") {
                    tokenizer.chat_template = json_parser::parse_string(cp, cend);
                } else {
                    json_parser::skip_value(cp, cend);
                }

                json_parser::skip_ws(cp, cend);
                if (cp < cend && *cp == ',') ++cp;
            }
        }
    }

    // Fallback: read config.json token IDs (e.g., gpt-oss uses eos_token_id=<|return|>)
    std::filesystem::path model_config_path = std::filesystem::path(model_dir) / "config.json";
    std::ifstream model_config_file(model_config_path);
    if (model_config_file.is_open()) {
        std::string cfg_content((std::istreambuf_iterator<char>(model_config_file)),
                                 std::istreambuf_iterator<char>());
        int32_t token_id = -1;
        if (tokenizer.eos_token_id < 0 &&
            extract_config_token_id(cfg_content, "eos_token_id", token_id)) {
            if (token_id >= 0 && token_id < static_cast<int32_t>(tokenizer.vocab.size())) {
                tokenizer.eos_token_id = token_id;
            }
        }
        token_id = -1;
        if (tokenizer.bos_token_id < 0 &&
            extract_config_token_id(cfg_content, "bos_token_id", token_id)) {
            if (token_id >= 0 && token_id < static_cast<int32_t>(tokenizer.vocab.size())) {
                tokenizer.bos_token_id = token_id;
            }
        }
        token_id = -1;
        if (tokenizer.pad_token_id < 0 &&
            extract_config_token_id(cfg_content, "pad_token_id", token_id)) {
            if (token_id >= 0 && token_id < static_cast<int32_t>(tokenizer.vocab.size())) {
                tokenizer.pad_token_id = token_id;
            }
        }
    }

    // Debug: Print tokenizer info
    STCPP_DEBUG_LOG("[DEBUG] load_tokenizer: vocab_size=%zu, bos_id=%d, eos_id=%d, pad_id=%d\n",
                    tokenizer.vocab.size(), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id);

    return true;
}

static std::string codepoint_to_utf8(uint32_t codepoint) {
    std::string result;
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        result.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return result;
}

static const std::array<uint32_t, 256>& gpt2_byte_to_unicode_codepoints() {
    static const std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> out{};
        std::vector<int> bs;
        bs.reserve(256);
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            out[static_cast<size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
        }
        return out;
    }();
    return table;
}

static const std::array<int32_t, 512>& gpt2_unicode_to_byte_map() {
    static const std::array<int32_t, 512> table = []() {
        std::array<int32_t, 512> out{};
        out.fill(-1);
        const auto& b2u = gpt2_byte_to_unicode_codepoints();
        for (size_t b = 0; b < b2u.size(); ++b) {
            const uint32_t cp = b2u[b];
            if (cp < out.size()) {
                out[cp] = static_cast<int32_t>(b);
            }
        }
        return out;
    }();
    return table;
}

/* GPT-2 byte encoder table - maps bytes to Unicode characters */
static std::string byte_to_unicode(unsigned char b) {
    const auto& table = gpt2_byte_to_unicode_codepoints();
    return codepoint_to_utf8(table[b]);
}

/* Convert text to GPT-2 byte-level encoding */
static std::string text_to_gpt2_bytes(const std::string& text) {
    std::string result;
    for (unsigned char c : text) {
        result += byte_to_unicode(c);
    }
    return result;
}

/* Reverse mapping: Unicode codepoint to original byte */
static int unicode_to_byte(int codepoint) {
    const auto& table = gpt2_unicode_to_byte_map();
    if (codepoint >= 0 && static_cast<size_t>(codepoint) < table.size()) {
        return table[static_cast<size_t>(codepoint)];
    }
    return -1;
}

/* Convert GPT-2 byte-level encoding back to original text (hybrid mode)
 * Only converts GPT-2 specific codepoints (U+0100-U+01A1), preserves other UTF-8 as-is.
 * This handles hybrid tokenizers like Qwen2 that mix GPT-2 encoding with regular UTF-8.
 */
static std::string gpt2_bytes_to_text(const std::string& encoded) {
    std::string result;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(encoded.data());
    const unsigned char* end = p + encoded.size();

    while (p < end) {
        int codepoint;
        const unsigned char* char_start = p;
        size_t char_len = 1;

        // Decode UTF-8 to get codepoint
        if ((*p & 0x80) == 0) {
            // ASCII
            codepoint = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            // 2-byte UTF-8
            codepoint = ((*p & 0x1F) << 6) | (*(p + 1) & 0x3F);
            char_len = 2;
            p += 2;
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            // 3-byte UTF-8
            codepoint = ((*p & 0x0F) << 12) | ((*(p + 1) & 0x3F) << 6) | (*(p + 2) & 0x3F);
            char_len = 3;
            p += 3;
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            // 4-byte UTF-8
            codepoint = ((*p & 0x07) << 18) | ((*(p + 1) & 0x3F) << 12) |
                        ((*(p + 2) & 0x3F) << 6) | (*(p + 3) & 0x3F);
            char_len = 4;
            p += 4;
        } else {
            // Invalid UTF-8, skip byte
            ++p;
            continue;
        }

        // Convert GPT-2 encoded codepoint to original byte if it exists
        int byte_val = unicode_to_byte(codepoint);
        if (byte_val >= 0 && byte_val <= 255) {
            result += static_cast<char>(byte_val);
        } else {
            // Not a GPT-2 encoded character - preserve as-is (e.g., Chinese, emoji)
            result.append(reinterpret_cast<const char*>(char_start), char_len);
        }
    }

    return result;
}

static std::string merge_pair_key(const std::string& first, const std::string& second) {
    std::string key;
    key.reserve(first.size() + 1 + second.size());
    key.append(first);
    key.push_back('\x1f');
    key.append(second);
    return key;
}

static std::vector<std::string> pretokenize_segment(
    const TokenizerImpl& tokenizer,
    const std::string& segment,
    bool& already_byte_encoded
) {
    already_byte_encoded = false;
    if (tokenizer.pretokenizer_regex.empty()) {
        return {segment};
    }

    try {
        std::string input = segment;
        if (tokenizer.pretokenizer_byte_encoded) {
            input = text_to_gpt2_bytes(segment);
            already_byte_encoded = true;
        }
        return ::unicode_regex_split(input, {tokenizer.pretokenizer_regex});
    } catch (const std::exception& ex) {
        STCPP_DEBUG_LOG("[DEBUG] tokenize: pretokenizer regex failed: %s\n", ex.what());
        already_byte_encoded = false;
        return {segment};
    }
}

/* BPE tokenize a segment (no special tokens) */
static void bpe_tokenize_segment(
    const TokenizerImpl& tokenizer,
    const std::string& segment,
    std::vector<int32_t>& tokens,
    bool already_byte_encoded
) {
    if (segment.empty()) return;

    // Convert text to GPT-2 byte encoding unless already encoded
    std::string encoded = already_byte_encoded ? segment : text_to_gpt2_bytes(segment);

    // Split into individual byte-encoded characters (UTF-8 aware)
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < encoded.size()) {
        size_t char_len = 1;
        unsigned char c = encoded[i];
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len <= encoded.size()) {
            chars.push_back(encoded.substr(i, char_len));
        }
        i += char_len;
    }

    STCPP_DEBUG_LOG("[DEBUG] tokenize: split into %zu chars\n", chars.size());

    if (tokenizer.ignore_merges) {
        int found = 0, not_found = 0, byte_fallback = 0;
        for (const auto& tok : chars) {
            auto it = tokenizer.vocab_to_id.find(tok);
            if (it != tokenizer.vocab_to_id.end()) {
                tokens.push_back(it->second);
                found++;
            } else {
                not_found++;
                // Try byte fallback
                for (unsigned char byte : tok) {
                    std::string byte_token = "<0x" +
                        std::string(1, "0123456789ABCDEF"[byte >> 4]) +
                        std::string(1, "0123456789ABCDEF"[byte & 0xF]) + ">";
                    auto byte_it = tokenizer.vocab_to_id.find(byte_token);
                    if (byte_it != tokenizer.vocab_to_id.end()) {
                        tokens.push_back(byte_it->second);
                        byte_fallback++;
                    }
                }
            }
        }
        STCPP_DEBUG_LOG("[DEBUG] tokenize: ignore_merges=true, found=%d, not_found=%d, byte_fallback=%d\n",
                        found, not_found, byte_fallback);
        return;
    }

    if (!tokenizer.merge_ranks.empty()) {
        while (chars.size() >= 2) {
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            size_t best_idx = chars.size();
            for (size_t j = 0; j + 1 < chars.size(); ++j) {
                auto it = tokenizer.merge_ranks.find(merge_pair_key(chars[j], chars[j + 1]));
                if (it != tokenizer.merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = j;
                }
            }

            if (best_idx >= chars.size()) {
                break;
            }

            const std::string first = chars[best_idx];
            const std::string second = chars[best_idx + 1];
            std::vector<std::string> new_chars;
            new_chars.reserve(chars.size());
            for (size_t j = 0; j < chars.size(); ) {
                if (j + 1 < chars.size() && chars[j] == first && chars[j + 1] == second) {
                    new_chars.push_back(first + second);
                    j += 2;
                } else {
                    new_chars.push_back(chars[j]);
                    j += 1;
                }
            }
            chars = std::move(new_chars);
        }
    }

    STCPP_DEBUG_LOG("[DEBUG] tokenize: after BPE, %zu pieces\n", chars.size());

    // Look up token IDs
    int found = 0, not_found = 0, byte_fallback = 0;
    for (const auto& tok : chars) {
        auto it = tokenizer.vocab_to_id.find(tok);
        if (it != tokenizer.vocab_to_id.end()) {
            tokens.push_back(it->second);
            found++;
        } else {
            not_found++;
            // Try byte fallback
            for (unsigned char byte : tok) {
                std::string byte_token = "<0x" +
                    std::string(1, "0123456789ABCDEF"[byte >> 4]) +
                    std::string(1, "0123456789ABCDEF"[byte & 0xF]) + ">";
                auto byte_it = tokenizer.vocab_to_id.find(byte_token);
                if (byte_it != tokenizer.vocab_to_id.end()) {
                    tokens.push_back(byte_it->second);
                    byte_fallback++;
                }
            }
        }
    }

    STCPP_DEBUG_LOG("[DEBUG] tokenize: bpe lookup found=%d, not_found=%d, byte_fallback=%d\n",
                    found, not_found, byte_fallback);
}

/* Simple BPE tokenization with special token handling */
bool tokenize(
    const TokenizerImpl& tokenizer,
    const std::string& text,
    std::vector<int32_t>& tokens,
    bool add_bos,
    std::string& error
) {
    STCPP_DEBUG_LOG("[DEBUG] tokenize: entered, text_len=%zu, vocab_size=%zu, merges=%zu, special_tokens=%zu, ignore_merges=%s\n",
                    text.size(), tokenizer.vocab.size(), tokenizer.merges.size(), tokenizer.special_tokens.size(),
                    tokenizer.ignore_merges ? "true" : "false");

    tokens.clear();

    // Add BOS token if requested
    if (add_bos && tokenizer.bos_token_id >= 0) {
        tokens.push_back(tokenizer.bos_token_id);
        STCPP_DEBUG_LOG("[DEBUG] tokenize: added BOS token %d\n", tokenizer.bos_token_id);
    }

    if (text.empty()) {
        STCPP_DEBUG_LOG("[DEBUG] tokenize: empty text, returning\n");
        return true;
    }

    // Process text, extracting special tokens first
    size_t pos = 0;
    while (pos < text.size()) {
        bool found_special = false;

        // Try to match special tokens at current position
        for (const auto& special : tokenizer.special_tokens) {
            if (pos + special.size() <= text.size() &&
                text.compare(pos, special.size(), special) == 0) {
                // Found a special token
                auto it = tokenizer.vocab_to_id.find(special);
                if (it != tokenizer.vocab_to_id.end()) {
                    tokens.push_back(it->second);
                    STCPP_DEBUG_LOG("[DEBUG] tokenize: found special token '%s' -> %d\n",
                                    special.c_str(), it->second);
                }
                pos += special.size();
                found_special = true;
                break;
            }
        }

        if (!found_special) {
            // Find the next special token position (or end of text)
            size_t next_special_pos = text.size();
            for (const auto& special : tokenizer.special_tokens) {
                size_t sp_pos = text.find(special, pos);
                if (sp_pos != std::string::npos && sp_pos < next_special_pos) {
                    next_special_pos = sp_pos;
                }
            }

            // Extract segment between current position and next special token
            if (next_special_pos > pos) {
                std::string segment = text.substr(pos, next_special_pos - pos);
                bool already_byte_encoded = false;
                auto pieces = pretokenize_segment(tokenizer, segment, already_byte_encoded);
                for (const auto& piece : pieces) {
                    if (piece.empty()) continue;
                    bpe_tokenize_segment(tokenizer, piece, tokens, already_byte_encoded);
                }
                pos = next_special_pos;
            }
        }
    }

    STCPP_DEBUG_LOG("[DEBUG] tokenize: final_tokens=%zu\n", tokens.size());
    if (tokens.size() > 0 && stcpp_debug_enabled()) {
        fprintf(stderr, "[DEBUG] tokenize: first 5 tokens: ");
        for (size_t i = 0; i < std::min(tokens.size(), static_cast<size_t>(5)); i++) {
            fprintf(stderr, "%d ", tokens[i]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    return true;
}

/* Detect if tokenizer uses GPT-2 byte-level encoding */
static bool uses_gpt2_byte_encoding(const TokenizerImpl& tokenizer) {
    int cached = tokenizer.uses_gpt2_byte_encoding.load(std::memory_order_acquire);
    if (cached != -1) {
        return cached == 1;
    }
    // Check for GPT-2 style tokens like "Ġ" (U+0120 = space in GPT-2 encoding)
    // Qwen2/LLama3 use different encoding where tokens are plain UTF-8
    //
    // GPT-2 encoding: " hello" is stored as "Ġhello" where Ġ = U+0120
    // Qwen2 encoding: tokens are stored as-is in UTF-8
    //
    // Detection: if vocab contains tokens starting with U+0100-U+01FF range
    // (GPT-2 byte encoding), we need GPT-2 decoding. Otherwise, use direct UTF-8.

    // Check first 1000 tokens for GPT-2 encoding markers
    size_t gpt2_marker_count = 0;
    size_t checked = 0;
    for (size_t i = 0; i < std::min(tokenizer.vocab.size(), (size_t)1000); i++) {
        const std::string& token = tokenizer.vocab[i];
        if (token.empty()) continue;
        checked++;

        // Check for 2-byte UTF-8 sequences in U+0100-U+01FF range
        // These are encoded as 0xC4 0x80-0xBF (U+0100-U+013F) or 0xC5 0x80-0xBF (U+0140-U+017F)
        // or 0xC6 0x80-0xBF (U+0180-U+01BF) or 0xC7 0x80-0x81 (U+01C0-U+01C1, but we go to U+01FF)
        for (size_t j = 0; j + 1 < token.size(); j++) {
            unsigned char c1 = static_cast<unsigned char>(token[j]);
            unsigned char c2 = static_cast<unsigned char>(token[j + 1]);
            // Check for C4/C5/C6/C7 prefix (covers U+0100-U+01FF)
            if ((c1 == 0xC4 || c1 == 0xC5 || c1 == 0xC6 || c1 == 0xC7) &&
                (c2 & 0xC0) == 0x80) {
                gpt2_marker_count++;
                break;  // Count once per token
            }
        }
    }

    // If more than 10% of checked tokens have GPT-2 markers, assume GPT-2 encoding
    bool uses_gpt2 = (checked > 0) && (gpt2_marker_count * 10 > checked);
    int expected = -1;
    if (!tokenizer.uses_gpt2_byte_encoding.compare_exchange_strong(
            expected,
            uses_gpt2 ? 1 : 0,
            std::memory_order_release,
            std::memory_order_relaxed)) {
        uses_gpt2 = tokenizer.uses_gpt2_byte_encoding.load(std::memory_order_acquire) == 1;
    }
    STCPP_DEBUG_LOG("[DEBUG] uses_gpt2_byte_encoding: checked=%zu, gpt2_markers=%zu, uses_gpt2=%d\n",
                    checked, gpt2_marker_count, uses_gpt2);
    return uses_gpt2;
}

/* Detokenize token IDs to text */
bool detokenize(
    const TokenizerImpl& tokenizer,
    const std::vector<int32_t>& tokens,
    std::string& result,
    std::string& error
) {
    result.clear();
    std::string accumulated;

    STCPP_DEBUG_LOG("[DEBUG] detokenize: %zu tokens, vocab_size=%zu\n",
                    tokens.size(), tokenizer.vocab.size());
    if (stcpp_debug_enabled()) {
        fprintf(stderr, "[DEBUG] detokenize: first 10 token IDs: ");
        for (size_t i = 0; i < std::min(tokens.size(), static_cast<size_t>(10)); i++) {
            fprintf(stderr, "%d ", tokens[i]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    bool use_gpt2_decoding = uses_gpt2_byte_encoding(tokenizer);

    int skipped_negative = 0;
    int skipped_oob = 0;
    int skipped_special = 0;
    int processed = 0;

    STCPP_DEBUG_LOG("[DEBUG] detokenize: %zu tokens, vocab_size=%zu\n",
                    tokens.size(), tokenizer.vocab.size());

    for (int32_t id : tokens) {
        if (id < 0) {
            skipped_negative++;
            continue;  // Skip invalid tokens
        }
        if (id >= static_cast<int32_t>(tokenizer.vocab.size())) {
            skipped_oob++;
            STCPP_DEBUG_LOG("[DEBUG] detokenize: token %d out of vocab bounds (%zu)\n",
                            id, tokenizer.vocab.size());
            continue;
        }

        const std::string& token = tokenizer.vocab[id];

        // Skip special tokens in output
        if (id == tokenizer.bos_token_id ||
            id == tokenizer.eos_token_id ||
            id == tokenizer.pad_token_id) {
            skipped_special++;
            continue;
        }

        // Debug: show first few tokens being processed
        if (processed < 10) {
            STCPP_DEBUG_LOG("[DEBUG] detokenize: id=%d -> token='%s' (len=%zu)\n",
                            id, token.c_str(), token.size());
        }

        // Handle byte tokens like <0xXX> (Llama-style byte fallback)
        if (token.size() == 6 && token.substr(0, 3) == "<0x" && token[5] == '>') {
            char hex[3] = {token[3], token[4], 0};
            unsigned int byte_val;
            if (sscanf(hex, "%x", &byte_val) == 1) {
                if (use_gpt2_decoding) {
                    // Convert byte to GPT-2 Unicode encoding for later decoding
                    accumulated += byte_to_unicode(static_cast<unsigned char>(byte_val));
                } else {
                    // Direct byte output
                    accumulated += static_cast<char>(byte_val);
                }
                processed++;
                continue;
            }
        }

        // Check if token is empty
        if (token.empty()) {
            STCPP_DEBUG_LOG("[DEBUG] detokenize: WARNING id=%d has empty token\n", id);
            continue;
        }

        // Accumulate token
        accumulated += token;
        processed++;
    }

    STCPP_DEBUG_LOG("[DEBUG] detokenize: processed=%d, skipped_neg=%d, skipped_oob=%d, skipped_special=%d\n",
                    processed, skipped_negative, skipped_oob, skipped_special);
    STCPP_DEBUG_LOG("[DEBUG] detokenize: accumulated len=%zu, use_gpt2_decoding=%d\n",
                    accumulated.size(), use_gpt2_decoding);

    if (use_gpt2_decoding) {
        // Convert GPT-2 byte encoding back to original text
        result = gpt2_bytes_to_text(accumulated);
    } else {
        // Direct UTF-8 output (Qwen2, LLama3, etc.)
        result = accumulated;
    }

    STCPP_DEBUG_LOG("[DEBUG] detokenize: final result len=%zu, first 50 chars='%.*s'\n",
                    result.size(), (int)std::min(result.size(), static_cast<size_t>(50)), result.c_str());

    return true;
}

}  // namespace stcpp
