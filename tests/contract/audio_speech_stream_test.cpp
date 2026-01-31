#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "api/audio_endpoints.h"
#include "api/http_server.h"
#include "api/node_endpoints.h"
#include "api/openai_endpoints.h"
#include "core/audio_manager.h"
#include "core/inference_engine.h"
#include "core/onnx_tts_manager.h"
#include "models/model_registry.h"
#include "runtime/state.h"
#include "utils/config.h"

namespace fs = std::filesystem;

namespace {

std::string makeWavData() {
    const int sample_rate = 16000;
    const int channels = 1;
    const int bits_per_sample = 16;
    const std::vector<int16_t> samples = {0, 1000, -1000, 0};

    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t file_size = 44 + data_size;

    std::vector<uint8_t> wav;
    wav.reserve(file_size);

    auto append_u16 = [&wav](uint16_t value) {
        wav.push_back(static_cast<uint8_t>(value & 0xFF));
        wav.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    };
    auto append_u32 = [&wav](uint32_t value) {
        wav.push_back(static_cast<uint8_t>(value & 0xFF));
        wav.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
        wav.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
        wav.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
    };

    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    append_u32(file_size - 8);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(static_cast<uint16_t>(channels));
    append_u32(static_cast<uint32_t>(sample_rate));
    append_u32(static_cast<uint32_t>(sample_rate * channels * (bits_per_sample / 8)));
    append_u16(static_cast<uint16_t>(channels * (bits_per_sample / 8)));
    append_u16(static_cast<uint16_t>(bits_per_sample));
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(data_size);

    for (int16_t sample : samples) {
        wav.push_back(static_cast<uint8_t>(sample & 0xFF));
        wav.push_back(static_cast<uint8_t>((sample >> 8) & 0xFF));
    }

    return std::string(reinterpret_cast<const char*>(wav.data()), wav.size());
}

class TempDir {
public:
    TempDir() {
        base_ = fs::temp_directory_path() / "xllm-audio-contract-XXXXXX";
        std::string tmpl = base_.string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        char* created = mkdtemp(buf.data());
        base_ = created ? fs::path(created) : fs::temp_directory_path();
        fs::create_directories(base_);
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(base_, ec);
    }

    const fs::path& path() const { return base_; }

private:
    fs::path base_;
};

class AudioSpeechContractFixture : public ::testing::Test {
protected:
    void SetUp() override {
        xllm::set_ready(true);

        // Minimal server components required by HttpServer
        registry_.setModels({"dummy"});
        server_ = std::make_unique<xllm::HttpServer>(18096, openai_, node_);

        // Provide deterministic TTS output for contract tests
        tts_manager_.setSynthesizeHookForTest(
            [](const std::string&, const std::string&, const xllm::SpeechParams&) {
                xllm::SpeechResult result;
                result.success = true;
                result.format = "wav";
                result.sample_rate = 22050;
                result.channels = 1;
                result.bits_per_sample = 16;
                result.audio_data.resize(48 * 1024);
                for (size_t i = 0; i < result.audio_data.size(); ++i) {
                    result.audio_data[i] = static_cast<uint8_t>(i % 251);
                }
                return result;
            });

        audio_manager_.setTranscribeHookForTest(
            [](const std::string&, const std::vector<float>& audio, int sample_rate,
               const xllm::TranscriptionParams&) {
                xllm::TranscriptionResult result;
                result.success = true;
                result.text = "hello from asr";
                result.language = "en";
                if (sample_rate > 0) {
                    result.duration_seconds = static_cast<double>(audio.size()) / sample_rate;
                } else {
                    result.duration_seconds = 1.0;
                }
                return result;
            });

        audio_endpoints_ = std::make_unique<xllm::AudioEndpoints>(audio_manager_, tts_manager_);
        audio_endpoints_->registerRoutes(server_->getServer());
        server_->start();
    }

    void TearDown() override {
        server_->stop();
    }

    TempDir tmp_;
    xllm::AudioManager audio_manager_{tmp_.path().string()};
    xllm::OnnxTtsManager tts_manager_{tmp_.path().string()};

    xllm::ModelRegistry registry_;
    xllm::InferenceEngine engine_{};
    xllm::NodeConfig config_{};
    xllm::OpenAIEndpoints openai_{registry_, engine_, config_, xllm::GpuBackend::Cpu};
    xllm::NodeEndpoints node_{};

    std::unique_ptr<xllm::HttpServer> server_;
    std::unique_ptr<xllm::AudioEndpoints> audio_endpoints_;
};

}  // namespace

// T301: TTSストリーミングの契約テスト
TEST_F(AudioSpeechContractFixture, SpeechEndpointStreamsAudioWhenRequested) {
    httplib::Client cli("127.0.0.1", 18096);
    const std::string body = R"({
        "model": "vibevoice",
        "input": "hello streaming",
        "response_format": "wav",
        "stream": true
    })";

    auto res = cli.Post("/v1/audio/speech", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "audio/wav");

    // Contract: streaming still yields the complete audio payload
    EXPECT_GE(res->body.size(), 48u * 1024u);
}

TEST_F(AudioSpeechContractFixture, SpeechEndpointReturnsAudioWithoutStreaming) {
    httplib::Client cli("127.0.0.1", 18096);
    const std::string body = R"({
        "model": "vibevoice",
        "input": "hello",
        "response_format": "wav"
    })";

    auto res = cli.Post("/v1/audio/speech", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "audio/wav");
    EXPECT_GE(res->body.size(), 48u * 1024u);
}

TEST_F(AudioSpeechContractFixture, TranscriptionsReturnsJson) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", makeWavData(), "sample.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
        {"language", "en", "", ""},
        {"response_format", "json", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "application/json");

    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["text"], "hello from asr");
    EXPECT_EQ(j["language"], "en");
}

TEST_F(AudioSpeechContractFixture, TranscriptionsReturnsText) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", makeWavData(), "sample.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
        {"response_format", "text", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "text/plain");
    EXPECT_NE(res->body.find("hello from asr"), std::string::npos);
}

TEST_F(AudioSpeechContractFixture, TranscriptionsReturnsSrt) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", makeWavData(), "sample.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
        {"response_format", "srt", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "text/plain");
    EXPECT_NE(res->body.find("00:00:00"), std::string::npos);
    EXPECT_NE(res->body.find("hello from asr"), std::string::npos);
}

TEST_F(AudioSpeechContractFixture, TranscriptionsReturnsVtt) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", makeWavData(), "sample.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
        {"response_format", "vtt", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "text/vtt");
    EXPECT_NE(res->body.find("WEBVTT"), std::string::npos);
    EXPECT_NE(res->body.find("hello from asr"), std::string::npos);
}

TEST_F(AudioSpeechContractFixture, TranscriptionsRejectsMissingFile) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"model", "whisper-1", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["error"]["code"], "missing_file");
}

TEST_F(AudioSpeechContractFixture, TranscriptionsRejectsMissingModel) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", makeWavData(), "sample.wav", "audio/wav"},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["error"]["code"], "missing_model");
}

TEST_F(AudioSpeechContractFixture, TranscriptionsRejectsEmptyFile) {
    httplib::Client cli("127.0.0.1", 18096);
    httplib::MultipartFormDataItems items = {
        {"file", "", "sample.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
    };

    auto res = cli.Post("/v1/audio/transcriptions", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["error"]["code"], "empty_file");
}
