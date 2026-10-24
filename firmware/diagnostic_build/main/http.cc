#include "http.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cJSON.h"
#include "deployment_metadata.h"
#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "gru_predictor.h"
#include "model_data.h"
#include "resource_probe.h"
#include "preprocessing_constants.h"
#include "psa/crypto.h"
#include "sdkconfig.h"

namespace {

constexpr char kTag[] = "http";
constexpr std::size_t kMaxBodyBytes = 8192;
constexpr std::size_t kResponseCacheEntries = 32;
constexpr char kProtocolVersion[] = "gru-esp32-v1";

struct CachedResponse {
    bool valid = false;
    std::string run_id;
    std::string timestamp_ns;
    std::string request_id;
    GruPrediction result{};
    int64_t execution_time_us = 0;
    std::array<float, kGruTimesteps> input_kwh{};
};

std::array<CachedResponse, kResponseCacheEntries> response_cache{};
std::size_t next_cache_slot = 0;

std::string to_hex(const unsigned char* bytes, std::size_t length) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output(length * 2, '0');
    for (std::size_t i = 0; i < length; ++i) {
        output[2 * i] = digits[(bytes[i] >> 4) & 0x0f];
        output[2 * i + 1] = digits[bytes[i] & 0x0f];
    }
    return output;
}

std::string sha256_memory(const void* data, std::size_t length) {
    std::array<unsigned char, PSA_HASH_LENGTH(PSA_ALG_SHA_256)> digest{};
    std::size_t digest_length = 0;
    const psa_status_t status = psa_hash_compute(
        PSA_ALG_SHA_256,
        static_cast<const uint8_t*>(data),
        length,
        digest.data(),
        digest.size(),
        &digest_length
    );
    if (status != PSA_SUCCESS || digest_length != digest.size()) {
        ESP_LOGE(kTag, "PSA SHA-256 failed: %ld", static_cast<long>(status));
        return "ERROR";
    }
    return to_hex(digest.data(), digest.size());
}

std::string running_app_elf_sha256() {
    const esp_app_desc_t* description = esp_app_get_description();
    if (description == nullptr) {
        return "ERROR";
    }
    return to_hex(description->app_elf_sha256, sizeof(description->app_elf_sha256));
}

const CachedResponse* find_cached(const std::string& request_id) {
    for (const CachedResponse& entry : response_cache) {
        if (entry.valid && entry.request_id == request_id) {
            return &entry;
        }
    }
    return nullptr;
}

void store_cached(const CachedResponse& response) {
    response_cache[next_cache_slot] = response;
    next_cache_slot = (next_cache_slot + 1) % response_cache.size();
}

esp_err_t send_json(httpd_req_t* request, cJSON* root) {
    char* text = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (text == nullptr) {
        httpd_resp_send_500(request);
        return ESP_FAIL;
    }
    httpd_resp_set_type(request, "application/json");
    const esp_err_t result = httpd_resp_send(request, text, HTTPD_RESP_USE_STRLEN);
    cJSON_free(text);
    return result;
}

esp_err_t send_service_unavailable(httpd_req_t* request, const char* message) {
    httpd_resp_set_status(request, "503 Service Unavailable");
    httpd_resp_set_type(request, "text/plain");
    return httpd_resp_sendstr(request, message);
}

std::string float32_bits_hex(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    char output[9];
    std::snprintf(
        output, sizeof(output), "%08lx", static_cast<unsigned long>(bits)
    );
    return output;
}

esp_err_t send_prediction_response(
    httpd_req_t* request,
    const CachedResponse& value
) {
    cJSON* root = cJSON_CreateObject();
    if (root == nullptr) {
        httpd_resp_send_500(request);
        return ESP_FAIL;
    }
    cJSON_AddStringToObject(root, "run_id", value.run_id.c_str());
    cJSON_AddStringToObject(root, "timestamp_ns", value.timestamp_ns.c_str());
    cJSON_AddStringToObject(root, "request_id", value.request_id.c_str());
    cJSON_AddNumberToObject(root, "prediction", value.result.prediction_kwh);
    cJSON_AddNumberToObject(
        root, "execution_time_us", value.execution_time_us
    );
    cJSON* quantized = cJSON_AddArrayToObject(root, "input_int8");
    for (std::size_t i = 0; i < kGruTimesteps; ++i) {
        cJSON_AddItemToArray(
            quantized, cJSON_CreateNumber(value.result.input_int8[i])
        );
    }
    cJSON_AddNumberToObject(root, "output_int8", value.result.output_int8);
    return send_json(request, root);
}

bool required_string(cJSON* root, const char* name, std::string* destination) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (!cJSON_IsString(item) || item->valuestring == nullptr) {
        return false;
    }
    *destination = item->valuestring;
    return !destination->empty();
}

esp_err_t metadata_get_handler(httpd_req_t* request) {
    if (!kDeploymentMetadataConfigured || kPreprocessingConstantsArePlaceholder) {
        return send_service_unavailable(
            request,
            "Deployment metadata or preprocessing constants are not configured"
        );
    }
    const GruRuntimeMetadata& runtime = get_gru_runtime_metadata();
    uint32_t flash_bytes = 0;
    if (esp_flash_get_physical_size(esp_flash_default_chip, &flash_bytes)
        != ESP_OK) {
        flash_bytes = 0;
    }
    esp_chip_info_t chip_info{};
    esp_chip_info(&chip_info);
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "protocol_version", kProtocolVersion);
    cJSON_AddStringToObject(root, "run_id", kRunId);
    cJSON_AddStringToObject(
        root,
        "model_sha256",
        sha256_memory(optimized_model_tflite, optimized_model_tflite_len).c_str()
    );
    cJSON_AddNumberToObject(root, "model_bytes", optimized_model_tflite_len);
    cJSON_AddStringToObject(root, "model_data_cc_sha256", kModelDataCcSha256);
    cJSON_AddStringToObject(
        root,
        "preprocessing_constants_h_sha256",
        kPreprocessingConstantsSha256
    );
    cJSON_AddStringToObject(
        root, "training_mean_kwh_float32_hex", kTrainingMeanFloat32Hex
    );
    cJSON_AddStringToObject(
        root, "training_std_kwh_float32_hex", kTrainingStdFloat32Hex
    );
    cJSON_AddNumberToObject(root, "input_scale", runtime.input_scale);
    cJSON_AddStringToObject(
        root,
        "input_scale_float32_bits",
        float32_bits_hex(runtime.input_scale).c_str()
    );
    cJSON_AddNumberToObject(
        root, "input_zero_point", runtime.input_zero_point
    );
    cJSON_AddNumberToObject(root, "output_scale", runtime.output_scale);
    cJSON_AddStringToObject(
        root,
        "output_scale_float32_bits",
        float32_bits_hex(runtime.output_scale).c_str()
    );
    cJSON_AddNumberToObject(
        root, "output_zero_point", runtime.output_zero_point
    );
    cJSON_AddStringToObject(root, "timing_field", "execution_time_us");
    cJSON_AddStringToObject(root, "firmware_commit", kFirmwareCommit);
    cJSON_AddStringToObject(root, "firmware_build_id", kFirmwareBuildId);
    cJSON_AddStringToObject(root, "board_or_module", kBoardOrModule);
    cJSON_AddStringToObject(root, "idf_target", kIdfTarget);
    cJSON_AddNumberToObject(root, "cpu_frequency_mhz", kCpuFrequencyMhz);
    cJSON_AddNumberToObject(root, "flash_bytes", flash_bytes);
    cJSON_AddNumberToObject(
        root, "psram_bytes", heap_caps_get_total_size(MALLOC_CAP_SPIRAM)
    );
    cJSON_AddNumberToObject(root, "chip_model", chip_info.model);
    cJSON_AddNumberToObject(root, "chip_revision", chip_info.revision);
    cJSON_AddStringToObject(root, "tflm_revision", kTflmRevision);
    cJSON_AddStringToObject(
        root, "operator_resolver_id", kOperatorResolverId
    );
    cJSON_AddNumberToObject(
        root, "arena_used_bytes", runtime.arena_used_bytes
    );
    cJSON_AddNumberToObject(
        root, "arena_capacity_bytes", runtime.arena_capacity_bytes
    );
    cJSON_AddBoolToObject(root, "arena_in_psram", runtime.arena_in_psram);
    cJSON_AddStringToObject(
        root, "heap_measurement_caps", "MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT"
    );
    cJSON_AddNumberToObject(
        root,
        "min_internal_free_heap_bytes",
        CONFIG_GRU_MIN_FREE_HEAP_AFTER_MODEL
    );
    cJSON_AddNumberToObject(
        root,
        "free_heap_before_allocate",
        runtime.free_heap_before_allocate
    );
    cJSON_AddNumberToObject(
        root,
        "free_heap_after_allocate",
        runtime.free_heap_after_allocate
    );
    cJSON_AddStringToObject(
        root, "app_elf_sha256", running_app_elf_sha256().c_str()
    );
    cJSON_AddStringToObject(root, "sdkconfig_sha256", kSdkconfigSha256);
    cJSON_AddStringToObject(
        root, "resolver_source_sha256", kResolverSourceSha256
    );
    add_resource_probe(root);
    return send_json(request, root);
}

esp_err_t post_data_handler(httpd_req_t* request) {
    if (!kDeploymentMetadataConfigured || kPreprocessingConstantsArePlaceholder) {
        return send_service_unavailable(request, "Deployment is not configured");
    }
    if (request->content_len <= 0
        || request->content_len > static_cast<int>(kMaxBodyBytes)) {
        httpd_resp_send_err(
            request, HTTPD_400_BAD_REQUEST, "Invalid body length"
        );
        return ESP_FAIL;
    }

    std::string body(request->content_len, '\0');
    int received_total = 0;
    while (received_total < request->content_len) {
        const int received = httpd_req_recv(
            request,
            body.data() + received_total,
            request->content_len - received_total
        );
        if (received == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (received <= 0) {
            httpd_resp_send_500(request);
            return ESP_FAIL;
        }
        received_total += received;
    }

    cJSON* root = cJSON_ParseWithLength(body.data(), body.size());
    if (root == nullptr) {
        httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }
    std::string run_id;
    std::string timestamp_ns;
    std::string request_id;
    cJSON* size = cJSON_GetObjectItemCaseSensitive(root, "size");
    cJSON* data = cJSON_GetObjectItemCaseSensitive(root, "data");
    cJSON* diagnostic = cJSON_GetObjectItemCaseSensitive(root, "diagnostic");
    const bool valid_header =
        required_string(root, "run_id", &run_id)
        && required_string(root, "timestamp_ns", &timestamp_ns)
        && required_string(root, "request_id", &request_id)
        && cJSON_IsNumber(size)
        && size->valuedouble == static_cast<double>(kGruTimesteps)
        && cJSON_IsTrue(diagnostic)
        && cJSON_IsArray(data)
        && cJSON_GetArraySize(data) == static_cast<int>(kGruTimesteps);
    const bool valid_timestamp =
        !timestamp_ns.empty()
        && timestamp_ns.size() <= 20
        && timestamp_ns.find_first_not_of("0123456789") == std::string::npos;
    if (!valid_header || !valid_timestamp || run_id != kRunId
        || request_id.size() > 192) {
        cJSON_Delete(root);
        httpd_resp_send_err(
            request, HTTPD_400_BAD_REQUEST, "Invalid protocol envelope"
        );
        return ESP_FAIL;
    }

    float input[kGruTimesteps];
    for (std::size_t i = 0; i < kGruTimesteps; ++i) {
        cJSON* item = cJSON_GetArrayItem(data, i);
        if (!cJSON_IsNumber(item) || !std::isfinite(item->valuedouble)) {
            cJSON_Delete(root);
            httpd_resp_send_err(
                request, HTTPD_400_BAD_REQUEST, "Non-finite input"
            );
            return ESP_FAIL;
        }
        input[i] = static_cast<float>(item->valuedouble);
        if (!std::isfinite(input[i])) {
            cJSON_Delete(root);
            httpd_resp_send_err(
                request, HTTPD_400_BAD_REQUEST, "Input outside float32 range"
            );
            return ESP_FAIL;
        }
    }
    cJSON_Delete(root);

    if (const CachedResponse* cached = find_cached(request_id);
        cached != nullptr) {
        // The server is single-threaded. Comparing exact float32 values binds a
        // retry to the values that were actually consumed by the interpreter.
        const bool same_identity =
            cached->run_id == run_id
            && cached->timestamp_ns == timestamp_ns
            && std::memcmp(
                   cached->input_kwh.data(),
                   input,
                   sizeof(input)
               ) == 0;
        if (!same_identity) {
            httpd_resp_send_err(
                request, HTTPD_400_BAD_REQUEST, "Request ID collision"
            );
            return ESP_FAIL;
        }
        return send_prediction_response(request, *cached);
    }

    CachedResponse fresh;
    fresh.run_id = run_id;
    fresh.timestamp_ns = timestamp_ns;
    fresh.request_id = request_id;
    std::memcpy(fresh.input_kwh.data(), input, sizeof(input));
    const int64_t start_time = esp_timer_get_time();
    const bool ok = predict_gru(input, &fresh.result);
    fresh.execution_time_us = esp_timer_get_time() - start_time;
    if (!ok || !std::isfinite(fresh.result.prediction_kwh)) {
        httpd_resp_send_500(request);
        return ESP_FAIL;
    }
    fresh.valid = true;
    store_cached(fresh);
    ESP_LOGI(
        kTag,
        "request_id=%s prediction=%f execution_time_us=%lld",
        fresh.request_id.c_str(),
        fresh.result.prediction_kwh,
        static_cast<long long>(fresh.execution_time_us)
    );
    return send_prediction_response(request, fresh);
}

const httpd_uri_t metadata_uri = {
    .uri = "/metadata",
    .method = HTTP_GET,
    .handler = metadata_get_handler,
    .user_ctx = nullptr,
};

const httpd_uri_t data_uri = {
    .uri = "/data",
    .method = HTTP_POST,
    .handler = post_data_handler,
    .user_ctx = nullptr,
};

}  // namespace

httpd_handle_t start_webserver() {
    httpd_handle_t server = nullptr;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.lru_purge_enable = true;
    config.server_port = 8080;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(kTag, "Failed to start HTTP server");
        return nullptr;
    }
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &metadata_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &data_uri));
    return server;
}

void stop_webserver(httpd_handle_t server) {
    if (server != nullptr) {
        httpd_stop(server);
    }
}
