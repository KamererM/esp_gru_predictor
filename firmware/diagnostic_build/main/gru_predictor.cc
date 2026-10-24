#include "gru_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "deployment_metadata.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "model_data.h"
#include "preprocessing_constants.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr std::size_t kTensorArenaSize = CONFIG_GRU_TENSOR_ARENA_BYTES;
constexpr char kTag[] = "GRU_PREDICTOR";

uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
GruRuntimeMetadata runtime_metadata{};

float normalize_float32(float raw_kwh) {
    return (raw_kwh - kTrainingMeanKwh) / kTrainingStdKwh;
}

float denormalize_float32(float normalized) {
    return normalized * kTrainingStdKwh + kTrainingMeanKwh;
}

int8_t quantize_float32(float normalized) {
    const float transformed =
        normalized / runtime_metadata.input_scale
        + static_cast<float>(runtime_metadata.input_zero_point);
    const float rounded = roundf(transformed);
    const float clipped = std::min(127.0f, std::max(-128.0f, rounded));
    return static_cast<int8_t>(clipped);
}

float dequantize_float32(int8_t value) {
    return (
        static_cast<float>(value) - runtime_metadata.output_zero_point
    ) * runtime_metadata.output_scale;
}

bool register_ops(tflite::MicroMutableOpResolver<20>& resolver) {
    if (resolver.AddFullyConnected() != kTfLiteOk) return false;
    if (resolver.AddPack() != kTfLiteOk) return false;
    if (resolver.AddUnpack() != kTfLiteOk) return false;
    if (resolver.AddQuantize() != kTfLiteOk) return false;
    if (resolver.AddDequantize() != kTfLiteOk) return false;
    if (resolver.AddShape() != kTfLiteOk) return false;
    if (resolver.AddStridedSlice() != kTfLiteOk) return false;
    if (resolver.AddFill() != kTfLiteOk) return false;
    if (resolver.AddAdd() != kTfLiteOk) return false;
    if (resolver.AddLogistic() != kTfLiteOk) return false;
    if (resolver.AddSub() != kTfLiteOk) return false;
    if (resolver.AddMul() != kTfLiteOk) return false;
    if (resolver.AddTanh() != kTfLiteOk) return false;
    return true;
}

}  // namespace

bool init_gru_model() {
    const tflite::Model* model = tflite::GetModel(optimized_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "TFLite schema mismatch: %d != %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<20> resolver;
    if (!register_ops(resolver)) {
        ESP_LOGE(kTag, "Failed to register TFLM operators");
        return false;
    }

    runtime_metadata.free_heap_before_allocate =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    tensor_arena = static_cast<uint8_t*>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );
    runtime_metadata.arena_in_psram = tensor_arena != nullptr;
    if (tensor_arena == nullptr) {
        tensor_arena = static_cast<uint8_t*>(
            heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
        );
    }
    if (tensor_arena == nullptr) {
        ESP_LOGE(kTag, "Unable to allocate %zu-byte tensor arena", kTensorArenaSize);
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors failed");
        return false;
    }

    runtime_metadata.arena_capacity_bytes = kTensorArenaSize;
    runtime_metadata.arena_used_bytes = interpreter->arena_used_bytes();
    runtime_metadata.free_heap_after_allocate =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (runtime_metadata.free_heap_after_allocate
        < CONFIG_GRU_MIN_FREE_HEAP_AFTER_MODEL) {
        ESP_LOGE(
            kTag,
            "Only %zu bytes free after model allocation; configured minimum is %d",
            runtime_metadata.free_heap_after_allocate,
            CONFIG_GRU_MIN_FREE_HEAP_AFTER_MODEL
        );
        return false;
    }
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    if (input_tensor == nullptr || output_tensor == nullptr) {
        ESP_LOGE(kTag, "Input/output tensor unavailable");
        return false;
    }
    if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
        ESP_LOGE(kTag, "Model is not INT8 at input/output");
        return false;
    }
    if (input_tensor->dims->size != 3
        || input_tensor->dims->data[0] != 1
        || input_tensor->dims->data[1] != static_cast<int>(kGruTimesteps)
        || input_tensor->dims->data[2] != 1) {
        ESP_LOGE(kTag, "Unexpected input shape");
        return false;
    }

    runtime_metadata.input_scale = input_tensor->params.scale;
    runtime_metadata.input_zero_point = input_tensor->params.zero_point;
    runtime_metadata.output_scale = output_tensor->params.scale;
    runtime_metadata.output_zero_point = output_tensor->params.zero_point;
    if (kDeploymentMetadataConfigured
        && (runtime_metadata.input_scale != kExpectedInputScale
            || runtime_metadata.input_zero_point != kExpectedInputZeroPoint
            || runtime_metadata.output_scale != kExpectedOutputScale
            || runtime_metadata.output_zero_point != kExpectedOutputZeroPoint)) {
        ESP_LOGE(kTag, "Tensor quantization differs from deployment metadata");
        return false;
    }
    ESP_LOGI(kTag, "Model ready; arena used: %zu bytes",
             runtime_metadata.arena_used_bytes);
    return true;
}

bool predict_gru(
    const float input_kwh[kGruTimesteps],
    GruPrediction* result
) {
    if (interpreter == nullptr || input_tensor == nullptr
        || output_tensor == nullptr || result == nullptr) {
        return false;
    }

    for (std::size_t i = 0; i < kGruTimesteps; ++i) {
        const float normalized = normalize_float32(input_kwh[i]);
        result->input_int8[i] = quantize_float32(normalized);
    }
    std::memcpy(
        input_tensor->data.int8,
        result->input_int8,
        kGruTimesteps * sizeof(int8_t)
    );
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "Invoke failed");
        return false;
    }

    result->output_int8 = output_tensor->data.int8[0];
    const float normalized_prediction =
        dequantize_float32(result->output_int8);
    result->prediction_kwh = denormalize_float32(normalized_prediction);
    return std::isfinite(result->prediction_kwh);
}

const GruRuntimeMetadata& get_gru_runtime_metadata() {
    return runtime_metadata;
}
