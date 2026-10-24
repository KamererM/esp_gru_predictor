#ifndef GRU_PREDICTOR_H
#define GRU_PREDICTOR_H

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"

constexpr std::size_t kGruTimesteps = 32;

struct GruPrediction {
    float prediction_kwh;
    int8_t input_int8[kGruTimesteps];
    int8_t output_int8;
};

struct GruRuntimeMetadata {
    std::size_t arena_capacity_bytes;
    std::size_t arena_used_bytes;
    std::size_t free_heap_before_allocate;
    std::size_t free_heap_after_allocate;
    bool arena_in_psram;
    float input_scale;
    int input_zero_point;
    float output_scale;
    int output_zero_point;
};

bool init_gru_model();
bool predict_gru(const float input_kwh[kGruTimesteps], GruPrediction* result);
const GruRuntimeMetadata& get_gru_runtime_metadata();

#endif
