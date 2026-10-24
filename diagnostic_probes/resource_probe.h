#pragma once
// Add #include "resource_probe.h" to http.cc. Inside metadata_get_handler(),
// call add_resource_probe(root) immediately before return send_json(...).
// This reports actual live RAM; never overwrite the original pre-allocation
// readings or interpret a configured guard threshold as measured min-free RAM.
#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static inline void add_resource_probe(cJSON* root) {
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    cJSON_AddStringToObject(root, "resource_probe_version", "gru-resource-v1");
    cJSON_AddStringToObject(root, "runtime_idf_version", esp_get_idf_version());
    cJSON_AddNumberToObject(root, "sample_uptime_us", esp_timer_get_time());
    cJSON_AddNumberToObject(root, "internal_total_bytes",
                           heap_caps_get_total_size(caps));
    cJSON_AddNumberToObject(root, "internal_free_now_bytes",
                           heap_caps_get_free_size(caps));
    cJSON_AddNumberToObject(root, "internal_min_free_since_boot_bytes",
                           heap_caps_get_minimum_free_size(caps));
    cJSON_AddNumberToObject(root, "internal_largest_free_block_bytes",
                           heap_caps_get_largest_free_block(caps));
    cJSON_AddNumberToObject(root, "psram_free_now_bytes",
                           heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    cJSON_AddNumberToObject(root, "psram_min_free_since_boot_bytes",
                           heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    // ESP-IDF reports this high-water mark in bytes; measured task is HTTP.
    cJSON_AddNumberToObject(root, "http_task_min_stack_free_bytes",
                           uxTaskGetStackHighWaterMark(nullptr));
}
