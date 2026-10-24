#include "main.h"

static const char *TAG = "APP_MAIN";

extern "C" void app_main(void) {
    if (!init_gru_model()) {
        ESP_LOGE(TAG, "Не удалось инициализировать модель. Перезагрузка...");
        return;
    }

    //init NVS
    esp_err_t ret = nvs_flash_init();
    if(ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ret = nvs_flash_erase();
        ESP_LOGI(TAG, "nvs_flash_erase: 0x%04x", ret);
        ret = nvs_flash_init();
        ESP_LOGI(TAG, "nvs_flash_init: 0x%04x", ret);
    }
    ESP_LOGI(TAG, "nvs_flash_init: 0x%04x", ret);

    ret = esp_netif_init();
    ESP_LOGI(TAG, "esp_netif_init: %d", ret);

    ret = esp_event_loop_create_default();
    ESP_LOGI(TAG, "esp_event_loop_create_default: %d", ret);

    ret = wifi_init_sta();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "wifi_init_sta failed: %s", esp_err_to_name(ret));
        return;
    }

    // while (true) {
    //     float prediction = predict_gru(test_input);
    //     ESP_LOGI(TAG, "Предсказанное значение: %.4f", prediction);
    //     vTaskDelay(pdMS_TO_TICKS(1000));
    // }
}
