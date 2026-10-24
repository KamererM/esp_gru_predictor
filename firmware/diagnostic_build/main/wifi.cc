#include "wifi.h"

#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "http.h"
#include "sdkconfig.h"

namespace {

constexpr char kTag[] = "wifi";
constexpr EventBits_t kConnectedBit = BIT0;
constexpr EventBits_t kFailedBit = BIT1;
constexpr int kMaximumRetries = 10;

EventGroupHandle_t event_group = nullptr;
httpd_handle_t server = nullptr;
int retry_count = 0;

void event_handler(
    void*,
    esp_event_base_t event_base,
    int32_t event_id,
    void*
) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
        return;
    }
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (server != nullptr) {
            stop_webserver(server);
            server = nullptr;
        }
        if (retry_count < kMaximumRetries) {
            ++retry_count;
            esp_wifi_connect();
        } else {
            xEventGroupSetBits(event_group, kFailedBit);
        }
        return;
    }
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        retry_count = 0;
        if (server == nullptr) {
            server = start_webserver();
        }
        xEventGroupSetBits(event_group, kConnectedBit);
    }
}

}  // namespace

esp_err_t wifi_init_sta() {
    if (std::strlen(CONFIG_GRU_WIFI_SSID) == 0) {
        ESP_LOGE(kTag, "Wi-Fi SSID is empty; configure it with menuconfig");
        return ESP_ERR_INVALID_ARG;
    }

    event_group = xEventGroupCreate();
    if (event_group == nullptr) {
        return ESP_ERR_NO_MEM;
    }

    esp_netif_create_default_wifi_sta();
    const wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_config));
    ESP_ERROR_CHECK(
        esp_event_handler_register(
            WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, nullptr
        )
    );
    ESP_ERROR_CHECK(
        esp_event_handler_register(
            IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, nullptr
        )
    );

    wifi_config_t wifi_config{};
    // [REDACTED for publication: two lines that copied a hard-coded SSID and password;
    //  they were overwritten by the CONFIG_GRU_WIFI_* values below.]
    std::strncpy(
        reinterpret_cast<char*>(wifi_config.sta.ssid),
        CONFIG_GRU_WIFI_SSID,
        sizeof(wifi_config.sta.ssid) - 1
    );
    std::strncpy(
        reinterpret_cast<char*>(wifi_config.sta.password),
        CONFIG_GRU_WIFI_PASSWORD,
        sizeof(wifi_config.sta.password) - 1
    );
    wifi_config.sta.scan_method = WIFI_ALL_CHANNEL_SCAN;
    wifi_config.sta.sort_method = WIFI_CONNECT_AP_BY_SIGNAL;
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    const EventBits_t bits = xEventGroupWaitBits(
        event_group,
        kConnectedBit | kFailedBit,
        pdFALSE,
        pdFALSE,
        portMAX_DELAY
    );
    if ((bits & kConnectedBit) != 0) {
        ESP_LOGI(kTag, "Wi-Fi connected; HTTP server listening on port 8080");
        return ESP_OK;
    }
    ESP_LOGE(kTag, "Wi-Fi connection failed after %d retries", retry_count);
    return ESP_FAIL;
}
