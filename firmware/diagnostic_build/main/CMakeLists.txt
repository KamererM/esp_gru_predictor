idf_component_register(SRCS "main.cc" "gru_predictor.cc" "model_data.cc" "wifi.cc" "http.cc"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro esp_event esp_wifi esp_netif esp_http_server nvs_flash esp_timer esp_app_format esp_system spi_flash mbedtls)
