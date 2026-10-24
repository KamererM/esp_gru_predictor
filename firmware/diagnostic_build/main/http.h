#ifndef MAIN_HTTP_H_
#define MAIN_HTTP_H_

#include "esp_http_server.h"

httpd_handle_t start_webserver();
void stop_webserver(httpd_handle_t server);

#endif
