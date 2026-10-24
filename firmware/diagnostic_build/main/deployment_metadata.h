#pragma once
#include <cstddef>

constexpr bool kDeploymentMetadataConfigured = true;
constexpr char kRunId[] = "56f38c60-e7c8-4ba0-9d07-c6a184fe6680";
constexpr char kModelSha256[] = "ced2bd0441434481f8813e80dca49931222a8eaebd76b49f67a8cac6e9318bca";
constexpr char kModelDataCcSha256[] = "b7ff54a4f21cec00455123398e89f5ced2aec85eea0d1458aa26b427c35e2806";
constexpr char kPreprocessingConstantsSha256[] = "e5e06f192894ff54f202e3c8e8b5e3dc1ba539033a0cce991d613cc13d52c466";
constexpr char kTrainingMeanFloat32Hex[] = "0x1.4750560000000p+0";
constexpr char kTrainingStdFloat32Hex[] = "0x1.1310aa0000000p+0";
constexpr float kExpectedInputScale = 0x1.6198780000000p-6f;
constexpr int kExpectedInputZeroPoint = -80;
constexpr float kExpectedOutputScale = 0x1.c5e4a80000000p-7f;
constexpr int kExpectedOutputZeroPoint = -41;
constexpr char kFirmwareCommit[] = "8f3c2928e83d671b8bcbdd6324f4795d89ac15fc";
constexpr char kFirmwareBuildId[] = "174136112180465d6d865d16c929ae6708c07f1da0b7c1c58bb9f72c93bba03e";
constexpr char kBoardOrModule[] = "ESP32-S3";
constexpr char kIdfTarget[] = "esp32s3";
constexpr int kCpuFrequencyMhz = 160;
constexpr char kTflmRevision[] = "espressif/esp-tflite-micro@1.4.1";
constexpr char kOperatorResolverId[] = "gru_resolver_v1";
constexpr char kSdkconfigSha256[] = "81908759986cdb4de190ba53692f44f7a8408d7385076527cf0263e1dd576ec5";
constexpr char kResolverSourceSha256[] = "24158b7d3be3db0362ace6588d14af48af7ede637bc966f9b6022b67bf0090f2";
