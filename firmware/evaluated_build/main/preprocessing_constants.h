#pragma once
constexpr float kTrainingMeanKwh = 0x1.4750560000000p+0f;
constexpr float kTrainingStdKwh = 0x1.1310aa0000000p+0f;
constexpr bool kPreprocessingConstantsArePlaceholder = false;
/* Quantize with roundf and clamp to [-128, 127] before int8_t cast. */
