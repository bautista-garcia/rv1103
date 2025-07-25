#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "tensorflow/lite/c/c_api.h"
#include <time.h>
#include <inttypes.h>

static inline uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);          /* OK on uClibc */
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

#define TIME_BLOCK_US(start)        uint64_t start = now_us()
#define PRINT_DURATION(start,label) \
    printf("[PROFILE] %-22s %8" PRIu64 " μs\n", label, now_us() - (start))
#define MODEL_PATH "mobilenet_v2.tflite"
#define LABELS_PATH "synset.txt"
#define INPUT_WIDTH 224
#define INPUT_HEIGHT 224
#define INPUT_CHANNELS 3
#define TOP_K 5
#define MAX_LABEL_LEN 256
#define NUM_CLASSES 1000


static void tflite_stderr(void* /*user_data*/,
                          const char* format, va_list args) {
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
}

// Loads a raw RGB file (224x224x3, uint8) and converts to float32 NHWC
int load_and_preprocess_image(const char* rgb_path, float* out_data) {
    FILE* f = fopen(rgb_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open RGB file: %s\n", rgb_path);
        return -1;
    }
    size_t expected = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    uint8_t* buf = (uint8_t*)malloc(expected);
    if (!buf) {
        fprintf(stderr, "Failed to allocate buffer for image\n");
        fclose(f);
        return -1;
    }
    size_t n = fread(buf, 1, expected, f);
    fclose(f);
    if (n != expected) {
        fprintf(stderr, "RGB file size mismatch: got %zu, expected %zu\n", n, expected);
        free(buf);
        return -1;
    }
    // Convert to float32 (normalize to 0-1 if needed)
    for (size_t i = 0; i < expected; ++i) {
        out_data[i] = buf[i] / 127.5f - 1.0f;
    }
    free(buf);
    return 0;
}

// Loads class labels from synset.txt
int load_labels(const char* filename, char labels[NUM_CLASSES][MAX_LABEL_LEN]) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;
    int i = 0;
    while (i < NUM_CLASSES && fgets(labels[i], MAX_LABEL_LEN, f)) {
        size_t len = strlen(labels[i]);
        if (len > 0 && labels[i][len-1] == '\n') labels[i][len-1] = '\0';
        i++;
    }
    fclose(f);
    return (i == NUM_CLASSES) ? 0 : -2;
}

// Applies softmax to logits
void softmax(float* data, int n) {
    float max = data[0];
    for (int i = 1; i < n; ++i) if (data[i] > max) max = data[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        data[i] = expf(data[i] - max);
        sum += data[i];
    }
    for (int i = 0; i < n; ++i) data[i] /= sum;
}

// Find top-k indices
void top_k(float* data, int n, int* indices, int k) {
    for (int i = 0; i < k; ++i) indices[i] = i;
    for (int i = k; i < n; ++i) {
        int min_idx = 0;
        for (int j = 1; j < k; ++j) if (data[indices[j]] < data[indices[min_idx]]) min_idx = j;
        if (data[i] > data[indices[min_idx]]) indices[min_idx] = i;
    }
    // Sort top-k by score descending
    for (int i = 0; i < k-1; ++i) {
        for (int j = i+1; j < k; ++j) {
            if (data[indices[j]] > data[indices[i]]) {
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image_rgb_file>\n", argv[0]);
        return 1;
    }
    const char* image_path = argv[1];

    TIME_BLOCK_US(t_total);

    TIME_BLOCK_US(t_labels);
    printf("[INFO] Loading labels from %s...\n", LABELS_PATH);
    char labels[NUM_CLASSES][MAX_LABEL_LEN];
    if (load_labels(LABELS_PATH, labels) != 0) {
        fprintf(stderr, "[ERROR] Failed to load labels from %s\n", LABELS_PATH);
        return 1;
    }
    PRINT_DURATION(t_labels, "Load labels");

    TIME_BLOCK_US(t_model);
    printf("[INFO] Loading model from %s...\n", MODEL_PATH);
    TfLiteModel* model = TfLiteModelCreateFromFile(MODEL_PATH);
    if (!model) {
        fprintf(stderr, "[ERROR] Failed to load model from %s\n", MODEL_PATH);
        return 1;
    }
    PRINT_DURATION(t_model, "Load model");

    TIME_BLOCK_US(t_interpreter);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetErrorReporter(options, tflite_stderr, NULL);
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteModelDelete(model);
    TfLiteInterpreterOptionsDelete(options);
    if (!interpreter) {
        fprintf(stderr, "[ERROR] Failed to create interpreter\n");
        return 1;
    }
    PRINT_DURATION(t_interpreter, "Create interpreter");

    TIME_BLOCK_US(t_allocate);
    printf("[INFO] Allocating tensors...\n");
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "[ERROR] Failed to allocate tensors\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }
    PRINT_DURATION(t_allocate, "Allocate tensors");

    TIME_BLOCK_US(t_input);
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    if (!input_tensor) {
        fprintf(stderr, "[ERROR] Failed to get input tensor\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }

    if (TfLiteTensorType(input_tensor) != kTfLiteFloat32) {
        fprintf(stderr, "[ERROR] Expected float32 input tensor\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }


    float* input_data = (float*)TfLiteTensorData(input_tensor);
    if (load_and_preprocess_image(image_path, input_data) != 0) {
        fprintf(stderr, "[ERROR] Failed to load and preprocess image\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }

    PRINT_DURATION(t_input, "Load and preprocess input tensor");


    TIME_BLOCK_US(t_invoke);
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "[ERROR] Failed to invoke interpreter\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }
    PRINT_DURATION(t_invoke, "Inference");

    TIME_BLOCK_US(t_output);
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    if (!output_tensor) {
        fprintf(stderr, "[ERROR] Failed to get output tensor\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }

    if (TfLiteTensorType(output_tensor) != kTfLiteFloat32) {
        fprintf(stderr, "[ERROR] Expected float32 output tensor\n");
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }
    const float* output_data = (const float*)TfLiteTensorData(output_tensor);
    int output_size = TfLiteTensorByteSize(output_tensor) / sizeof(float);
    if (output_size != NUM_CLASSES) {
        fprintf(stderr, "[ERROR] Unexpected output size: %d\n", output_size);
        TfLiteInterpreterDelete(interpreter);
        return 1;
    }
    PRINT_DURATION(t_output, "Get output tensor");

    TIME_BLOCK_US(t_postprocess);
    float probs[NUM_CLASSES];
    memcpy(probs, output_data, sizeof(float) * NUM_CLASSES);

    float sum = 0.f;
    for (int i = 0; i < NUM_CLASSES; ++i) sum += output_data[i];
    printf("raw-sum = %.4f\n", sum);   /* should print ≈ 1.0000 */

    // softmax(probs, NUM_CLASSES);
    int topk[TOP_K];
    top_k(probs, NUM_CLASSES, topk, TOP_K);

    printf("Top-%d predictions:\n", TOP_K);
    for (int i = 0; i < TOP_K; ++i) {
        printf("[%d] %.4f %s\n", topk[i], output_data[topk[i]], labels[topk[i]]);
    }

    PRINT_DURATION(t_postprocess, "Postprocess");

    TfLiteInterpreterDelete(interpreter);
    printf("[INFO] Done.\n");

    PRINT_DURATION(t_total, "Total");
    return 0;
}
