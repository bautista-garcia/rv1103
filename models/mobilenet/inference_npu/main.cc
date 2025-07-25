#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mobilenet.h"
#include "image_utils.h"
#include "file_utils.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.cpp"
#endif
#include <time.h>
#include <inttypes.h>

static inline uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);          
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

#define TIME_BLOCK_US(start)        uint64_t start = now_us()
#define PRINT_DURATION(start,label) \
    printf("[PROFILE] %-22s %8" PRIu64 " μs\n", label, now_us() - (start))

#define IMAGENET_CLASSES_FILE "./model/synset.txt"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    int line_count;
    char** lines = read_lines_from_file(IMAGENET_CLASSES_FILE, &line_count);
    if (lines == NULL) {
        printf("read classes label file fail! path=%s\n", IMAGENET_CLASSES_FILE);
        return -1;
    }

    TIME_BLOCK_US(t_total);

    TIME_BLOCK_US(t_init);
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_mobilenet_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    PRINT_DURATION(t_init, "Init");

    TIME_BLOCK_US(t_load_image);
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
    }

#if defined(RV1106_1103) 
    //RV1106 rga requires that input and output bufs are memory allocated by dma
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                       (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif
    PRINT_DURATION(t_load_image, "Load image");

    TIME_BLOCK_US(t_inference);
    int topk = 5;
    mobilenet_result result[topk];

    ret = inference_mobilenet_model(&rknn_app_ctx, &src_image, result, topk);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d\n", ret);
        goto out;
    }
    PRINT_DURATION(t_inference, "Inference");

    for (int i = 0; i < topk; i++) {
        printf("[%d] score=%.6f class=%s\n", result[i].cls, result[i].score, lines[result[i].cls]);
    }

out:
    ret = release_mobilenet_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_mobilenet_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103) 
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }
    if (lines != NULL) {
        free_lines(lines, line_count);
    }

    PRINT_DURATION(t_total, "Total");

    return 0;
}
