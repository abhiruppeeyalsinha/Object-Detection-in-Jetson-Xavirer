// In Detect class, modify all stream usages:

// Add this helper method to get native CUDA stream
cudaStream_t getNativeStream() {
    return cv::cuda::StreamAccessor::getStream(cvStream);
}

// Modify all kernel launches and CUDA API calls:
void preProcess(const cv::Mat& frame, void* gpuInputBuff) {
    // ... existing code ...
    
    cudaStream_t rawStream = getNativeStream();
    preprocessKernel<<<grid, block, 0, rawStream>>>(
        gpuFrame.ptr<uchar3>(),
        static_cast<__half*>(gpuInputBuff),
        frame.cols, frame.rows,
        scale, offset
    );
}

void inference() {
    // ... existing code ...
    
    cudaStream_t rawStream = getNativeStream();
    Context->enqueueV2(Buffers.data(), rawStream, nullptr);
    
    fp16_to_fp32<<<blocks, threads, 0, rawStream>>>(
        (const __half*)Buffers[1],
        d_out_fp32,
        total
    );

    CHECK_CUDA(cudaMemcpyAsync(
        h_out, d_out_fp32,
        total * sizeof(float),
        cudaMemcpyDeviceToHost,
        rawStream  // Use native stream here
    ));
}

void postProcessAndDraw(int width, int height) {
    // ... existing code ...
    
    cudaStream_t rawStream = getNativeStream();
    if (!boxes.empty()) {
        drawBoxesKernel<<<grid, block, 0, rawStream>>>(
            d_displayFrame.ptr<uchar3>(), width, height,
            d_boxes, d_scores, d_class_ids, boxes.size()
        );
    }
}