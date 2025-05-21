#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <unistd.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace std::chrono;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
             << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

const Size MODEL_DIM(640, 640);
const int OUTPUT_DETECTIONS = 25200;
const int DETECTION_SIZE = 9;
const float ConfThreshold = 0.25f;
const float IOUThreshold = 0.55f;
const int VideoFPS = 25;
const Scalar BOX_COLOR(0, 255, 0);
const int BOX_THICKNESS = 2;

string VideoPath;
string EngineFilePath;
string ClassName[4] = {"Bull", "Fox", "Hen", "Dog"};

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            cout << "[TensorRT] " << msg << endl;
        }
    }
} gLogger;

// Combined preprocessing kernel with aspect ratio preservation
__global__ void preprocessKernel(uchar3* input, __half* output, 
                               int inWidth, int inHeight,
                               float scale, int2 offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= MODEL_DIM.width || y >= MODEL_DIM.height) return;
    
    // Calculate original image coordinates
    int orig_x = (x - offset.x) / scale;
    int orig_y = (y - offset.y) / scale;
    
    uchar3 pixel;
    if (orig_x >= 0 && orig_x < inWidth && orig_y >= 0 && orig_y < inHeight) {
        pixel = input[orig_y * inWidth + orig_x];
    } else {
        pixel = make_uchar3(114, 114, 114); // YOLOv5 padding color
    }
    
    // Normalize and convert to FP16
    int idx = y * MODEL_DIM.width + x;
    int imageArea = MODEL_DIM.width * MODEL_DIM.height;
    output[0 * imageArea + idx] = __float2half(pixel.x / 255.0f);
    output[1 * imageArea + idx] = __float2half(pixel.y / 255.0f);
    output[2 * imageArea + idx] = __float2half(pixel.z / 255.0f);
}

// GPU-based NMS kernel
__global__ void nmsKernel(const float4* boxes, const float* scores, 
                         int* indices, int* count, 
                         int numBoxes, float iouThreshold) {
    extern __shared__ int selected[];
    int tid = threadIdx.x;
    
    if (tid == 0) *count = 0;
    __syncthreads();
    
    for (int i = tid; i < numBoxes; i += blockDim.x) {
        if (scores[i] < ConfThreshold) continue;
        
        bool keep = true;
        for (int j = 0; j < *count; ++j) {
            float4 box1 = boxes[i];
            float4 box2 = boxes[selected[j]];
            
            float xx1 = max(box1.x, box2.x);
            float yy1 = max(box1.y, box2.y);
            float xx2 = min(box1.z, box2.z);
            float yy2 = min(box1.w, box2.w);
            
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float inter = w * h;
            
            float area1 = (box1.z - box1.x) * (box1.w - box1.y);
            float area2 = (box2.z - box2.x) * (box2.w - box2.y);
            float iou = inter / (area1 + area2 - inter);
            
            if (iou > iouThreshold) {
                keep = false;
                break;
            }
        }
        
        if (keep) {
            int pos = atomicAdd(count, 1);
            selected[pos] = i;
        }
    }
    __syncthreads();
    
    if (tid < *count) {
        indices[tid] = selected[tid];
    }
}

// Kernel to draw bounding boxes on GPU
__global__ void drawBoxesKernel(uchar3* image, int width, int height,
                               const float4* boxes, const float* confidences,
                               const int* class_ids, int num_detections) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int i = 0; i < num_detections; ++i) {
        float4 box = boxes[i];
        int x1 = static_cast<int>(box.x);
        int y1 = static_cast<int>(box.y);
        int x2 = static_cast<int>(box.z);
        int y2 = static_cast<int>(box.w);
        
        // Draw bounding box
        if ((x >= x1 && x <= x2 && (y >= y1 && y <= y1 + BOX_THICKNESS)) ||  // Top edge
            (x >= x1 && x <= x2 && (y >= y2 - BOX_THICKNESS && y <= y2)) ||  // Bottom edge
            (y >= y1 && y <= y2 && (x >= x1 && x <= x1 + BOX_THICKNESS)) ||  // Left edge
            (y >= y1 && y <= y2 && (x >= x2 - BOX_THICKNESS && x <= x2))) {  // Right edge
            image[y * width + x] = make_uchar3(0, 255, 0);
        }
        
        // Draw label background
        if (i == 0 && x >= x1 && x <= x1 + 100 && y >= y1 - 20 && y <= y1) {
            image[y * width + x] = make_uchar3(0, 255, 0);
        }
    }
}

__global__ void fp16_to_fp32(const __half* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = __half2float(input[idx]);
}

class Detect {
private:
    unique_ptr<nvinfer1::ICudaEngine> Engine;
    unique_ptr<nvinfer1::IExecutionContext> Context;
    cv::cuda::Stream cvStream;
    vector<void*> Buffers;
    vector<int> BufferSizes;
    float* d_out_fp32;
    float* h_out;
    cuda::HostMem pinnedFrame;
    float scale;
    int2 offset;
    GpuMat d_displayFrame;
    int* d_indices;
    int* d_count;
    float4* d_boxes;
    float* d_scores;
    int* d_class_ids;

    struct {
        int64_t frame_copy;
        int64_t upload;
        int64_t preprocess;
        int64_t inference;
        int64_t postprocess;
        int64_t visualization;
    } stats {0,0,0,0,0,0};

    void loadEngine() {
        ifstream file(EngineFilePath, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Failed to open engine file: " + EngineFilePath);
        }
        file.seekg(0, ios::end);
        size_t size = file.tellg();
        file.seekg(0, ios::beg);
        vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        auto runtime = unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(gLogger));
        Engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
        Context.reset(Engine->createExecutionContext());

        int nbBindings = Engine->getNbBindings();
        Buffers.resize(nbBindings);
        BufferSizes.resize(nbBindings);

        for(int i=0; i<nbBindings; ++i){
            auto dims = Engine->getBindingDimensions(i);
            auto type = Engine->getBindingDataType(i);
            int64_t elt = (type==nvinfer1::DataType::kFLOAT?4:2);
            size_t bytes = elt;

            for(int d=0; d<dims.nbDims; ++d)
                bytes *= dims.d[d];
            BufferSizes[i] = bytes;
            CHECK_CUDA(cudaMalloc(&Buffers[i], (bytes + 127) & ~127));
        }
        
        CHECK_CUDA(cudaMalloc(&d_out_fp32, OUTPUT_DETECTIONS*DETECTION_SIZE*sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_out, OUTPUT_DETECTIONS*DETECTION_SIZE*sizeof(float)));
        
        // Allocate device memory for postprocessing
        CHECK_CUDA(cudaMalloc(&d_indices, OUTPUT_DETECTIONS * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_boxes, OUTPUT_DETECTIONS * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_scores, OUTPUT_DETECTIONS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_class_ids, OUTPUT_DETECTIONS * sizeof(int)));
    }

    void calculatePadding(int width, int height) {
        scale = min(static_cast<float>(MODEL_DIM.width) / width,
                   static_cast<float>(MODEL_DIM.height) / height);
        offset = make_int2(
            (MODEL_DIM.width - width * scale) / 2,
            (MODEL_DIM.height - height * scale) / 2
        );
    }

public:
    Detect() {
        loadEngine();
        cout << "Using FP16 precision with GPU-only processing\n" << endl;
    }

    ~Detect() {
        for (auto& b : Buffers) cudaFree(b);
        cudaFree(d_out_fp32);
        cudaFreeHost(h_out);
        cudaFree(d_indices);
        cudaFree(d_count);
        cudaFree(d_boxes);
        cudaFree(d_scores);
        cudaFree(d_class_ids);
    }

    void preProcess(const cv::Mat& frame, void* gpuInputBuff) {
        auto t_start = high_resolution_clock::now();

        calculatePadding(frame.cols, frame.rows);

        if (pinnedFrame.empty() || pinnedFrame.size() != frame.size()) {
            pinnedFrame.create(frame.rows, frame.cols, frame.type());
        }
        frame.copyTo(pinnedFrame);
        stats.frame_copy += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();

        t_start = high_resolution_clock::now();
        GpuMat gpuFrame(pinnedFrame);
        
        if (d_displayFrame.empty() || d_displayFrame.size() != frame.size()) {
            d_displayFrame.create(frame.rows, frame.cols, CV_8UC3);
        }
        gpuFrame.copyTo(d_displayFrame, cvStream);
        stats.upload += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();

        t_start = high_resolution_clock::now();
        if(gpuFrame.channels() == 1) {
            cv::cuda::GpuMat gpuBGR;
            cv::cuda::cvtColor(gpuFrame, gpuBGR, cv::COLOR_GRAY2BGR, 0, cvStream);
            gpuFrame = gpuBGR;
        }

        dim3 block(32, 32);
        dim3 grid((MODEL_DIM.width + block.x - 1)/block.x, 
                 (MODEL_DIM.height + block.y - 1)/block.y);

        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(cvStream);
        preprocessKernel<<<grid, block, 0, rawStream>>>(
            gpuFrame.ptr<uchar3>(),
            static_cast<__half*>(gpuInputBuff),
            frame.cols, frame.rows,
            scale, offset
        );

        stats.preprocess += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
    }

    void inference() {
        auto t_start = high_resolution_clock::now();
        
        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(cvStream);
        Context->enqueueV2(Buffers.data(), rawStream, nullptr);

        int total = OUTPUT_DETECTIONS * DETECTION_SIZE;
        dim3 threads(256), blocks((total + 255) / 256);
        fp16_to_fp32<<<blocks, threads, 0, rawStream>>>(
            (const __half*)Buffers[1],
            d_out_fp32,
            total
        );

        CHECK_CUDA(cudaMemcpyAsync(
            h_out, d_out_fp32,
            total * sizeof(float),
            cudaMemcpyDeviceToHost,
            rawStream
        ));

        cvStream.waitForCompletion();
        stats.inference += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
    }

    void postProcessAndDraw(int width, int height) {
        auto t_start = high_resolution_clock::now();

        // Process detections on host
        vector<float4> boxes;
        vector<float> scores;
        vector<int> class_ids;
        
        float inv_scale = 1.0f / scale;
        
        for (int i = 0; i < OUTPUT_DETECTIONS; ++i) {
            float* det = h_out + i * DETECTION_SIZE;
            float obj_conf = det[4];
            if (obj_conf < ConfThreshold) continue;

            float max_score = det[5];
            int max_idx = 0;
            for (int c = 1; c < 4; ++c) {
                if (det[5 + c] > max_score) {
                    max_score = det[5 + c];
                    max_idx = c;
                }
            }
            float conf = obj_conf * max_score;
            if (conf < ConfThreshold) continue;

            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            float x1 = (cx - w/2 - offset.x) * inv_scale;
            float y1 = (cy - h/2 - offset.y) * inv_scale;
            float x2 = (cx + w/2 - offset.x) * inv_scale;
            float y2 = (cy + h/2 - offset.y) * inv_scale;

            x1 = max(0.0f, min(static_cast<float>(width - 1), x1));
            y1 = max(0.0f, min(static_cast<float>(height - 1), y1));
            x2 = max(0.0f, min(static_cast<float>(width - 1), x2));
            y2 = max(0.0f, min(static_cast<float>(height - 1), y2));

            if (x2 > x1 && y2 > y1) {
                boxes.push_back(make_float4(x1, y1, x2, y2));
                scores.push_back(conf);
                class_ids.push_back(max_idx);
            }
        }

        // Copy detections to device for drawing
        if (!boxes.empty()) {
            CHECK_CUDA(cudaMemcpyAsync(d_boxes, boxes.data(), 
                                     boxes.size() * sizeof(float4),
                                     cudaMemcpyHostToDevice, cvStream));
            CHECK_CUDA(cudaMemcpyAsync(d_scores, scores.data(),
                                     scores.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, cvStream));
            CHECK_CUDA(cudaMemcpyAsync(d_class_ids, class_ids.data(),
                                     class_ids.size() * sizeof(int),
                                     cudaMemcpyHostToDevice, cvStream));
        }

        // Draw boxes on GPU
        dim3 block(32, 32);
        dim3 grid((width + block.x - 1)/block.x, 
                 (height + block.y - 1)/block.y);
        
        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(cvStream);
        if (!boxes.empty()) {
            drawBoxesKernel<<<grid, block, 0, rawStream>>>(
                d_displayFrame.ptr<uchar3>(), width, height,
                d_boxes, d_scores, d_class_ids, boxes.size()
            );
        }

        cvStream.waitForCompletion();
        stats.postprocess += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
    }

    void displayResult(Mat& displayFrame, bool draw = true) {
        auto t_start = high_resolution_clock::now();
        
        if (draw) {
            d_displayFrame.download(displayFrame, cvStream);
            cvStream.waitForCompletion();
        }
        
        stats.visualization += duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
    }

    void Run(cv::Mat& frame, bool draw = true, bool print = false) {
        auto t_start = high_resolution_clock::now();
        
        preProcess(frame, (__half*)Buffers[0]);
        inference();
        postProcessAndDraw(frame.cols, frame.rows);
        displayResult(frame, draw);

        if (print) {
            PrintDiagnostics();
        }
    }

    void PrintDiagnostics() const {
        cout << "\n--- Performance Diagnostics ---" << endl;
        cout << "Frame copy:    " << stats.frame_copy / 1000.0 << " ms" << endl;
        cout << "Upload:        " << stats.upload / 1000.0 << " ms" << endl;
        cout << "Preprocess:    " << stats.preprocess / 1000.0 << " ms" << endl;
        cout << "Inference:     " << stats.inference / 1000.0 << " ms" << endl;
        cout << "Postprocess:   " << stats.postprocess / 1000.0 << " ms" << endl;
        cout << "Visualization: " << stats.visualization / 1000.0 << " ms" << endl;
        cout << "Total:         " << (stats.frame_copy + stats.upload + stats.preprocess + 
                                     stats.inference + stats.postprocess + stats.visualization) / 1000.0 
             << " ms" << endl;
        cout << "-----------------------------" << endl;
    }
};

int main(int argc, char const* argv[]) {
    high_resolution_clock::time_point timCntStart, timCntEnd;
    duration<double, milli> timDuration;

    bool displayFlag = false;
    bool saveFlag = false;
    bool diagnostics = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-S") saveFlag = true;
        else if (arg == "-DF") displayFlag = true;
        else if (arg == "-V" && ++i < argc) VideoPath = argv[i];
        else if (arg == "-M" && ++i < argc) EngineFilePath = argv[i];
        else if (arg == "-D") diagnostics = true;
    }

    CHECK_CUDA(cudaFree(0));
    cout << "CUDA context initialized" << endl;

    VideoCapture cap(VideoPath);
    if (!cap.isOpened()) {
        cerr << "Error opening video source: " << VideoPath << endl;
        return -1;
    }

    unique_ptr<Detect> detect;
    try {
        detect = make_unique<Detect>();
        Mat frame;
        VideoWriter outVid;

        while (true) {
            timCntStart = high_resolution_clock::now();

            if (!cap.read(frame) || frame.empty()) {
                cerr << "End of video stream" << endl;
                break;
            }

            detect->Run(frame, displayFlag || saveFlag, diagnostics);
        
            timCntEnd = high_resolution_clock::now();
            timDuration = timCntEnd - timCntStart;
            cout << "Frame processed in " << timDuration.count() << " ms" << endl;

            if (displayFlag) {
                imshow("Object Detection", frame);
                if (waitKey(1) == 'q') break;
            }

            if (saveFlag) {
                if (!outVid.isOpened()) {
                    outVid.open("output.avi", 
                               VideoWriter::fourcc('M','J','P','G'),
                               VideoFPS, frame.size(), true);
                }
                outVid.write(frame);
            }
        }

        if (diagnostics) {
            detect->PrintDiagnostics();
        }
        destroyAllWindows();
    }
    catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
