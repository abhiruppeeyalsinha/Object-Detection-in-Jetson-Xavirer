#include "Detect.h"

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            cout << "[TensorRT] " << msg << endl;
        }
    }
} logger;

// CUDA Kernel: FP32 → FP16 Conversion (Normalized, CHW Layout)
__global__ void kernel_convertToFP16NormalizedCHW(
    const float* ch0, const float* ch1, const float* ch2, 
    __half* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int imageArea = width * height;
    output[0 * imageArea + idx] = __float2half(ch0[idx] / 255.0f);  // B
    output[1 * imageArea + idx] = __float2half(ch1[idx] / 255.0f);  // G
    output[2 * imageArea + idx] = __float2half(ch2[idx] / 255.0f);  // R
}

// CUDA Kernel: FP16 → FP32 Conversion
__global__ void kernel_fp16_to_fp32(const __half* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __half2float(input[idx]);
    }
}

// Wrapper Functions for Kernels
void convertToFP16NormalizedCHW(
    const float* ch0, const float* ch1, const float* ch2, 
    __half* output, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel_convertToFP16NormalizedCHW<<<grid, block, 0, stream>>>(ch0, ch1, ch2, output, width, height);
}

void fp16_to_fp32(const __half* input, float* output, int N, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    kernel_fp16_to_fp32<<<grid, block, 0, stream>>>(input, output, N);
}

// Constructor
Detect::Detect() : Yolov5_DIM(640, 640), ConfThreshold(0.25f), IOUThreshold(0.35f) {
    bBox_Dim[0] /= 2;
    bBox_Dim[1] /= 2;
    if (!LoadEngineFile()) {
        cerr << "Failed to initialize TensorRT engine!" << endl;
    }
}

// Destructor
Detect::~Detect() {
    for (auto& ptr : Bindings) cudaFree(ptr);
    cudaStreamDestroy(Stream);
}

// Load TensorRT Engine
bool Detect::LoadEngineFile() {
    ifstream engineFile(EngineFilePath, ios::binary);
    if (!engineFile) return false;

    engineFile.seekg(0, ios::end);
    size_t size = engineFile.tellg();
    engineFile.seekg(0, ios::beg);
    vector<char> engineData(size);
    engineFile.read(engineData.data(), size);

    runtime.reset(createInferRuntime(logger));
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    context.reset(engine->createExecutionContext());

    // Allocate GPU buffers
    Bindings.resize(engine->getNbBindings());
    for (int i = 0; i < engine->getNbBindings(); i++) {
        Dims dims = engine->getBindingDimensions(i);
        BindingSizes[i] = getMemorySize(dims, sizeof(__half));
        cudaMalloc(&Bindings[i], BindingSizes[i]);
    }

    cudaStreamCreate(&Stream);
    return true;
}

// Preprocess with CUDA Kernel
void Detect::preProcess(Mat frame, __half* gpuInput) {
    PadX = (Yolov5_DIM.width - frame.cols) / 2;
    PadY = (Yolov5_DIM.height - frame.rows) / 2;

    // Pad and normalize
    Mat padded;
    copyMakeBorder(frame, padded, PadY, PadY, PadX, PadX, BORDER_CONSTANT, Scalar(0, 0, 0));
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);

    // Split channels and upload to GPU
    vector<Mat> channels(3);
    split(padded, channels);
    float* d_channels[3];
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_channels[i], Yolov5_DIM.area() * sizeof(float));
        cudaMemcpyAsync(d_channels[i], channels[i].ptr<float>(), Yolov5_DIM.area() * sizeof(float), cudaMemcpyHostToDevice, Stream);
    }

    // Launch CUDA kernel
    convertToFP16NormalizedCHW(d_channels[0], d_channels[1], d_channels[2], gpuInput, Yolov5_DIM.width, Yolov5_DIM.height, Stream);

    // Cleanup
    for (int i = 0; i < 3; i++) cudaFree(d_channels[i]);
}

// Postprocess with CUDA Kernel
void Detect::postProcess(vector<__half>& gpuOutput, UInt16_t width, UInt16_t height) {
    // Convert FP16 → FP32 on GPU
    float* d_output_fp32;
    cudaMalloc(&d_output_fp32, OUTPUT_DETECTIONS * DETECTION_SIZE * sizeof(float));
    fp16_to_fp32(gpuOutput.data(), d_output_fp32, OUTPUT_DETECTIONS * DETECTION_SIZE, Stream);

    // Download to CPU
    vector<float> cpuOutput(OUTPUT_DETECTIONS * DETECTION_SIZE);
    cudaMemcpyAsync(cpuOutput.data(), d_output_fp32, cpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost, Stream);
    cudaStreamSynchronize(Stream);
    cudaFree(d_output_fp32);

    // Parse detections
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;
    for (int i = 0; i < OUTPUT_DETECTIONS; i++) {
        float* det = &cpuOutput[i * DETECTION_SIZE];
        float conf = det[4];
        if (conf < ConfThreshold) continue;

        float* scores = &det[5];
        int classId = max_element(scores, scores + 4) - scores;
        float confidence = conf * scores[classId];
        if (confidence < ConfThreshold) continue;

        float cx = det[0] - PadX;
        float cy = det[1] - PadY;
        float w = det[2];
        float h = det[3];
        int x1 = max(0, static_cast<int>(cx - w / 2));
        int y1 = max(0, static_cast<int>(cy - h / 2));
        int x2 = min(width - 1, static_cast<int>(cx + w / 2));
        int y2 = min(height - 1, static_cast<int>(cy + h / 2));

        if (x2 > x1 && y2 > y1) {
            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            confidences.push_back(confidence);
            classIds.push_back(classId);
        }
    }

    // NMS
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, ConfThreshold, IOUThreshold, indices);

    // Store results
    TargetData.clear();
    for (int idx : indices) {
        TargetData.push_back({
            static_cast<UInt8_t>(classIds[idx]),
            Point(boxes[idx].x + boxes[idx].width / 2, boxes[idx].y + boxes[idx].height / 2),
            boxes[idx],
            confidences[idx],
            0  // DetectionStatus (unused)
        });
    }
    targetDataFlag = true;
}

// Draw bounding boxes and center lines
void Detect::drawBoxes(Mat& frame) {
    Point frameCenter(frame.cols / 2, frame.rows / 2);
    DetectionCnt = 0;

    for (const auto& target : TargetData) {
        DetectionCnt++;
        rectangle(frame, target.BoundingBox, Scalar(255, 255, 255), 1);

        // Draw label with baseline adjustment
        string label = ClassName[target.ClassID] + " " + to_string(int(target.Confidance * 100)) + "%";
        if (target.ClassID == 1 || target.ClassID == 2) {  // Drones
            int relX = target.CenterPoint.x - frameCenter.x;
            int relY = target.CenterPoint.y - frameCenter.y;
            label += " (" + to_string(relX) + ", " + to_string(relY) + ")";
            line(frame, frameCenter, target.CenterPoint, Scalar(255, 255, 255), 1);
        }

        // Text placement with baseline
        int baseline = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        Point textOrg(target.BoundingBox.x, target.BoundingBox.y - baseline - 2);
        putText(frame, label, textOrg, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }
}

// Main detection pipeline
void Detect::Run_enqueueV2(Mat& frame) {
    // Preprocess (CUDA-accelerated)
    preProcess(frame, static_cast<__half*>(Bindings[0]));

    // Inference
    context->enqueueV2(Bindings.data(), Stream, nullptr);

    // Postprocess (CUDA-accelerated)
    postProcess(vector<__half>(static_cast<__half*>(Bindings[1]), 
              static_cast<__half*>(Bindings[1]) + BindingSizes[1] / sizeof(__half)), 
              frame.cols, frame.rows);

    // Draw results
    drawBoxes(frame);
}