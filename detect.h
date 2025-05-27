#ifndef DETECT_H_
#define DETECT_H_

#include "Includes.h"

// Extern Variables
extern bool targetDataFlag;
extern Size VIDEO_DIM;
extern UInt32_t DetectionCnt;
extern vector<struct _TargetData_> TargetData;
extern Int16_t Loc_Val[2];
extern UInt16_t bBox_Dim[2];
extern string ClassName[4];
extern string EngineFilePath;

// Constants from CUDA reference
const int OUTPUT_DETECTIONS = 25200;
const int DETECTION_SIZE = 9;

// CUDA Kernel Declarations (Added)
void convertToFP16NormalizedCHW(const float* ch0, const float* ch1, const float* ch2, __half* output, int width, int height, cudaStream_t stream);
void fp16_to_fp32(const __half* input, float* output, int N, cudaStream_t stream);

class Detect {
private:
    Size Yolov5_DIM;
    int PadX, PadY;
    float ConfThreshold;
    float IOUThreshold;

    // GPU Buffers
    vector<void*> Bindings;
    vector<size_t> BindingSizes;
    cudaStream_t Stream;

    // TensorRT Objects
    unique_ptr<IRuntime> runtime;
    unique_ptr<ICudaEngine> engine;
    unique_ptr<IExecutionContext> context;

public:
    Detect();
    ~Detect();

    size_t getMemorySize(Dims dims, size_t ElementSize);
    bool LoadEngineFile();
    void preProcess(Mat frame, __half* gpuInput);
    void postProcess(vector<__half>& gpuOutput, UInt16_t width, UInt16_t height);
    void drawBoxes(Mat& frame);
    void Run_enqueueV2(Mat& frame);
};

#endif /* DETECT_H_ */