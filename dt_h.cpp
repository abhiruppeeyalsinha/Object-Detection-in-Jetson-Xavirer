#include "Detect.h"

// Logger for TensorRT messages
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            cout << "[TensorRT] " << msg << endl;
        }
    }
};

Logger logger;

// External variables
bool targetDataFlag = false;
Size VIDEO_DIM(640, 480);
UInt32_t DetectionCnt = 0;
vector<struct _TargetData_> TargetData;
Int16_t Loc_Val[2] = {0};
UInt16_t bBox_Dim[2] = {0};
string ClassName[4] = {"Bird", "Fixed", "Hexa", "Plane"};
string EngineFilePath = "yolov5s.engine";

// Constructor
Detect::Detect()
{
    bBox_Dim[0] = bBox_Dim[0] / 2;
    bBox_Dim[1] = bBox_Dim[1] / 2;

    Yolov5_DIM.height = 640;
    Yolov5_DIM.width = 640;

    PadX = 0;
    PadY = 0;

    ConfThreshold = 0.25f;  // Updated threshold
    IOUThreshold = 0.35f;   // Updated threshold

    if(LoadEngineFile())
    {
        cout << "Engine Loaded Successfully" << endl;
    }
}

// Destructor
Detect::~Detect()
{
    try
    {
        for(void *ptr : Bindings)
        {
            if(ptr) cudaFree(ptr);
        }
        cudaStreamDestroy(Stream);
    }
    catch(Exception &e) {}
}

size_t Detect::getMemorySize(Dims dims, size_t ElementSize)
{
    size_t size = ElementSize;
    for(Int32_t i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}

bool Detect::LoadEngineFile()
{
    try
    {
        ifstream EngineFile(EngineFilePath, ios::binary);
        if (!EngineFile) {
            cerr << "Failed to read engine file." << endl;
            return false;
        }

        EngineFile.seekg(0, ios::end);
        auto fsize = EngineFile.tellg();
        EngineFile.seekg(0, ios::beg);

        vector<char> engine_data(fsize);
        EngineFile.read(engine_data.data(), fsize);

        runtime.reset(createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(engine_data.data(), fsize));
        context.reset(engine->createExecutionContext());

        Int32_t nbBindings = engine->getNbBindings();
        Bindings.resize(nbBindings, nullptr);
        BindingSizes.resize(nbBindings, 0);

        for(Int32_t i = 0; i < nbBindings; i++)
        {
            Dims dims = engine->getBindingDimensions(i);
            BindingSizes[i] = getMemorySize(dims, sizeof(__half));
            cudaMalloc(&Bindings[i], BindingSizes[i]);
        }

        cudaStreamCreate(&Stream);

        InpBuff.resize(BindingSizes[0] / sizeof(__half));
        OutBuff.resize(BindingSizes[1] / sizeof(__half));

        return true;
    }
    catch(Exception &e)
    {
        return false;
    }
}

void Detect::preProcess(Mat VideoFrame, __half *InputBuff)
{
    // Calculate padding
    PadX = (Yolov5_DIM.width - VideoFrame.cols) / 2;
    PadY = (Yolov5_DIM.height - VideoFrame.rows) / 2;

    // Resize with padding
    copyMakeBorder(VideoFrame, VideoFrame, PadY, PadY, PadX, PadX, BORDER_CONSTANT, Scalar(0,0,0));

    // Convert to float and normalize
    VideoFrame.convertTo(VideoFrame, CV_32F, 1.0/255);
    
    // Split channels and convert to FP16
    vector<Mat> FrameChannels(3);
    split(VideoFrame, FrameChannels);

    float* ChannelData;
    for(UInt32_t ch = 0; ch < 3; ch++)
    {
        ChannelData = reinterpret_cast<float*>(FrameChannels[ch].data);
        for(Int32_t i = 0; i < Yolov5_DIM.width * Yolov5_DIM.height; i++)
        {
            InpBuff[ch * Yolov5_DIM.width * Yolov5_DIM.height + i] = __float2half(ChannelData[i]);
        }
    }
}

void Detect::postProcess(vector<__half> &Output, UInt16_t VideoWidth, UInt16_t VideoHeight)
{
    vector<Rect> Boxes;
    vector<Int32_t> Indices;
    vector<UInt8_t> ClassIds;
    vector<float> Confidences;
    const UInt32_t Detections = 25200;

    for(UInt32_t i = 0; i < Detections; i++)
    {
        float objConf = __half2float(Output[i*9 + 4]);
        if(objConf < ConfThreshold) continue;

        float classScores[4];
        for(int j = 0; j < 4; j++) {
            classScores[j] = __half2float(Output[i*9 + 5 + j]);
        }

        UInt8_t classID = max_element(classScores, classScores+4) - classScores;
        float classConf = classScores[classID];
        float confidence = objConf * classConf;

        if(confidence <= ConfThreshold) continue;

        // Calculate coordinates
        float cx = __half2float(Output[i*9 + 0]) - PadX;
        float cy = __half2float(Output[i*9 + 1]) - PadY;
        float width = __half2float(Output[i*9 + 2]);
        float height = __half2float(Output[i*9 + 3]);

        int x1 = max(0, (int)(cx - width/2));
        int y1 = max(0, (int)(cy - height/2));
        int x2 = min(VideoWidth - 1, (int)(cx + width/2));
        int y2 = min(VideoHeight - 1, (int)(cy + height/2));

        if(x2 >= x1 && y2 >= y1)
        {
            Boxes.emplace_back(Rect(x1, y1, x2-x1, y2-y1));
            Confidences.emplace_back(confidence);
            ClassIds.emplace_back(classID);
        }
    }

    // Apply NMS
    dnn::NMSBoxes(Boxes, Confidences, ConfThreshold, IOUThreshold, Indices);

    // Store results
    TargetData.clear();
    for(UInt32_t idx : Indices)
    {
        _TargetData_ DetectData;
        DetectData.ClassID = ClassIds[idx];
        DetectData.Confidance = Confidences[idx];
        DetectData.BoundingBox = Boxes[idx];
        DetectData.CenterPoint = Point(Boxes[idx].x + Boxes[idx].width/2, 
                                      Boxes[idx].y + Boxes[idx].height/2);
        TargetData.push_back(DetectData);
    }
    targetDataFlag = true;
}

void Detect::drawBoxes(Mat &VideoFrame)
{
    string Label = "";
    Point FrameCentre(VideoFrame.cols / 2, VideoFrame.rows / 2);

    DetectionCnt = 0;

    for(auto &Target : TargetData)
    {
        DetectionCnt++;

        if(Target.ClassID == 0 || Target.ClassID == 3) // Bird or Plane
        {
            rectangle(VideoFrame, Target.BoundingBox, Scalar(255, 255, 255), 1);
            Label = ClassName[Target.ClassID] + " " + to_string((UInt32_t)(Target.Confidance * 100)) + "%";
            putText(VideoFrame, Label, Target.BoundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }
        else if(Target.ClassID == 1 || Target.ClassID == 2) // Drones
        {
            rectangle(VideoFrame, Target.BoundingBox, Scalar(255, 255, 255), 1);
            int relX = Target.CenterPoint.x - VideoFrame.cols/2;
            int relY = Target.CenterPoint.y - VideoFrame.rows/2;
            Label = ClassName[Target.ClassID] + " " + to_string((UInt32_t)(Target.Confidance * 100)) + 
                   "% (" + to_string(relX) + ", " + to_string(relY) + ")";
            putText(VideoFrame, Label, Target.BoundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            line(VideoFrame, FrameCentre, Target.CenterPoint, Scalar(255, 255, 255), 1);
        }
        else
        {
            DetectionCnt--;
        }
    }
}

void Detect::Run_enqueueV2(Mat &VideoFrame)
{
    high_resolution_clock::time_point timCntStart, timCntEnd;
    duration<double, milli> timDuration;

    // PreProcess
    timCntStart = high_resolution_clock::now();
    preProcess(VideoFrame, InpBuff.data());
    timCntEnd = high_resolution_clock::now();
    timDuration = timCntEnd - timCntStart;
    cout << "PreProcess Time - " << timDuration.count() << " ms" << endl << flush;

    // Inference
    timCntStart = high_resolution_clock::now();
    cudaMemcpyAsync(Bindings[0], InpBuff.data(), BindingSizes[0], cudaMemcpyHostToDevice, Stream);
    context->enqueueV2(Bindings.data(), Stream, nullptr);
    cudaMemcpyAsync(OutBuff.data(), Bindings[1], BindingSizes[1], cudaMemcpyDeviceToHost, Stream);
    cudaStreamSynchronize(Stream);
    timCntEnd = high_resolution_clock::now();
    timDuration = timCntEnd - timCntStart;
    cout << "Inference Time - " << timDuration.count() << " ms" << endl << flush;

    // PostProcess
    postProcess(OutBuff, VideoFrame.cols, VideoFrame.rows);
    drawBoxes(VideoFrame);
}