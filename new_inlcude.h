#ifndef INCLUDES_H_
#define INCLUDES_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Standard Includes
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>  // Added for max_element

// System Includes
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <termios.h>
#include <arpa/inet.h>

// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>  // Added for CUDA-OpenCV interop

// CUDA/TensorRT Includes
#include <cuda_runtime.h>
#include <cuda_fp16.h>          // For __half support
#include <device_launch_parameters.h>  // For kernel launch configs
#include <NvInfer.h>
#include <NvInferRuntime.h>

// Namespaces
using namespace cv;
using namespace std;
using namespace nvinfer1;
using namespace std::chrono;

// Typedefs (from DataTypes.h)
typedef signed char             Int8_t;
typedef unsigned char           UInt8_t;
typedef short                   Int16_t;
typedef unsigned short          UInt16_t;
typedef int                     Int32_t;
typedef unsigned int            UInt32_t;
typedef long long               Int64_t;
typedef unsigned long long      UInt64_t;

// Target Data Structure
struct _TargetData_ {
    UInt8_t ClassID;            // Class ID (0: Bird, 1: Fixed, 2: Hexa, 3: Plane)
    Point CenterPoint;          // Center coordinates (relative to frame)
    Rect BoundingBox;           // Bounding box (x, y, w, h)
    float Confidance;           // Confidence score (0-100%)
    UInt8_t DetectionStatus;    // Reserved for future use
};

#pragma GCC diagnostic pop

#endif /* INCLUDES_H_ */