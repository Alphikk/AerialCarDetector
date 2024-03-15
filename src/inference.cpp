    #include "inference.h"
    #include <regex>

    #define benchmark

    DCSP_CORE::DCSP_CORE() {

    }


    DCSP_CORE::~DCSP_CORE() {
        delete session_;
    }

    template<typename T>
    char *BlobFromImage(cv::Mat &iImg, T &iBlob) {
        int channels = iImg.channels();
        int imgHeight = iImg.rows;
        int imgWidth = iImg.cols;

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < imgHeight; h++) {
                for (int w = 0; w < imgWidth; w++) {
                    iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                            (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
                }
            }
        }
        return RET_OK;
    }


    char *PostProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg) {
        cv::Mat img = iImg.clone();
        cv::resize(iImg, oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        if (img.channels() == 1) {
            cv::cvtColor(oImg, oImg, cv::COLOR_GRAY2BGR);
        }
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
        return RET_OK;
    }


    char *DCSP_CORE::CreateSession(DCSP_INIT_PARAM &iParams) {
        char *Ret = RET_OK;
        std::regex pattern("[\u4e00-\u9fa5]");
        bool result = std::regex_search(iParams.ModelPath, pattern);
        if (result) {
            Ret = "[DCSP_ONNX]:Model path error.Change your model path without chinese characters.";
            std::cout << Ret << std::endl;
            return Ret;
        }
        try {
            rectConfidenceThreshold_ = iParams.RectConfidenceThreshold;
            iouThreshold_ = iParams.iouThreshold;
            imgSize_ = iParams.imgSize;
            modelType_ = iParams.ModelType;
            env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
            Ort::SessionOptions sessionOption;
            sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            sessionOption.SetIntraOpNumThreads(iParams.IntraOpNumThreads);
            sessionOption.SetLogSeverityLevel(iParams.LogSeverityLevel);
            const char * modelPath = iParams.ModelPath.c_str();
            session_ = new Ort::Session(env_, modelPath, sessionOption);
            Ort::AllocatorWithDefaultOptions allocator;
            size_t inputNodesNum = session_->GetInputCount();
            for (size_t i = 0; i < inputNodesNum; i++) {
                Ort::AllocatedStringPtr input_node_name = session_->GetInputNameAllocated(i, allocator);
                char *temp_buf = new char[50];
                strcpy(temp_buf, input_node_name.get());
                inputNodeNames_.push_back(temp_buf);
            }
            size_t OutputNodesNum = session_->GetOutputCount();
            for (size_t i = 0; i < OutputNodesNum; i++) {
                Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(i, allocator);
                char *temp_buf = new char[10];
                strcpy(temp_buf, output_node_name.get());
                outputNodeNames_.push_back(temp_buf);
            }
            options_ = Ort::RunOptions{nullptr};
            WarmUpSession();
            return RET_OK;
        }
        catch (const std::exception &e) {
            const char *str1 = "[DCSP_ONNX]:";
            const char *str2 = e.what();
            std::string result = std::string(str1) + std::string(str2);
            char *merged = new char[result.length() + 1];
            std::strcpy(merged, result.c_str());
            std::cout << merged << std::endl;
            delete[] merged;
            return "[DCSP_ONNX]:Create session failed.";
        }

    }


char *DCSP_CORE::RunSession(cv::Mat &iImg, std::vector<DCSP_RESULT> &oResult)
{
    char *Ret = RET_OK;
    cv::Mat processedImg;
    PostProcess(iImg, imgSize_, processedImg);
    if (modelType_ < 4)
    {
        float *blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = {1, 3, imgSize_.at(0), imgSize_.at(1)};
        TensorProcess(iImg, blob, inputNodeDims, oResult);
    }

    return Ret;
}


template<typename N>
char *DCSP_CORE::TensorProcess( cv::Mat &iImg,
                                N &blob,
                                std::vector<int64_t> &inputNodeDims,
                                std::vector<DCSP_RESULT> &oResult )
{
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                blob,
                3 * imgSize_.at(0) * imgSize_.at(1),
                inputNodeDims.data(),
                inputNodeDims.size());

    auto outputTensor = session_->Run(options_,
                                     inputNodeNames_.data(),
                                     &inputTensor,
                                     1,
                                     outputNodeNames_.data(),
                                     outputNodeNames_.size());

        Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
        auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
        auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
        delete blob;
        switch (modelType_) {
            case 1://V8_ORIGIN_FP32
            case 4://V8_ORIGIN_FP16
            {
                int strideNum = outputNodeDims[2];
                int signalResultNum = outputNodeDims[1];
                std::vector<int> class_ids;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                cv::Mat rawData;
                if (modelType_ == 1) {
                    // FP32
                    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
                } else {
                    // FP16
                    rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
                    rawData.convertTo(rawData, CV_32F);
                }
                rawData = rawData.t();
                float *data = (float *) rawData.data;

                float x_factor = iImg.cols / 640.;
                float y_factor = iImg.rows / 640.;
                for (int i = 0; i < strideNum; ++i) {
                    float *classesScores = data + 4;
                    cv::Mat scores(1, this->classes_.size(), CV_32FC1, classesScores);
                    cv::Point class_id;
                    double maxClassScore;
                    cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                    if (maxClassScore > rectConfidenceThreshold_) {
                        confidences.push_back(maxClassScore);
                        class_ids.push_back(class_id.x);

                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];

                        int left = int((x - 0.5 * w) * x_factor);
                        int top = int((y - 0.5 * h) * y_factor);

                        int width = int(w * x_factor);
                        int height = int(h * y_factor);

                        boxes.emplace_back(left, top, width, height);
                    }
                    data += signalResultNum;
                }

                std::vector<int> nmsResult;
                cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold_, iouThreshold_, nmsResult);

                for (int i = 0; i < nmsResult.size(); ++i)
                {
                    int idx = nmsResult[i];
                    DCSP_RESULT result;
                    result.classId = class_ids[idx];
                    result.confidence = confidences[idx];
                    result.box = boxes[idx];
                    oResult.push_back(result);
                }
                break;
            }
        }
        return RET_OK;
    }


char *DCSP_CORE::WarmUpSession()
{
        cv::Mat iImg = cv::Mat(cv::Size(imgSize_.at(0), imgSize_.at(1)), CV_8UC3);
        cv::Mat processedImg;
        PostProcess(iImg, imgSize_, processedImg);
        if (modelType_ < 4)
        {
            float *blob = new float[iImg.total() * 3];
            BlobFromImage(processedImg, blob);
            std::vector<int64_t> YOLO_input_node_dims = {1, 3, imgSize_.at(0), imgSize_.at(1)};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                    blob,
                    3 * imgSize_.at(0) * imgSize_.at(1),
                    YOLO_input_node_dims.data(),
                    YOLO_input_node_dims.size());

            auto output_tensors = session_->Run(options_, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(),
                                               outputNodeNames_.size());
            delete[] blob;
        }
        return RET_OK;
    }
