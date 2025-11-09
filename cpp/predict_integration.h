//---------------------------------------------------------------------------
// XRD Prediction Integration
// Simple C-style functions for calling Python predictor
//---------------------------------------------------------------------------

#ifndef PredictIntegrationH
#define PredictIntegrationH

//---------------------------------------------------------------------------
// Структура параметрів деформації
//---------------------------------------------------------------------------
struct DeformParams {
    double Dmax1;
    double D01;
    double L1;
    double Rp1;
    double D02;
    double L2;
    double Rp2;
};

//---------------------------------------------------------------------------
// Predict параметри з експериментальної кривої
//
// curve: масив 661 точок (після crop [40:701])
// params: вихідна структура з параметрами
// predictor_path: шлях до predict.exe (default: "predict.exe")
// model_path: шлях до checkpoint моделі (default: "checkpoints/model.pt")
//
// Returns: 1 якщо успішно, 0 якщо помилка
//---------------------------------------------------------------------------
int PredictFromCurve(double* curve, int curve_length,
                     DeformParams* params,
                     const char* predictor_path = "predict.exe",
                     const char* model_path = "checkpoints/model.pt");

#endif
