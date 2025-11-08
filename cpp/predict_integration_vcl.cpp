//---------------------------------------------------------------------------
// XRD Prediction Integration (VCL VERSION)
// Чиста VCL версія для прямої інтеграції в Difuz.cpp
//---------------------------------------------------------------------------

#include "predict_integration.h"
#include <vcl.h>
#include <System.SysUtils.hpp>

//---------------------------------------------------------------------------
// VCL версія з String API (найзручніша для C++ Builder)
//---------------------------------------------------------------------------
int PredictFromCurve(double* curve, int curve_length,
                     DeformParams* params,
                     const char* predictor_path)
{
    // Перевірка вхідних даних
    if (!curve || !params || curve_length != 661) {
        return 0;
    }

    // Директорія застосунку
    String appDir = ExtractFilePath(Application->ExeName);

    // Повні шляхи до тимчасових файлів
    String tempCurve = appDir + "temp_curve.txt";
    String tempParams = appDir + "temp_params.txt";

    // Повний шлях до predictor.exe
    String predictorExe;
    if (FileExists(predictor_path)) {
        predictorExe = predictor_path;
    } else {
        predictorExe = appDir + predictor_path;
    }

    if (!FileExists(predictorExe)) {
        return 0;  // Predictor не знайдено
    }

    // 1. Зберегти криву у файл
    try {
        TStringList* curveFile = new TStringList();
        try {
            for (int i = 0; i < curve_length; i++) {
                curveFile->Add(FloatToStrF(curve[i], ffExponent, 6, 0));
            }
            curveFile->SaveToFile(tempCurve);
        }
        __finally {
            delete curveFile;
        }
    }
    catch (...) {
        return 0;
    }

    // 2. Запустити predictor
    // УВАГА: system() блокує UI на 1-2 секунди!
    String cmd = "\"" + predictorExe + "\" \"" + tempCurve + "\" \"" + tempParams + "\"";

    int result = system(AnsiString(cmd).c_str());
    if (result != 0) {
        DeleteFile(tempCurve);
        return 0;
    }

    // 3. Прочитати результат
    if (!FileExists(tempParams)) {
        DeleteFile(tempCurve);
        return 0;
    }

    try {
        TStringList* paramsFile = new TStringList();
        try {
            paramsFile->LoadFromFile(tempParams);

            // Ініціалізувати нулями
            params->Dmax1 = 0.0;
            params->D01 = 0.0;
            params->L1 = 0.0;
            params->Rp1 = 0.0;
            params->D02 = 0.0;
            params->L2 = 0.0;
            params->Rp2 = 0.0;

            // Парсити кожен рядок
            for (int i = 0; i < paramsFile->Count; i++) {
                String line = paramsFile->Strings[i].Trim();

                // Пропустити коментарі та порожні рядки
                if (line.IsEmpty() || line[1] == '#') {
                    continue;
                }

                // Парсити: "Dmax1    0.015234"
                int spacePos = line.Pos(" ");
                if (spacePos > 0) {
                    String name = line.SubString(1, spacePos - 1).Trim();
                    String valueStr = line.SubString(spacePos + 1, line.Length()).Trim();

                    try {
                        double value = StrToFloat(valueStr);

                        if (name == "Dmax1") params->Dmax1 = value;
                        else if (name == "D01") params->D01 = value;
                        else if (name == "L1") params->L1 = value;
                        else if (name == "Rp1") params->Rp1 = value;
                        else if (name == "D02") params->D02 = value;
                        else if (name == "L2") params->L2 = value;
                        else if (name == "Rp2") params->Rp2 = value;
                    }
                    catch (...) {
                        // Пропустити невалідні рядки
                    }
                }
            }
        }
        __finally {
            delete paramsFile;
        }
    }
    catch (...) {
        DeleteFile(tempCurve);
        DeleteFile(tempParams);
        return 0;
    }

    // 4. Видалити тимчасові файли
    DeleteFile(tempCurve);
    DeleteFile(tempParams);

    return 1;  // Успішно
}

//---------------------------------------------------------------------------
// Приклад використання в Difuz.cpp
//---------------------------------------------------------------------------
/*

void __fastcall TForm1::PredictButtonClick(TObject *Sender)
{
    // Підготувати дані кривої (661 точка з R_vseZ[40:701])
    double curve[661];
    for (int i = 0; i < 661; i++) {
        curve[i] = R_vseZ[i + 40];
    }

    // Показати progress indicator
    Cursor = crHourGlass;
    StatusBar->SimpleText = "Predicting parameters...";
    Application->ProcessMessages();

    // Викликати predictor
    DeformParams predicted;
    int success = PredictFromCurve(curve, 661, &predicted, "predict.exe");

    Cursor = crDefault;

    if (success) {
        // Застосувати передбачені параметри
        Edit1->Text = FloatToStrF(predicted.Dmax1, ffFixed, 8, 6);
        Edit2->Text = FloatToStrF(predicted.D01, ffFixed, 8, 6);
        Edit3->Text = FloatToStrF(predicted.L1, ffExponent, 8, 2);
        Edit4->Text = FloatToStrF(predicted.Rp1, ffExponent, 8, 2);
        Edit5->Text = FloatToStrF(predicted.D02, ffFixed, 8, 6);
        Edit6->Text = FloatToStrF(predicted.L2, ffExponent, 8, 2);
        Edit7->Text = FloatToStrF(predicted.Rp2, ffExponent, 8, 2);

        StatusBar->SimpleText = "Prediction successful!";
    } else {
        ShowMessage("Prediction failed! Check if predict.exe exists.");
        StatusBar->SimpleText = "Prediction failed";
    }
}

*/
