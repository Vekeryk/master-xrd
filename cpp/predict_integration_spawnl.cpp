//---------------------------------------------------------------------------
// XRD Prediction Integration - SPAWNL VERSION
// Використовує _spawnl замість system() - безпечніше і без проблем з кавичками
//---------------------------------------------------------------------------

#include "predict_integration.h"
#include <vcl.h>
#include <System.SysUtils.hpp>
#include <process.h>  // Для _spawnl

//---------------------------------------------------------------------------
// Версія з _spawnl - проста і безпечна
// ✅ Аргументи передаються як масив (не потрібні кавички)
// ✅ Безпечно для шляхів з пробілами
// ✅ Блокує UI, але простіше ніж CreateProcess
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

    // 2. Запустити predictor через _spawnl
    // _spawnl автоматично екранує аргументи - НЕ ПОТРІБНІ КАВИЧКИ!
    // _P_WAIT = блокує до завершення (як system, але безпечніше)
    AnsiString exePath = predictorExe;
    AnsiString curvePath = tempCurve;
    AnsiString paramsPath = tempParams;

    int result = _spawnl(
        _P_WAIT,                    // Чекати завершення
        exePath.c_str(),            // Шлях до .exe
        exePath.c_str(),            // argv[0] (ім'я програми)
        curvePath.c_str(),          // argv[1] (input curve)
        paramsPath.c_str(),         // argv[2] (output params)
        NULL                        // Кінець аргументів
    );

    // _spawnl повертає:
    //   >= 0  - успіх (exit code процесу)
    //   -1    - помилка запуску
    if (result == -1) {
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
// Приклад використання
//---------------------------------------------------------------------------
/*

void __fastcall TForm1::PredictButtonClick(TObject *Sender)
{
    // Підготувати дані кривої
    double curve[661];
    for (int i = 0; i < 661; i++) {
        curve[i] = R_vseZ[i + 40];
    }

    // Показати індикатор
    Cursor = crHourGlass;
    StatusBar->SimpleText = "ML prediction...";
    Application->ProcessMessages();

    // Predict (блокує UI на 1-2 сек, але безпечно)
    DeformParams predicted;
    int success = PredictFromCurve(curve, 661, &predicted, "predict.exe");

    Cursor = crDefault;

    if (success) {
        // Застосувати параметри
        Edit1->Text = FloatToStrF(predicted.Dmax1, ffFixed, 8, 6);
        Edit2->Text = FloatToStrF(predicted.D01, ffFixed, 8, 6);
        // ... інші параметри ...
        StatusBar->SimpleText = "Success!";
    } else {
        ShowMessage("Prediction failed!");
    }
}

*/
