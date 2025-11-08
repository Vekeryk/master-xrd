//---------------------------------------------------------------------------
// XRD Prediction Integration - TIMEOUT VERSION
// Використовує CreateProcessA з timeout - найбезпечніший варіант
//---------------------------------------------------------------------------

#include "predict_integration.h"
#include <vcl.h>
#include <System.SysUtils.hpp>
#include <windows.h>  // Для CreateProcessA

//---------------------------------------------------------------------------
// EXTENDED API з timeout
//---------------------------------------------------------------------------
int PredictFromCurveWithTimeout(double* curve, int curve_length,
                                DeformParams* params,
                                const char* predictor_path,
                                DWORD timeout_ms = 30000)  // 30 sec default
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

    // 2. Запустити predictor через CreateProcess з timeout
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    // Приховати вікно консолі
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    // Побудувати командний рядок
    String cmdLine = "\"" + predictorExe + "\" \"" + tempCurve + "\" \"" + tempParams + "\"";

    // CreateProcess МОДИФІКУЄ командний рядок - потрібен writable buffer!
    char cmdBuffer[2048];
    strncpy(cmdBuffer, AnsiString(cmdLine).c_str(), sizeof(cmdBuffer) - 1);
    cmdBuffer[sizeof(cmdBuffer) - 1] = '\0';

    // Запустити процес
    BOOL created = CreateProcessA(
        NULL,           // Ім'я програми (NULL = взяти з cmdLine)
        cmdBuffer,      // Командний рядок (WRITABLE!)
        NULL,           // Security attrs процесу
        NULL,           // Security attrs потоку
        FALSE,          // Не наслідувати handles
        CREATE_NO_WINDOW,  // Без вікна консолі
        NULL,           // Environment
        NULL,           // Current directory
        &si,            // Startup info
        &pi             // Process info (OUT)
    );

    if (!created) {
        DeleteFile(tempCurve);
        return 0;
    }

    // Чекати завершення з timeout
    DWORD waitResult = WaitForSingleObject(pi.hProcess, timeout_ms);

    // Обробити результат
    int success = 0;

    if (waitResult == WAIT_OBJECT_0) {
        // Процес завершився вчасно - перевірити exit code
        DWORD exitCode;
        if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
            success = (exitCode == 0) ? 1 : 0;
        }
    }
    else if (waitResult == WAIT_TIMEOUT) {
        // Timeout! Вбити процес
        TerminateProcess(pi.hProcess, 1);
        success = 0;

        // Можна показати повідомлення:
        // ShowMessage("Prediction timeout! Process killed after " + IntToStr(timeout_ms/1000) + " seconds.");
    }
    else {
        // Помилка очікування
        success = 0;
    }

    // Закрити handles
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    if (!success) {
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
// Зворотньо сумісна функція (використовує timeout 30 сек)
//---------------------------------------------------------------------------
int PredictFromCurve(double* curve, int curve_length,
                     DeformParams* params,
                     const char* predictor_path)
{
    return PredictFromCurveWithTimeout(curve, curve_length, params, predictor_path, 30000);
}

//---------------------------------------------------------------------------
// Приклад використання з кастомним timeout
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

    // ВАРІАНТ 1: Базовий (30 sec timeout)
    DeformParams predicted;
    int success = PredictFromCurve(curve, 661, &predicted, "predict.exe");

    // ВАРІАНТ 2: З кастомним timeout (10 sec)
    // int success = PredictFromCurveWithTimeout(curve, 661, &predicted, "predict.exe", 10000);

    Cursor = crDefault;

    if (success) {
        // Застосувати параметри
        Edit1->Text = FloatToStrF(predicted.Dmax1, ffFixed, 8, 6);
        Edit2->Text = FloatToStrF(predicted.D01, ffFixed, 8, 6);
        Edit3->Text = FloatToStrF(predicted.L1, ffExponent, 8, 2);
        Edit4->Text = FloatToStrF(predicted.Rp1, ffExponent, 8, 2);
        Edit5->Text = FloatToStrF(predicted.D02, ffFixed, 8, 6);
        Edit6->Text = FloatToStrF(predicted.L2, ffExponent, 8, 2);
        Edit7->Text = FloatToStrF(predicted.Rp2, ffExponent, 8, 2);

        StatusBar->SimpleText = "Prediction successful!";
    } else {
        ShowMessage("Prediction failed!\n\n"
                   "Possible reasons:\n"
                   "1. predict.exe not found\n"
                   "2. Timeout (>30 seconds)\n"
                   "3. Invalid curve data\n"
                   "4. Write permission denied");
        StatusBar->SimpleText = "Error";
    }
}

//---------------------------------------------------------------------------
// ADVANCED: Progress bar з periodic UI updates
//---------------------------------------------------------------------------

void __fastcall TForm1::PredictWithProgressClick(TObject *Sender)
{
    double curve[661];
    for (int i = 0; i < 661; i++) {
        curve[i] = R_vseZ[i + 40];
    }

    // Setup UI
    ProgressBar->Position = 0;
    ProgressBar->Max = 100;
    Cursor = crHourGlass;

    // Start prediction in background (would need threading for real progress)
    // For now, just show indeterminate progress
    for (int p = 0; p < 50; p++) {
        ProgressBar->Position = p;
        Application->ProcessMessages();
        Sleep(20);  // Fake progress
    }

    DeformParams predicted;
    int success = PredictFromCurveWithTimeout(curve, 661, &predicted, "predict.exe", 30000);

    ProgressBar->Position = 100;
    Cursor = crDefault;

    if (success) {
        // ... apply parameters ...
    }
}

*/

//---------------------------------------------------------------------------
// BONUS: Non-blocking версія з TThread (для справжнього async виконання)
//---------------------------------------------------------------------------
/*

class TPredictThread : public TThread
{
private:
    double* FCurve;
    DeformParams FResult;
    bool FSuccess;
    String FPredictorPath;

protected:
    void __fastcall Execute()
    {
        FSuccess = PredictFromCurveWithTimeout(
            FCurve, 661, &FResult, FPredictorPath.c_str(), 30000
        );
    }

public:
    __fastcall TPredictThread(double* curve, const String& predictorPath)
        : TThread(true)  // Create suspended
    {
        FCurve = new double[661];
        memcpy(FCurve, curve, 661 * sizeof(double));
        FPredictorPath = predictorPath;
        FSuccess = false;
        FreeOnTerminate = true;
    }

    bool GetSuccess() { return FSuccess; }
    DeformParams GetResult() { return FResult; }
};

void __fastcall TForm1::PredictAsyncButtonClick(TObject *Sender)
{
    double curve[661];
    for (int i = 0; i < 661; i++) {
        curve[i] = R_vseZ[i + 40];
    }

    TPredictThread* thread = new TPredictThread(curve, "predict.exe");

    // Можна підписатись на OnTerminate для callback
    thread->OnTerminate = PredictThreadTerminate;
    thread->Resume();

    StatusBar->SimpleText = "Prediction running in background...";
}

void __fastcall TForm1::PredictThreadTerminate(TObject *Sender)
{
    TPredictThread* thread = (TPredictThread*)Sender;

    if (thread->GetSuccess()) {
        DeformParams result = thread->GetResult();
        // Apply parameters...
        StatusBar->SimpleText = "Async prediction successful!";
    } else {
        StatusBar->SimpleText = "Async prediction failed";
    }
}

*/
