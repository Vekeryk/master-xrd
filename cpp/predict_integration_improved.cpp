//---------------------------------------------------------------------------
// XRD Prediction Integration (IMPROVED VERSION)
// Покращена версія з надійними шляхами для C++ Builder
//---------------------------------------------------------------------------

#include "predict_integration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __BORLANDC__  // C++ Builder
#include <vcl.h>     // Для Application->ExeName
#endif

//---------------------------------------------------------------------------
// Helper: Створити повний шлях до файлу в директорії застосунку
//---------------------------------------------------------------------------
static void GetFullPath(char* buffer, int size, const char* filename)
{
#ifdef __BORLANDC__  // C++ Builder VCL
    // ВАРІАНТ 1 (РЕКОМЕНДОВАНИЙ): Використовувати директорію застосунку
    String appDir = ExtractFilePath(Application->ExeName);
    String fullPath = appDir + filename;
    strncpy(buffer, fullPath.c_str(), size - 1);
    buffer[size - 1] = '\0';
#else
    // ВАРІАНТ 2: Використовувати TEMP директорію Windows
    char* temp_dir = getenv("TEMP");
    if (temp_dir) {
        snprintf(buffer, size, "%s\\%s", temp_dir, filename);
    } else {
        // Fallback: поточна директорія (ненадійно!)
        strncpy(buffer, filename, size - 1);
        buffer[size - 1] = '\0';
    }
#endif
}

//---------------------------------------------------------------------------
// Predict параметри з кривої
//---------------------------------------------------------------------------
int PredictFromCurve(double* curve, int curve_length,
                     DeformParams* params,
                     const char* predictor_path)
{
    // Перевірка вхідних даних
    if (!curve || !params || curve_length != 661) {
        return 0;
    }

    // Створити повні шляхи до тимчасових файлів
    char temp_curve[512];
    char temp_params[512];
    GetFullPath(temp_curve, sizeof(temp_curve), "temp_curve.txt");
    GetFullPath(temp_params, sizeof(temp_params), "temp_params.txt");

    // Якщо predictor_path - відносний шлях, зробити його повним
    char predictor_full[512];
    if (predictor_path[0] != '/' && predictor_path[1] != ':') {
        // Відносний шлях - додати директорію застосунку
        GetFullPath(predictor_full, sizeof(predictor_full), predictor_path);
    } else {
        // Вже повний шлях
        strncpy(predictor_full, predictor_path, sizeof(predictor_full) - 1);
        predictor_full[sizeof(predictor_full) - 1] = '\0';
    }

    // 1. Зберегти криву у файл
    FILE* f = fopen(temp_curve, "w");
    if (!f) {
        return 0;
    }

    for (int i = 0; i < curve_length; i++) {
        fprintf(f, "%.6e\n", curve[i]);
    }
    fclose(f);

    // 2. Запустити predictor
    // ВАЖЛИВО: system() БЛОКУЄ UI на 1-2 секунди!
    // Для non-blocking виконання використовуй CreateProcess() або запускай в окремому потоці
    char cmd[1024];
    sprintf(cmd, "\"%s\" \"%s\" \"%s\"", predictor_full, temp_curve, temp_params);

    int result = system(cmd);
    if (result != 0) {
        remove(temp_curve);
        return 0;
    }

    // 3. Прочитати результат
    f = fopen(temp_params, "r");
    if (!f) {
        remove(temp_curve);
        return 0;
    }

    // Ініціалізувати параметри нулями
    params->Dmax1 = 0.0;
    params->D01 = 0.0;
    params->L1 = 0.0;
    params->Rp1 = 0.0;
    params->D02 = 0.0;
    params->L2 = 0.0;
    params->Rp2 = 0.0;

    // Прочитати файл
    char line[256];
    char name[32];
    double value;

    while (fgets(line, sizeof(line), f)) {
        // Пропустити коментарі та порожні рядки
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        // Парсити рядок: <name> <value>
        if (sscanf(line, "%s %lf", name, &value) == 2) {
            if (strcmp(name, "Dmax1") == 0) {
                params->Dmax1 = value;
            } else if (strcmp(name, "D01") == 0) {
                params->D01 = value;
            } else if (strcmp(name, "L1") == 0) {
                params->L1 = value;
            } else if (strcmp(name, "Rp1") == 0) {
                params->Rp1 = value;
            } else if (strcmp(name, "D02") == 0) {
                params->D02 = value;
            } else if (strcmp(name, "L2") == 0) {
                params->L2 = value;
            } else if (strcmp(name, "Rp2") == 0) {
                params->Rp2 = value;
            }
        }
    }
    fclose(f);

    // 4. Видалити тимчасові файли
    remove(temp_curve);
    remove(temp_params);

    return 1;  // Успішно
}
