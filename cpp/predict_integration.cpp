//---------------------------------------------------------------------------
// XRD Prediction Integration
// Максимально простий C-style код для виклику Python predictor
//---------------------------------------------------------------------------

#include "predict_integration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//---------------------------------------------------------------------------
// Predict параметри з кривої
//---------------------------------------------------------------------------
int PredictFromCurve(double *curve, int curve_length,
                     DeformParams *params,
                     const char *predictor_path)
{
    char *ss;
    ss = new char[1000];

    AnsiString MyFName3 = "";
    if (SaveKDB1->Execute())
    {
        MyFName3 = SaveKDB1->FileName;
        TStringList *List3 = new TStringList;

        for (int q = 0; q <= m1_[1]; q++)
        {
            // w = TetaMin + q * ik_[1];
            // sprintf(ss, "%3.6lf\t%3.6le", w, intIk2d[q][1]);
            sprintf(ss, "%3.6le", intIk2d[q][1]);
            List3->Add(ss);
        }

        List3->SaveToFile(SaveKDB1->FileName);
    }

    // Перевірка вхідних даних
    if (!curve || !params || curve_length != 661)
    {
        return 0;
    }

    // Тимчасові файли
    const char *temp_curve = "temp_curve.txt";
    const char *temp_params = "temp_params.txt";

    // 1. Зберегти криву у файл
    FILE *f = fopen(temp_curve, "w");
    if (!f)
    {
        return 0;
    }

    for (int i = 0; i < curve_length; i++)
    {
        fprintf(f, "%.6e\n", curve[i]);
    }
    fclose(f);

    // 2. Запустити predictor
    char cmd[512];
    sprintf(cmd, "\"%s\" \"%s\" \"%s\"", predictor_path, temp_curve, temp_params);

    int result = system(cmd);
    if (result != 0)
    {
        remove(temp_curve);
        return 0;
    }

    // 3. Прочитати результат
    f = fopen(temp_params, "r");
    if (!f)
    {
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

    while (fgets(line, sizeof(line), f))
    {
        // Пропустити коментарі та порожні рядки
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        // Парсити рядок: <name> <value>
        if (sscanf(line, "%s %lf", name, &value) == 2)
        {
            if (strcmp(name, "Dmax1") == 0)
            {
                params->Dmax1 = value;
            }
            else if (strcmp(name, "D01") == 0)
            {
                params->D01 = value;
            }
            else if (strcmp(name, "L1") == 0)
            {
                params->L1 = value;
            }
            else if (strcmp(name, "Rp1") == 0)
            {
                params->Rp1 = value;
            }
            else if (strcmp(name, "D02") == 0)
            {
                params->D02 = value;
            }
            else if (strcmp(name, "L2") == 0)
            {
                params->L2 = value;
            }
            else if (strcmp(name, "Rp2") == 0)
            {
                params->Rp2 = value;
            }
        }
    }
    fclose(f);

    // 4. Видалити тимчасові файли
    remove(temp_curve);
    remove(temp_params);

    return 1; // Успішно
}
