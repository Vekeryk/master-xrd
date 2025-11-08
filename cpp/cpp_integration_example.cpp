/**
 * C++ Integration Example for XRD Predictor
 * ==========================================
 * Shows how to call predict.exe from C++ and parse results.
 *
 * МАКСИМАЛЬНО ПРОСТО - без JSON, без libcurl, тільки subprocess + text files!
 *
 * Compilation (GCC/Clang):
 *   g++ -std=c++11 cpp_integration_example.cpp -o test_predictor
 *
 * Compilation (MSVC):
 *   cl /EHsc cpp_integration_example.cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>

// Структура параметрів деформації (як у твоєму Difuz.cpp)
struct DeformationProfile {
    double Dmax1;  // Максимальна деформація (асим. гаусіана)
    double D01;    // Деформація на поверхні
    double L1;     // Товщина порушеного шару (см)
    double Rp1;    // Позиція максимуму (см)
    double D02;    // Деформація (спадна гаусіана)
    double L2;     // Товщина
    double Rp2;    // Позиція максимуму

    void print() const {
        std::cout << "Deformation Parameters:\n";
        std::cout << "  Dmax1 = " << Dmax1 << "\n";
        std::cout << "  D01   = " << D01 << "\n";
        std::cout << "  L1    = " << L1 << " cm\n";
        std::cout << "  Rp1   = " << Rp1 << " cm\n";
        std::cout << "  D02   = " << D02 << "\n";
        std::cout << "  L2    = " << L2 << " cm\n";
        std::cout << "  Rp2   = " << Rp2 << " cm\n";
    }
};

/**
 * Зберегти криву у текстовий файл
 *
 * Format: 661 float values, one per line
 */
bool saveCurveToFile(const std::vector<double>& curve, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file for writing: " << filepath << std::endl;
        return false;
    }

    for (const double& val : curve) {
        file << val << "\n";
    }

    file.close();
    return true;
}

/**
 * Запустити predict.exe і зачекати завершення
 *
 * Returns: 0 if success, non-zero if error
 */
int runPredictor(const std::string& predictorPath,
                 const std::string& inputPath,
                 const std::string& outputPath) {

    // Побудувати команду
    std::stringstream cmd;
    cmd << "\"" << predictorPath << "\" \"" << inputPath << "\" \"" << outputPath << "\"";

    std::cout << "Running: " << cmd.str() << std::endl;

    // Виконати команду (блокуючий виклик)
    int result = std::system(cmd.str().c_str());

    return result;
}

/**
 * Прочитати параметри з файлу
 *
 * Format:
 *   # Comments start with #
 *   Dmax1    0.012345
 *   D01      0.006789
 *   ...
 */
bool loadParamsFromFile(const std::string& filepath, DeformationProfile& params) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file for reading: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse line: <name> <value>
        std::istringstream iss(line);
        std::string name;
        double value;

        if (!(iss >> name >> value)) {
            std::cerr << "WARNING: Cannot parse line: " << line << std::endl;
            continue;
        }

        // Map to struct fields
        if (name == "Dmax1") {
            params.Dmax1 = value;
        } else if (name == "D01") {
            params.D01 = value;
        } else if (name == "L1") {
            params.L1 = value;
        } else if (name == "Rp1") {
            params.Rp1 = value;
        } else if (name == "D02") {
            params.D02 = value;
        } else if (name == "L2") {
            params.L2 = value;
        } else if (name == "Rp2") {
            params.Rp2 = value;
        }
    }

    file.close();
    return true;
}

/**
 * Головна функція - predict параметри з кривої
 *
 * ЦЕ ТЕ ЩО ТОБІ ТРЕБА ДОДАТИ У DIFUZ.CPP!
 */
bool predictDeformationProfile(const std::vector<double>& expCurve,
                                DeformationProfile& predictedParams,
                                const std::string& predictorPath = "./predict.exe") {

    // 1. Зберегти криву у тимчасовий файл
    std::string tempCurveFile = "temp_curve.txt";
    if (!saveCurveToFile(expCurve, tempCurveFile)) {
        return false;
    }

    // 2. Запустити predictor
    std::string tempParamsFile = "temp_params.txt";
    int result = runPredictor(predictorPath, tempCurveFile, tempParamsFile);

    if (result != 0) {
        std::cerr << "ERROR: Predictor failed with code " << result << std::endl;
        std::remove(tempCurveFile.c_str());
        return false;
    }

    // 3. Прочитати результат
    bool success = loadParamsFromFile(tempParamsFile, predictedParams);

    // 4. Видалити тимчасові файли
    std::remove(tempCurveFile.c_str());
    std::remove(tempParamsFile.c_str());

    return success;
}

/**
 * ПРИКЛАД ВИКОРИСТАННЯ У DIFUZ.CPP
 */
void OnPredictButtonClick() {
    std::cout << "=== XRD Prediction Example ===" << std::endl;

    // 1. Отримати експериментальну криву (661 точка після crop [40:701])
    //    У твоєму Difuz.cpp це буде R_vseZ[40:701]
    std::vector<double> experimentalCurve(661);

    // ПРИКЛАД: Завантажити з файлу експерименту
    // (У реальності візьмеш з пам'яті програми)
    std::ifstream expFile("experiments/experiment.txt");
    if (!expFile.is_open()) {
        std::cerr << "ERROR: Cannot open experiment file!" << std::endl;
        return;
    }

    // Прочитати лише Y values (skip X)
    std::string line;
    int count = 0;
    while (std::getline(expFile, line) && count < 701) {
        std::istringstream iss(line);
        double x, y;
        if (iss >> x >> y) {
            if (count >= 40) {  // Skip перші 40 точок (GGG peak)
                experimentalCurve[count - 40] = y;
            }
            count++;
        }
    }
    expFile.close();

    std::cout << "Loaded " << experimentalCurve.size() << " points from experiment" << std::endl;

    // 2. Predict параметри
    DeformationProfile predicted;
    bool success = predictDeformationProfile(
        experimentalCurve,
        predicted,
        "./predict.exe"  // Або "dist/predict.exe"
    );

    if (!success) {
        std::cerr << "ERROR: Prediction failed!" << std::endl;
        return;
    }

    // 3. Використати результат
    std::cout << "\n✅ Prediction successful!\n" << std::endl;
    predicted.print();

    // 4. Заповнити поля GUI
    // У Difuz.cpp щось типу:
    // SetDlgItemText(IDC_EDIT_DMAX1, std::to_string(predicted.Dmax1).c_str());
    // SetDlgItemText(IDC_EDIT_D01, std::to_string(predicted.D01).c_str());
    // ... і т.д.

    std::cout << "\n💡 Now fill GUI fields with these values!" << std::endl;
}

int main() {
    // Тестовий приклад
    OnPredictButtonClick();
    return 0;
}

/*
 * ============================================================================
 * INTEGRATION INTO DIFUZ.CPP
 * ============================================================================
 *
 * 1. Скопіюй функції:
 *    - saveCurveToFile()
 *    - runPredictor()
 *    - loadParamsFromFile()
 *    - predictDeformationProfile()
 *
 * 2. У обробнику кнопки "Predict":
 *
 *    void CYourDialog::OnBnClickedPredict() {
 *        // Отримати експериментальну криву
 *        std::vector<double> curve(661);
 *        for (int i = 0; i < 661; i++) {
 *            curve[i] = R_vseZ[i + 40];  // Crop [40:701]
 *        }
 *
 *        // Predict
 *        DeformationProfile params;
 *        if (predictDeformationProfile(curve, params, "predict.exe")) {
 *            // Заповнити поля
 *            SetDlgItemText(IDC_EDIT_DMAX1, CString(std::to_string(params.Dmax1).c_str()));
 *            SetDlgItemText(IDC_EDIT_D01, CString(std::to_string(params.D01).c_str()));
 *            SetDlgItemText(IDC_EDIT_L1, CString(std::to_string(params.L1).c_str()));
 *            SetDlgItemText(IDC_EDIT_RP1, CString(std::to_string(params.Rp1).c_str()));
 *            SetDlgItemText(IDC_EDIT_D02, CString(std::to_string(params.D02).c_str()));
 *            SetDlgItemText(IDC_EDIT_L2, CString(std::to_string(params.L2).c_str()));
 *            SetDlgItemText(IDC_EDIT_RP2, CString(std::to_string(params.Rp2).c_str()));
 *
 *            MessageBox("Parameters predicted successfully!", "Success", MB_OK);
 *        } else {
 *            MessageBox("Prediction failed!", "Error", MB_OK | MB_ICONERROR);
 *        }
 *    }
 *
 * 3. Переконайся що predict.exe знаходиться у тій самій папці що і Difuz.exe
 *
 * ============================================================================
 */
