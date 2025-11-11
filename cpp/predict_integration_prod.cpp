void __fastcall TForm1::Button3Click(TObject *Sender)
{
  AnsiString oldCap = Button3->Caption;
  bool oldEn = Button3->Enabled;
  try
  {
    Button3->Caption = "Àâòîï³äá³ð…";
    Button3->Enabled = false;
    Button3->Repaint();
    Application->ProcessMessages();

    AnsiString appDir = ExtractFilePath(Application->ExeName);
    AnsiString predictExe = appDir + "predict.exe";
    AnsiString model = appDir + "predict_target_model.pt";
    AnsiString inFile = appDir + "predict_curve.txt";
    AnsiString outFile = appDir + "predict_params.txt";

    if (!FileExists(predictExe.c_str()))
    {
      ShowMessage("Predict exe íå çíàéäåíî!");
      return;
    }

    if (!FileExists(model.c_str()))
    {
      ShowMessage("Predict model íå çíàéäåíî!");
      return;
    }

    char *valueChars;
    valueChars = new char[100];

    TStringList *ListExperiment = new TStringList;
    for (int q = 0; q <= m1_[1]; q++)
    {
      // w = TetaMin + q * ik_[1];
      // sprintf(ss, "%3.6lf\t%3.6le", w, intIk2d[q][1]);
      sprintf(valueChars, "%3.6le", intIk2d[q][1]);
      ListExperiment->Add(valueChars);
    }
    ListExperiment->SaveToFile(inFile);

    if (!FileExists(inFile.c_str()))
    {
      ShowMessage("Âõ³äíèé ôàéë ç êðèâîþ íå çíàéäåíî!");
      return;
    }

    AnsiString args = "\"" + model + "\" \"" + inFile + "\" \"" + outFile + "\"";

    // --- çàïóñê ç î÷³êóâàííÿì (ShellExecuteEx + MsgWait) ---
    SHELLEXECUTEINFOA sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    sei.hwnd = Application->Handle;
    sei.lpVerb = "open";
    sei.lpFile = predictExe.c_str();
    sei.lpParameters = args.c_str();
    sei.lpDirectory = appDir.c_str(); // ðîáî÷à òåêà = äå ëåæèòü exe/ìîäåëü/IO
    sei.nShow = SW_HIDE;              // íå ïîêàçóâàòè êîíñîëüíå â³êíî

    if (!ShellExecuteExA(&sei))
    {
      ShowMessage("Íå âäàëîñÿ çàïóñòèòè predict.exe");
      return;
    }

    // ÷åêàòè áåç áëîêóâàííÿ UI
    for (;;)
    {
      DWORD w = MsgWaitForMultipleObjects(1, &sei.hProcess, FALSE, 50, QS_ALLINPUT);
      if (w == WAIT_OBJECT_0)
        break; // ïðîöåñ çàâåðøèâñÿ
      Application->ProcessMessages();
    }

    DWORD exitCode = 1;
    GetExitCodeProcess(sei.hProcess, &exitCode);
    CloseHandle(sei.hProcess);

    if (exitCode != 0)
    {
      ShowMessage("predict.exe çàâåðøèâñÿ ç ïîìèëêîþ.");
      return;
    }

    if (!FileExists(outFile))
    {
      ::MessageBoxW(Application->Handle,
                    L"Ôàéë ðåçóëüòàò³â predict_params.txt íå çíàéäåíî.",
                    L"Ïîìèëêà", MB_ICONERROR | MB_OK);
    }
    else
    {
      // ÷èòàºìî 7 ðÿäê³â ç ÷èñëàìè
      std::auto_ptr<TStringList> pl(new TStringList());
      pl->LoadFromFile(outFile);

      // ñòðàõóºìîñÿ: êðàùå ð³âíî 7
      if (pl->Count >= 7)
      {
        // ïàðñèìî ÿê C-÷èñëà ç êðàïêîþ
        double Dmax1 = atof(pl->Strings[0].c_str());
        double D01 = atof(pl->Strings[1].c_str());
        double L1_cm = atof(pl->Strings[2].c_str());
        double Rp1_cm = atof(pl->Strings[3].c_str());
        double D02 = atof(pl->Strings[4].c_str());
        double L2_cm = atof(pl->Strings[5].c_str());
        double Rp2_cm = atof(pl->Strings[6].c_str());

        // ïåðåâåäåííÿ äîâæèí ç ñì ó A
        const double CM_TO_ANG = 1e8;
        long L1A = (long)floor(L1_cm * CM_TO_ANG + 0.5);
        long Rp1A = (long)floor(Rp1_cm * CM_TO_ANG + 0.5);
        long L2A = (long)floor(L2_cm * CM_TO_ANG + 0.5);
        long Rp2A = (long)floor(Rp2_cm * CM_TO_ANG + 0.5);

        // ôîðìóºìî 7 ðÿäê³â
        char msg[512];
        // ïîêàçóºìî êðàïêó ÿê äåñÿòêîâèé ðîçä³ëüíèê
        const char oldDec = DecimalSeparator;
        DecimalSeparator = '.';
        snprintf(msg, sizeof(msg),
                 "Dmax1 = %.6f\n"
                 "D01 = %.6f\n"
                 "L1 (A) = %ld\n"
                 "Rp1 (A) = %ld\n"
                 "D02 = %.6f\n"
                 "L2 (A) = %ld\n"
                 "Rp2 (A) = %ld",
                 Dmax1, D01, L1A, Rp1A, D02, L2A, Rp2A);
        DecimalSeparator = oldDec;

        WideString wmsg = msg;
        ::MessageBoxW(Application->Handle, wmsg.c_bstr(),
                      L"Ðåçóëüòàò àâòîï³äáîðó", MB_ICONINFORMATION | MB_OK);
      }
      else
      {
        ::MessageBoxW(Application->Handle,
                      L"Íåäîñòàòíüî ðÿäê³â ó predict_params.txt (î÷³êóºòüñÿ 7).",
                      L"Ïîìèëêà", MB_ICONERROR | MB_OK);
      }
    }
  }
  __finally
  {
    Button3->Caption = oldCap;
    Button3->Enabled = oldEn;
    Button3->Repaint();
    Application->ProcessMessages();
  }
}