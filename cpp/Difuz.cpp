//---------------------------------------------------------------------------

#include <vcl.h>
#pragma hdrstop
#include <Math.h>
#include "math.h"
#include "math.hpp"
#include <vcl.h>
#include<stdio.h>
#include<stdlib.h>
#include<fcntl.h>
#include<sys\stat.h>
#include<io.h>
#include <math.h>
#include<string.h>
#include<complex.h>
#include<complex>
#include <cmath>
#include <iostream>
#include "Difuz.h"
#include "get_narray/str_tools.h"  // import ALL
#include "get_narray/file_tools.h" // import ALL
#include "get_narray/get_narray.h" // import ALL
//#include  <class T>
typedef double                T_real;
typedef std::complex<T_real>  T_compl;

//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TForm1 *Form1;
//---------------------------------------------------------------------------

void TForm1::clearArrays()
{
    // Спочатку видаляємо пам'ять під масиви старого розміру
//    delete this-> DeltaTeta;
    delete this->MuDSsum;
//    delete this-> DD;

}
//---------------------------------------------------------------------------
void TForm1::reinitArrays()
{
    // Спочатку видаляємо пам'ять під масиви старого розміру
    this->clearArrays();

    // Тепер виділяємо пам'ять під масив нового розміру
    // P.S. Тут бажано би перевіряти чи пам'ять виділилася...
//    this->DeltaTeta = new double[this->MM2 ];
    this->MuDSsum = new double[ this->MM1 ];
   // this->DD = new double[ this->KM1 ];
    //this->f = new double[ this->KM1 ];
//double *uh = new double[MM1];
//  this->DeltaTeta = new double[ this->MM1 ];
    // ... наприклад так
    // if (!this->MuDSsum ) {
    //          //throw new Exception()....
    //  showMessage("А-а-а-а... Караул!!!!")
    //}

    // Ініціалізуємо масиви нулями (про всяк випадок)
/*   for( unsigned long i = 0; i< this->MM1; i++ )
    {
        this->DeltaTeta[i] = 33;
        this->MuDSsum[i] = 2;
    }  */
//Label158->Caption=FloatToStr(DeltaTeta[1]);
}

//---------------------------------------------------------------------------
__fastcall TForm1::TForm1(TComponent* Owner)
        : TForm(Owner)
{
// SetWindowLong(Button23->Handle, GWL_STYLE,  // Чтобы вывести текст на кнопке в несколько строк
// GetWindowLong(Button23->Handle, GWL_STYLE) | BS_MULTILINE);
// Button23->Caption="Швидкий \n старт";

 // Ініціалізація змінних
// this->MM2 = MM;
 this->MM1 = MM;
 //this->KM1 = 50;
 //this->MuDSsum = new double[ this->MM1 ];
 this->reinitArrays();
}


/*void __fastcall TForm1::FormCreate(TObject *Sender)
{  Чтобы вывести текст на кнопке в несколько строк  http://cppbuilder.ru/articles/0092.php
    SetWindowLong(Button23->Handle, GWL_STYLE,
      GetWindowLong(Button23->Handle, GWL_STYLE) | BS_MULTILINE);
    Button23->Caption="Швидкий\nc старт";
}  */
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void __fastcall TForm1::Button13Click(TObject *Sender)  // Start
{
//MM=StrToInt(Edit27->Text);//Label157->Caption="fff";
Label156->Caption=FloatToStr(MM);
Label157->Caption=FloatToStr(KM);
//Chart3->LeftAxis->Automatic = False ;
//Chart3->LeftAxis->Maximum = StrToFloat(Edit192->Text); //гарно діє!!!!!!!!
//Chart3->LeftAxis->Minimum = 0 ;

//Занулення всіх даних:
//for (int i=0; i<=MM; i++)
//{
//R_dif[i]=0;
//R_cogerTT[i]=0;
//R_vse[i]=0;
//R_vseZ[i]=0;
//R_cogerTT_dla[i]=0;
//}
for (int i=0; i<=3*MM; i++)
{
PE[i]=0;
R_vseZa[i]=0;
}
    Memo1->Clear();
//    Memo2->Clear();
 //   Memo3->Clear();
//    Memo4->Clear();
    Memo8->Clear();
    Memo9->Clear();
if (CheckBox38->Checked==true)    Memo5->Clear();
if (CheckBox38->Checked==true)    Memo6->Clear();
if (CheckBox27->Checked==true)    Memo7->Clear();
//Series1->Clear();
//Series11->Clear();
Series8->Clear();
Series9->Clear();
Series15->Clear();
Series4->Clear();
Series13->Clear();
Series6->Clear();
Series14->Clear();
Series10->Clear();
Series12->Clear();
Series7->Clear();
Series16->Clear();
Series17->Clear();
Series27->Clear();
Series28->Clear();
Series32->Clear();
Series33->Clear();
Series5->Clear();
Series30->Clear();
Series31->Clear();
Series3->Clear();
Series34->Clear();
Series35->Clear();
Series36->Clear();
Series37->Clear();
Series38->Clear();
Series39->Clear();
Series40->Clear();
Series41->Clear();
Series42->Clear();
Series43->Clear();
Series44->Clear();
//Series45->Clear();
Series46->Clear();
Series47->Clear();
Series48->Clear();
Series49->Clear();
Series50->Clear();
Series52->Clear();
Series22->Clear();
Series23->Clear();
//Series27->Clear();
Series29->Clear();
Series54->Clear();
Series55->Clear();
Series56->Clear();
Series57->Clear();
Series58->Clear();

//Form2->Series1->Clear();
//Form2->Series2->Clear();
//Form2->Series3->Clear();


Edit99->Text="";
Edit327->Text="";
Edit328->Text="";
Edit100->Text="";
Edit343->Text="";

Edit17->Text="";
Edit18->Text="";
Edit378->Text="";
Edit379->Text="";
Edit380->Text="";
Edit381->Text="";
Edit58->Text="";
Edit59->Text="";
Edit190->Text="";
Edit191->Text="";

Edit184->Text="";
Edit185->Text="";
Edit186->Text="";
Edit187->Text="";
Edit188->Text="";
Edit189->Text="";

// Дані про дефекти в монокристалі:
Defekts_mon[1]=1e-8*StrToFloat(Edit54->Text);  // R001=
Defekts_mon[2]=StrToFloat(Edit53->Text);       // nL01=
Defekts_mon[4]=1e-8*StrToFloat(Edit65->Text);  // R002=
Defekts_mon[5]=StrToFloat(Edit64->Text);       // nL02=
Defekts_mon[7]=1e-8*StrToFloat(Edit51->Text);  // R0p01=
Defekts_mon[8]=StrToFloat(Edit50->Text);       // np01=
Defekts_mon[9]=StrToFloat(Edit52->Text);       // eps01=
Defekts_mon[10]=1e-8*StrToFloat(Edit47->Text); // R0p02=
Defekts_mon[11]=StrToFloat(Edit46->Text);      // np02=
Defekts_mon[12]=StrToFloat(Edit60->Text);      // eps02=
Defekts_mon[13]=1e-8*StrToFloat(Edit56->Text); // R0d0=
Defekts_mon[14]=StrToFloat(Edit55->Text);      // nd0=
Defekts_mon[15]=StrToFloat(Edit57->Text);      // eps0d=
Defekts_mon[16]=1e-8*StrToFloat(Edit62->Text); // R0p0td=
Defekts_mon[17]=StrToFloat(Edit61->Text);      // np0td=
Defekts_mon[18]=StrToFloat(Edit63->Text);      // eps0td=
Defekts_mon[19]=1e-8*StrToFloat(Edit216->Text);  // R00_an=
Defekts_mon[20]=StrToFloat(Edit192->Text);       // nL0_an=

// Дані про дефекти в плівці:
Defekts_film[1]=1e-8*StrToFloat(Edit175->Text);   // R001pl=
Defekts_film[2]=StrToFloat(Edit174->Text);        // nL01pl=
Defekts_film[4]=1e-8*StrToFloat(Edit177->Text);   // R002pl=
Defekts_film[5]=StrToFloat(Edit176->Text);        // nL02pl=
Defekts_film[7]=1e-8*StrToFloat(Edit169->Text);   // R0p01pl=
Defekts_film[8]=StrToFloat(Edit168->Text);        // np01pl=
Defekts_film[9]=StrToFloat(Edit170->Text);        // eps01pl=
Defekts_film[10]=1e-8*StrToFloat(Edit172->Text);  // R0p02pl=
Defekts_film[11]=StrToFloat(Edit171->Text);       // np02pl=
Defekts_film[12]=StrToFloat(Edit173->Text);       // eps02pl=
Defekts_film[13]=1e-8*StrToFloat(Edit179->Text);  // R0d0pl=
Defekts_film[14]=StrToFloat(Edit178->Text);       // nd0pl=
Defekts_film[15]=StrToFloat(Edit180->Text);       // eps0dpl=
Defekts_film[16]=1e-8*StrToFloat(Edit182->Text);  // R0p0tdpl=
Defekts_film[17]=StrToFloat(Edit181->Text);       // np0tdpl=
Defekts_film[18]=StrToFloat(Edit183->Text);       // eps0tdpl=

// Дані про дефекти в профілі:
Defekts_SL[1]=1e-8*StrToFloat(Edit3->Text);   // R0_max=
Defekts_SL[2]=StrToFloat(Edit2->Text);        // nL_max=
Defekts_SL[4]=1e-8*StrToFloat(Edit15->Text);  // R0p_max=
Defekts_SL[5]=StrToFloat(Edit14->Text);       // np_max=
Defekts_SL[6]=StrToFloat(Edit16->Text);       // eps=
Defekts_SL[7]=1e-8*StrToFloat(Edit25->Text);  // R0d_max=
Defekts_SL[8]=StrToFloat(Edit24->Text);       // nd_max=
Defekts_SL[9]=StrToFloat(Edit26->Text);       // epsd=
Defekts_SL[10]=1e-8*StrToFloat(Edit251->Text);  // R0pеtd_max=
Defekts_SL[11]=StrToFloat(Edit250->Text);       // nptd_max=
Defekts_SL[12]=StrToFloat(Edit252->Text);       // epstd=
Defekts_SL[13]=1e-8*StrToFloat(Edit219->Text);  // R0_max_an=
Defekts_SL[14]=StrToFloat(Edit218->Text);       // nL_max_an=
//Defekts_SL[12]=StrToFloat(Edit252->Text);       // epstd=

//number_KDV=StrToInt(Edit133->Text);
if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==false) number_KDV=1;
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==false) number_KDV=1;
if (CheckBox42->Checked==false && CheckBox43->Checked==false && CheckBox44->Checked==true) number_KDV=1;
if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==false) {number_KDV=2;}
if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==true) {number_KDV=3;}//2;  !!!!!!!!!!!!!!!
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==true) {number_KDV=3;}//2;  !!!!!!!!!!!!!!!!!
if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==true) {number_KDV=3; }
Edit146->Text=IntToStr(number_KDV);


double nskv1, kskv1,nskv2, kskv2,nskv3, kskv3;
double nskv1_r, kskv1_r,nskv2_r, kskv2_r,nskv3_r, kskv3_r;

if (vved_exper!=1 && vved_exper!=2) vved_exper=0;  //  межі кута і крок задані однакові для всіх КДВ в програмі

if (vved_exper==1 || vved_exper==2)  // Треба зчитувати кожен раз і все  !!!!!!!!!!!!!!!!!!!!!!!!!!
{
//if (KDV_lich==1)
//{
nskv1=StrToFloat(Edit69->Text);
kskv1=StrToFloat(Edit70->Text);
if (CheckBox22->Checked==true) nskv1_r=StrToFloat(Edit140->Text);
if (CheckBox22->Checked==true) kskv1_r=StrToFloat(Edit141->Text);
if (CheckBox42->Checked==true)
{
nskvi1=Ceil(nskv1/ik_[1]);
kskvi1=Ceil(kskv1/ik_[1]);
if (CheckBox22->Checked==true) nskvi1_r=Ceil(nskv1_r/ik_[1]);
if (CheckBox22->Checked==true) kskvi1_r=Ceil(kskv1_r/ik_[1]);
}  //}
//if (KDV_lich==2)
//{
nskv2=StrToFloat(Edit130->Text);
kskv2=StrToFloat(Edit129->Text);
if (CheckBox45->Checked==true) nskv2_r=StrToFloat(Edit142->Text);
if (CheckBox45->Checked==true) kskv2_r=StrToFloat(Edit143->Text);
if (CheckBox43->Checked==true)
{
nskvi2=Ceil(nskv2/ik_[2]);
kskvi2=Ceil(kskv2/ik_[2]);
if (CheckBox45->Checked==true) nskvi2_r=Ceil(nskv2_r/ik_[2]);
if (CheckBox45->Checked==true) kskvi2_r=Ceil(kskv2_r/ik_[2]);
}  //}
//if (KDV_lich==3)
//{
nskv3=StrToFloat(Edit89->Text);
kskv3=StrToFloat(Edit134->Text);
if (CheckBox46->Checked==true) nskv3_r=StrToFloat(Edit144->Text);
if (CheckBox46->Checked==true) kskv3_r=StrToFloat(Edit145->Text);
if (CheckBox44->Checked==true)
{
nskvi3=Ceil(nskv3/ik_[3]);
kskvi3=Ceil(kskv3/ik_[3]);
if (CheckBox46->Checked==true) nskvi3_r=Ceil(nskv3_r/ik_[3]);
if (CheckBox46->Checked==true) kskvi3_r=Ceil(kskv3_r/ik_[3]);
}  //}
//if (fitting==0 || ((fitting==1 || fitting==10) && vse==1))
//{
if (CheckBox42->Checked==true)
{
Edit193->Text=IntToStr(nskvi1);
Edit196->Text=IntToStr(kskvi1);
Edit199->Text=IntToStr(kskvi1-nskvi1+1);
}
if (CheckBox43->Checked==true)
{
Edit194->Text=IntToStr(nskvi2);
Edit197->Text=IntToStr(kskvi2);
Edit200->Text=IntToStr(kskvi2-nskvi2+1);
}
if (CheckBox44->Checked==true)
{
Edit195->Text=IntToStr(nskvi3);
Edit198->Text=IntToStr(kskvi3);
Edit201->Text=IntToStr(kskvi3-nskvi3+1);
}
if (CheckBox22->Checked==true)
{
Edit242->Text=IntToStr(nskvi1_r);
Edit356->Text=IntToStr(kskvi1_r);
//Edit199->Text=IntToStr(kskvi1-nskvi1+1);
}
if (CheckBox45->Checked==true)
{
Edit345->Text=IntToStr(nskvi2_r);
Edit358->Text=IntToStr(kskvi2_r);
//Edit199->Text=IntToStr(kskvi1-nskvi1+1);
}
if (CheckBox46->Checked==true)
{
Edit346->Text=IntToStr(nskvi3_r);
Edit361->Text=IntToStr(kskvi3_r);
//Edit199->Text=IntToStr(kskvi1-nskvi1+1);
}


//}
}

if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==false)
{
Label173->Caption=((444));
Label174->Caption=((444));
}
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==false)
{
Label173->Caption=((888));
Label174->Caption=((888));
}
if (CheckBox42->Checked==false && CheckBox43->Checked==false && CheckBox44->Checked==true)
{
Label173->Caption=((880));
Label174->Caption=((880));
}


if (number_KDV==1)
  {
  Chart6->LeftAxis->Automatic = False;  Chart6->LeftAxis->Minimum = 0.00001;
  Chart3->LeftAxis->Logarithmic=false; Chart6->LeftAxis->Logarithmic=true;
  }
  else
  {
  Chart3->LeftAxis->Automatic = False ;
  Chart6->LeftAxis->Automatic = False ;
  Chart11->LeftAxis->Automatic = False ;
  Chart3->LeftAxis->Minimum = 0.00001 ;
  Chart6->LeftAxis->Minimum = 0.00001 ;
  Chart11->LeftAxis->Minimum = 0.00001 ;
  if (CheckBox75->Checked==false)
    {
    Chart3->LeftAxis->Logarithmic=true;
    Chart6->LeftAxis->Logarithmic=true;
    Chart11->LeftAxis->Logarithmic=true;
    }
  if (CheckBox75->Checked==true)
    {
    Chart3->LeftAxis->Logarithmic=false;
    Chart6->LeftAxis->Logarithmic=false;
    Chart11->LeftAxis->Logarithmic=false;
    }
  }
if (number_KDV==1)   // (vved_exper!=1 && number_KDV==1)
{
Chart3->Visible = true;
Chart3->Height = 177;
Chart3->Top = 215;
Chart6->Height = 235;
Chart6->Top = 392;
Chart11->Visible = false;
}

if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==false)
{
Label173->Caption=((444));
Label174->Caption=((888));
//if (vved_exper!=1)
//{
Chart3->Visible = true;
Chart3->Height = 177;
Chart3->Top = 215;
Chart6->Height = 235;
Chart6->Top = 392;
Chart11->Visible = false;
//}
}
if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==true)
{
Label173->Caption=((444));
Label175->Caption=((880));
//if (vved_exper!=1)
//{
Chart3->Visible = true;
Chart3->Height = 177;
Chart3->Top = 215;
Chart6->Height = 25;
Chart6->Top = 592;
Chart11->Visible = true;
Chart11->Height = 177;
Chart11->Top = 392;
//}
}
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==true)
{
Label174->Caption=((888));
Label175->Caption=((880));
//if (vved_exper!=1)
//{
Chart3->Visible = false;
Chart6->Height = 235;
Chart6->Top = 392;
Chart11->Visible = true;
Chart11->Height = 177;
Chart11->Top = 215;
//}
}
if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==true)
{
Label173->Caption=((444));
Label174->Caption=((888));
Label175->Caption=((880));
//if (vved_exper!=1)
//{
Chart3->Visible = true;
Chart3->Height = 129;
Chart3->Top = 215;
Chart6->Height = 154;
Chart6->Top = 473;
//Chart3->Legend = true;
Chart11->Visible = true;
Chart11->Height = 129;
Chart11->Top = 344;
//}
}

}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void __fastcall TForm1::Button23Click(TObject *Sender)  // Швидкий Start
{
fitting=0;

if (vved_exper==2)   //  дані з файла експерим. КДВ
{
if (CheckBox42->Checked==true)
{
m1_[1]=StrToInt(Edit235->Text);
m10_[1]=StrToInt(Edit238->Text);
ik_[1]=StrToFloat(Edit202->Text);
}
if (CheckBox43->Checked==true)
{
m1_[2]=StrToInt(Edit236->Text);
m10_[2]=StrToInt(Edit239->Text);
ik_[2]=StrToFloat(Edit203->Text);
}
if (CheckBox44->Checked==true)
{
m1_[3]=StrToInt(Edit237->Text);
m10_[3]=StrToInt(Edit240->Text);
ik_[3]=StrToFloat(Edit204->Text);
}
//Memo9->Lines->Add(IntToStr(m1_[1])+'\t'+IntToStr(m10_[1])+'\t'+IntToStr(111));
Memo9->Lines->Add( "vved_exper==2 пройшло");
}

if (CheckBox59->Checked==true||CheckBox60->Checked==true||CheckBox71->Checked==true) OpenAF_Lorenz();  // має бути тут, бо в QuickStart() зразу зчитуються {m1z=m1_[11]; m10z=m10_[11];} і т.д.
QuickStart();
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void TForm1::QuickStart()
{
double K,ik, L_ext[3], w[3],SinTeta1,SinTeta2;

Lambda=1.5405*1e-8;                   // (см)    !!!!!!!!!!!!!!!!!!!!!!!!!!!!

if (RadioButton2->Checked==true)
  {
  SinTeta=Lambda*sqrt(4*4+4*4+4*4)/(2*12.383*1e-8);       // GGG(444)
  Monohr[1]= fabs(sqrt(1-2*SinTeta*SinTeta));   //cos(2Q)=1-2*sin(Q)*sin(Q)
  Edit100->Text=FloatToStr(Monohr[1]);
  }
if (RadioButton56->Checked==true)
  {
  SinTeta1=Lambda*sqrt(1*1+1*1+1*1)/(2*5.4307*1e-8);     // Si(111)
  SinTeta2=Lambda*sqrt(1*1+1*1+1*1)/(2*5.660*1e-8);     // Ge(111)
  Monohr[2]= fabs(sqrt(1-2*SinTeta1*SinTeta1))*fabs(sqrt(1-2*SinTeta2*SinTeta2)); //cos(2Q)=1-2*sin(Q)*sin(Q)
  Edit343->Text=FloatToStr(Monohr[2]);
  }

for (int j=1; j<=3; j++)  //    number_KDV
{
if (j==1) KDV_lich=1;
if (j==2) KDV_lich=2;
if (j==3) KDV_lich=3;

if (KDV_lich==1 && CheckBox42->Checked==true) {h=4; k=4;l=4;}
  else if (KDV_lich==1 && CheckBox42->Checked==false) goto prom_sikl;
if (KDV_lich==2 && CheckBox43->Checked==true) {h=8; k=8;l=8;}
  else if (KDV_lich==2 && CheckBox43->Checked==false) goto prom_sikl;
if (KDV_lich==3 && CheckBox44->Checked==true) {h=8; k=8;l=0;}
  else if (KDV_lich==3 && CheckBox44->Checked==false) goto prom_sikl;

if (CheckBox68->Checked==true) Xi_mon();  // Монокристал
if (CheckBox31->Checked==true) Xi_pl();   // Гетероструктура
if (CheckBox42->Checked==true) Xi_Si();   // Монокристал Si

// Дані про кристал та проміжні розрахунки:
t=1e-8*StrToFloat(Edit12->Text);
K=2*M_PI/Lambda;
Mu0=K*ModChiI0;
Mu0_pl=K*ModChiI0pl;
H=sqrt(h*h+k*k+l*l)/a;        //=1/d
H2Pi=2*M_PI*sqrt(h*h+k*k+l*l)/a;      //=2*pi/d
if (RadioButton7->Checked==true) b=a/2*sqrt(2);
if (RadioButton8->Checked==true) b=a;
if (RadioButton9->Checked==true) b=a*sqrt(2);
if (RadioButton28->Checked==true) b=a/3*sqrt(3);
if (RadioButton32->Checked==true) b=a/2*sqrt(3);
if (RadioButton29->Checked==true) b=a*sqrt(3);
koefLh=StrToFloat(Edit149->Text);
if (CheckBox58->Checked==true)  // врахування анізотропії в орієнтації петель
  {
  if (KDV_lich==1) koefLh=StrToFloat(Edit220->Text);
  if (KDV_lich==2) koefLh=StrToFloat(Edit220->Text);
  if (KDV_lich==3) koefLh=StrToFloat(Edit394->Text);
  if (RadioButton48->Checked==true || RadioButton49->Checked==true)
    {
    if (KDV_lich==1) D_loop=StrToFloat(Edit217->Text);
    if (KDV_lich==2) D_loop=StrToFloat(Edit221->Text);
    if (KDV_lich==3) D_loop=StrToFloat(Edit222->Text);
    }
  }
Kapa[1]=ModChiIH[1]/ModChiRH;
Kapa[2]=ModChiIH[2]/ModChiRH;
Kapa_pl[1]=ModChiIHpl[1]/ModChiRHpl;
Kapa_pl[2]=ModChiIHpl[2]/ModChiRHpl;
p[1]=Kapa[1];                                         // - центр.-сим. кристал
p[2]=Kapa[2];                                         // - центр.-сим. кристал
p_pl[1]=Kapa_pl[1];                                         // - центр.-сим. кристал
p_pl[2]=Kapa_pl[2];                                         // - центр.-сим. кристал
SinTeta=Lambda*sqrt(h*h+k*k+l*l)/(2*a);
CosTeta=sqrt(1-SinTeta*SinTeta);
Sin2Teta=2*CosTeta*SinTeta;
tb=asin(SinTeta);
C[1]=1;                               //- Sigma поляризація
C[2]=fabs(cos(2*tb));                 //- Pi    поляризація
if (RadioButton1->Checked==true)  {nC1=1; nC=1;}
if (RadioButton55->Checked==true) {nC1=2; nC=2;}
if (RadioButton2->Checked==true || RadioButton56->Checked==true) {nC1=1; nC=2;}

psi=acos((h*1+k*1+l*1)/sqrt((h*h+k*k+l*l)*3));  //для зрізу (111)
if (CheckBox18->Checked==false)
{
gamma0=sin(tb-psi);
gammah=sin(tb+psi);
};
if (CheckBox18->Checked==true)
{
gamma0=sin(tb+psi);
gammah=sin(tb-psi);
};
b_as=gamma0/fabs(gammah);
Edit338->Text=FloatToStr(b_as);
Edit340->Text=FloatToStr(psi);
g[1]=-ModChiI0*(sqrt(b_as)+1/sqrt(b_as))/(2*C[1]*ModChiRH);
g[2]=-ModChiI0*(sqrt(b_as)+1/sqrt(b_as))/(2*C[2]*ModChiRH);
g_pl[1]=-ModChiI0pl*(sqrt(b_as)+1/sqrt(b_as))/(2*C[1]*ModChiRHpl);
g_pl[2]=-ModChiI0pl*(sqrt(b_as)+1/sqrt(b_as))/(2*C[2]*ModChiRHpl);
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(/*2.*M_PI**/sqrt(C[1]*C[1])*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(/*2.*M_PI**/sqrt(C[2]*C[2])*ModChiRH);
w[1]=2*fabs(C[1])*ModChiRH/(sqrt(b_as)*Sin2Teta);
w[2]=2*fabs(C[2])*ModChiRH/(sqrt(b_as)*Sin2Teta);

if (fitting==0)
{
    Edit1->Text=FloatToStr(Lambda/1e-8);  // (A)
    Edit11->Text=FloatToStr(Mu0);             // см-1
    Edit266->Text=FloatToStr(1/Mu0*1e4);     // мкм
  if (KDV_lich==1)
  {
    if (RadioButton1->Checked==false) Edit99->Text=FloatToStr(C[2]);
    Edit264->Text=FloatToStr(L_ext[1]*1e4);      // мкм
    Edit5->Text=FloatToStr(L_ext[2]*1e4);      // мкм
    Edit265->Text=FloatToStr(w[1]);
    Edit6->Text=FloatToStr(w[2]);
  }
  if (KDV_lich==2)
  {
    if (RadioButton1->Checked==false) Edit327->Text=FloatToStr(C[2]);
    Edit284->Text=FloatToStr(L_ext[1]*1e4);      // мкм
    Edit7->Text=FloatToStr(L_ext[2]*1e4);      // мкм
    Edit285->Text=FloatToStr(w[1]);
    Edit88->Text=FloatToStr(w[2]);
  }
  if (KDV_lich==3)
  {
    if (RadioButton1->Checked==false) Edit328->Text=FloatToStr(C[2]);
    Edit302->Text=FloatToStr(L_ext[1]*1e4);      // мкм
    Edit386->Text=FloatToStr(L_ext[2]*1e4);      // мкм
    Edit303->Text=FloatToStr(w[1]);
    Edit387->Text=FloatToStr(w[2]);
  }
//Edit11->Text=sprintf("%1.1f",FloatToStr(12.45).c_str()); //визначило кількість знаків
//Edit11->Text=AnsiString(sprintf("%1.1f",FloatToStr(12.45).c_str())); //визначило кількість знаків
//Application->MessageBox(FloatToStr(Mu0).c_str(), MB_OK);
/*char *ss;
  ss  = new char[1000];
sprintf(ss,"%0.3f",Mu0);
Edit11->Text=AnsiString(ss);
delete ss;   */
}

// Дані про КДВ:
if (vved_exper==0) // межі кута і крок задані однакові для всіх КДВ в програмі (без експер. КДВ)
  {
  ik=StrToFloat(Edit19->Text);
  m1=StrToInt(Edit27->Text)-1;
  m10=StrToInt(Edit32->Text);
  if (KDV_lich==1) {m1z=m1_[11]; m10z=m10_[11];}
  if (KDV_lich==2) {m1z=m1_[12]; m10z=m10_[12];}
  if (KDV_lich==3) {m1z=m1_[13]; m10z=m10_[13];}
  // для запису при відсутності експ. КДВ
  if (KDV_lich==1) {ik_[1]=ik; m1_[1]=m1; m10_[1]=m10;}
  if (KDV_lich==2) {ik_[2]=ik; m1_[2]=m1; m10_[2]=m10;}
  if (KDV_lich==3) {ik_[3]=ik; m1_[3]=m1; m10_[3]=m10;}
  }

if (vved_exper==1 || vved_exper==2)
  {
  if (KDV_lich==1) {ik=ik_[1]; m1=m1_[1]; m10=m10_[1]; m1z=m1_[11]; m10z=m10_[11];}
  if (KDV_lich==2) {ik=ik_[2]; m1=m1_[2]; m10=m10_[2]; m1z=m1_[12]; m10z=m10_[12];}
  if (KDV_lich==3) {ik=ik_[3]; m1=m1_[3]; m10=m10_[3]; m1z=m1_[13]; m10z=m10_[13];}
  if (CheckBox65->Checked==true)
    {
    m1=StrToInt(Edit27->Text)-1;
    m10=StrToInt(Edit32->Text);
    }

  if (CheckBox41->Checked==true) // розрах. КДВ тільки в межах обчислення СКВ (при набл. ще не готове. і СКВ обчислює неправильно)
    {
    if (KDV_lich==1) {m1=kskvi1-nskvi1; m10=-nskvi1;}
    if (KDV_lich==2) {m1=kskvi2-nskvi2; m10=-nskvi2;}
    if (KDV_lich==3) {m1=kskvi3-nskvi3; m10=-nskvi3;}
//Memo9->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(222));
//Memo9->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(333));
    }

  CKV=0;
  }


if (CheckBox31->Checked==true)             //плівка
{
// Перерахунок профiлю з вiдн. од. вiдносно плiвки (DD) у вiдн. од. вiдносно пiдкладки (DD)
//int      odpl=Ceil((apl-a)/a*tan(tb)/(ik/3600.*M_PI/180.));
//double   DD0=odpl*(ik/3600.*M_PI/180.)/tan(tb);
double  DD0=(apl-a)/a;
	hpl=1e-4*StrToFloat(Edit166->Text);
if (CheckBox18->Checked==false)  DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)   DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
}
double  TetaMin; //,DeltaTeta_dTeta[MM];
int  ep, ek,  op,  ok,jp,jk;
 ep=-m10;
 ek=m1-m10;
 op=-m10z;      // АФ вся уточнена при розрахунку
 ok=m1z-m10z;   // АФ вся уточнена при розрахунку
 jp=ep-ok;
 jk=ek-op;
m10_teor=-jp;        // =m10+(m1z-m10z)
m1_teor=-jp+jk;      // =m10+(m1z-m10z)+m1-m10-(-m10z)=m1+m1z      =260
/*Memo2->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(op)+'\t'+IntToStr(ok)+'\t'+IntToStr(555)+'\t'+IntToStr(555));
Memo2->Lines->Add(IntToStr(ep)+'\t'+IntToStr(ek)+'\t'+IntToStr(jp)+'\t'+IntToStr(jk));
Memo2->Lines->Add(IntToStr(m1_teor)+'\t'+IntToStr(m10_teor)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Label157->Caption=FloatToStr(111);
Label160->Caption=FloatToStr(MuDSsum[2]);    */

TetaMin=-(m10_teor)*ik;
for (int i=0; i<=m1_teor; i++) DeltaTeta[i]=(TetaMin+i*ik)*M_PI/(3600.*180.);

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
int   koef_dTeta, iff;
double  nkoef_dTeta, kkoef_dTeta;

int m1_teor_ut=StrToInt(Edit382->Text);
double *DeltaTeta_dTeta = new double[m1_teor+m1_teor_ut];
if (KDV_lich==1)
{
koef_dTeta=StrToInt(Edit152->Text);
nkoef_dTeta=StrToFloat(Edit150->Text);
kkoef_dTeta=StrToFloat(Edit151->Text);
}
if (KDV_lich==2)
{
koef_dTeta=StrToInt(Edit385->Text);
nkoef_dTeta=StrToFloat(Edit383->Text);
kkoef_dTeta=StrToFloat(Edit384->Text);
}
if (KDV_lich==3)
{
koef_dTeta=StrToInt(Edit391->Text);
nkoef_dTeta=StrToFloat(Edit389->Text);
kkoef_dTeta=StrToFloat(Edit390->Text);
}

nkoef_dTetai=m10_teor+Ceil(nkoef_dTeta/ik);
kkoef_dTetai=m10_teor+Ceil(kkoef_dTeta/ik);
//Memo2->Lines->Add(IntToStr(555)+'\t'+IntToStr(koef_dTeta)+'\t'+IntToStr(nkoef_dTetai)+'\t'+IntToStr(kkoef_dTetai));
//Memo2->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10));
m1_teor=m1_teor+(kkoef_dTetai-nkoef_dTetai)*(koef_dTeta-1);
m10_teor=m10_teor+(m10_teor-nkoef_dTetai)*(koef_dTeta-1);
//Memo2->Lines->Add(IntToStr(m1_teor)+'\t'+IntToStr(m10_teor)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
for (int i=0; i<=m1_teor; i++) DeltaTeta_dTeta[i]=DeltaTeta[i];
for (int i=0; i<=m1_teor; i++)
{
   if (i<=nkoef_dTetai)
   {
   DeltaTeta[i]=DeltaTeta_dTeta[i];
   }
   if (i>nkoef_dTetai && i<nkoef_dTetai+(kkoef_dTetai-nkoef_dTetai)*koef_dTeta)
   {
   DeltaTeta[i]=(TetaMin+nkoef_dTetai*ik+(i-nkoef_dTetai)*ik/koef_dTeta)*M_PI/(3600*180);
   }
   if (i>=nkoef_dTetai+(kkoef_dTetai-nkoef_dTetai)*koef_dTeta)
   {
   iff= (nkoef_dTetai+(kkoef_dTetai-nkoef_dTetai)*koef_dTeta)-kkoef_dTetai;
   DeltaTeta[i]=DeltaTeta_dTeta[i-iff];
   }
//Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(DeltaTeta[i]));
}
delete DeltaTeta_dTeta;
}
Memo9->Lines->Add( "Швидкий старт пройшло");


RozrachDiduz();
RozrachKoger();
Zgortka();
//    Form2->Memo1->Lines->Add("Швидкий старт nom");
//    Form2->Memo1->Lines->Add(IntToStr(nom));


prom_sikl:
}
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void TForm1::Profil_defects(int &km, double *DD, double &dl)  // Розрахунок профілю за дефектами ас.+спадна гаусіана
{
//double Z_shod [2*KM],D_shod [2*KM];
//double DDPLk_1[KM],DDPLk_2[KM];
  double *Z_shod, *D_shod, *DDd, *DDPLk_1, *DDPLk_2, *DDprof;
  double *R0, *nL,*R0_an, *nL_an,*R0p, *np,*R0ptd, *nptd,*R0d, *nd, *DD_L,*DD_L_an, *DD_SfCl, *DD_SfCltd, *DD_DsCl;
  Z_shod = new double[2*KM];
  D_shod = new double[2*KM];
  DDd    = new double[KM];
  DDPLk_1= new double[KM];
  DDPLk_2= new double[KM];
  DDprof = new double[KM];
  nL= new double[KM];
  R0= new double[KM];
  nL_an= new double[KM];
  R0_an= new double[KM];
  np= new double[KM];
  R0p= new double[KM];
  nptd= new double[KM];
  R0ptd= new double[KM];
  nd= new double[KM];
  R0d= new double[KM];
  DD_L= new double[KM];
  DD_L_an= new double[KM];
  DD_SfCl= new double[KM];
  DD_SfCltd= new double[KM];
  DD_DsCl= new double[KM];
int kk,km_1,km_2,kmd;
double s1,s2,s3,ss,z,L;
double Dmax ,Dmax1, D01, L1, Rp1, D02, L2, Rp2,Dmin,DD0;
double   Gama, Vcl, hp;
 //якщо не наближає чи наближає диф. фон в ПШ чи набл. профіль деф.):
if (fitting==0 || fitting==1) // fitting==1 без vse==1, бо тільки при старті набл. викор. Profil(), а Calc... сюди не заходить!!!
{
Dmax1=StrToFloat(Edit35->Text);
D01=StrToFloat(Edit36->Text);
L1=1e-8*StrToFloat(Edit37->Text);
Rp1=1e-8*StrToFloat(Edit38->Text);
D02=StrToFloat(Edit39->Text);
L2=1e-8*StrToFloat(Edit40->Text);
Rp2=1e-8*StrToFloat(Edit41->Text);
Dmin=StrToFloat(Edit42->Text);
dl=1e-8*StrToFloat(Edit33->Text);
}
if (fitting==10 || CheckBox50->Checked==true)
{	// Профіль та параметри профіля: 1) Dmax1; 2) D01; 3) L1; 4) Rp1; 5) D02; 6) L2; 7) Rp2  8) dl
Dmax1=PARAM[method_lich][1];
D01  =PARAM[method_lich][2];
L1   =PARAM[method_lich][3];
Rp1  =PARAM[method_lich][4];
D02  =PARAM[method_lich][5];
L2     =PARAM[method_lich][6];
Rp2  =PARAM[method_lich][7];
kEW  =PARAM[method_lich][8];
Dmin =PARAM[0][14];
}
if (CheckBox50->Checked==true) dl=1e-8*StrToFloat(Edit97->Text);

// Розрахунок масиву профіля та функції f:
      if (Dmax1!=Dmin) s1=(L1-Rp1)*(L1-Rp1)/LogN(M_E,Dmax1/Dmin); else s1=dl;
      if (Dmax1!=D01)  s2=Rp1*Rp1/LogN(M_E,Dmax1/D01); else s2=10000;
      if (D02!=Dmin)   s3=L2*(L2-2*Rp2)/LogN(M_E,D02/Dmin);  else s3=dl;
      ss=s2;
      kk=0;

      DDPLk_1[kk]=Dmax1;
      while (DDPLk_1[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      if (z>=Rp1) ss=s1;
      DDPLk_1[kk]=Dmax1*exp(-(z-Rp1)*(z-Rp1)/ss);
}
      km_1=kk-1;
for (int k=1; k<=km_1;k++)  DDPL1[k]=DDPLk_1[km_1-k+1];

      kk=0;
      DDPLk_2[kk]=D02;
      while (DDPLk_2[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      DDPLk_2[kk]=D02*exp(Rp2*Rp2/s3)*exp(-(z-Rp2)*(z-Rp2)/s3);
}
      km_2=kk-1;
for (int k=1; k<=km_2;k++)  DDPL2[k]=DDPLk_2[km_2-k+1];

if (km_2<km_1)
{
for (int k=km_1; k>km_1-km_2;k--) DDPL2[k]=DDPL2[k-(km_1-km_2)];
for (int k=km_1-km_2; k>=1;k--) DDPL2[k]=0;
for (int k=1; k<=km_1;k++) DDprof[k]=DDPL1[k]+DDPL2[k];
km=km_1;
}
if (km_2>=km_1)
{
for (int k=km_2; k>km_2-km_1;k--) DDPL1[k]=DDPL1[k-(km_2-km_1)];
for (int k=km_2-km_1; k>=1;k--) DDPL1[k]=0;
for (int k=1; k<=km_2;k++) DDprof[k]=DDPL1[k]+DDPL2[k];
km=km_2;
}


   if (CheckBox52->Checked==true)   // подвійна імплантація
   {
   Profil_double(kmd, DDd);
   if (kmd<km)
   {
   for (int k=km; k>km-kmd;k--) DDd[k]=DDd[k-(km-kmd)];
   for (int k=km-kmd; k>=1;k--) DDd[k]=0;
   for (int k=1; k<=km;k++) DDprof[k]=DDprof[k]+DDd[k];
   }
   if (kmd>=km)
   {
   for (int k=kmd; k>kmd-km;k--) {DDprof[k]=DDprof[k-(kmd-km)]; DDPL1[k]=DDPL1[k-(kmd-km)]; DDPL2[k]=DDPL2[k-(kmd-km)];}
   for (int k=kmd-km; k>=1;k--) {DDprof[k]=0; DDPL1[k]=0; DDPL2[k]=0;}
   for (int k=1; k<=kmd;k++) DDprof[k]=DDprof[k]+DDd[k];
   km=kmd;
   }
   }

L=km*dl;
//dl0=t-L;
for (int k=1; k<=km;k++)    Dl[k]=dl;
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DDprof[k])) Dmax=fabs(DDprof[k]);
for (int k=1; k<=km;k++) f[k]=fabs(DDprof[k]/Dmax);


for (int k=1; k<=km;k++)
{
DD_L[k]=0;
DD_SfCl[k]=0;
DD_SfCltd[k]=0;
DD_DsCl[k]=0;
}

//Розрахунок для дислокаційних петель (Defekts_SL[1], Defekts_SL[2]  -- R0_max, nL_max)
if (CheckBox1->Checked==true)
{
if (CheckBox6->Checked==true) for (int k=1; k<=km;k++) nL[k]=Defekts_SL[2]*f[k];
else for (int k=1; k<=km;k++) nL[k]=Defekts_SL[2];
if (CheckBox7->Checked==true) for (int k=1; k<=km;k++) R0[k]=Defekts_SL[1]*f[k];
else for (int k=1; k<=km;k++) R0[k]=Defekts_SL[1];
for (int k=1; k<=km;k++) DD_L[k]=M_PI*b*R0[k]*R0[k]*nL[k];
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD_L[k])) Dmax=fabs(DD_L[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD_L[k]/Dmax);
Edit243->Text=FloatToStr(Dmax);
}
//Розрахунок для дислокаційних петель (Defekts_SL[13], Defekts_SL[14]-- R0_max_an, nL_max_an)
if (CheckBox58->Checked==true)
{
if (CheckBox39->Checked==true) for (int k=1; k<=km;k++) nL[k]=Defekts_SL[14]*f[k];
else for (int k=1; k<=km;k++) nL[k]=Defekts_SL[14];
if (CheckBox61->Checked==true) for (int k=1; k<=km;k++) R0[k]=Defekts_SL[13]*f[k];
else for (int k=1; k<=km;k++) R0[k]=Defekts_SL[13];
for (int k=1; k<=km;k++) DD_L_an[k]=M_PI*b*R0[k]*R0[k]*nL[k];
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD_L_an[k])) Dmax=fabs(DD_L_an[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD_L[k]/Dmax);
Edit288->Text=FloatToStr(Dmax);
}

//розрахунок для сферичних кластерів
if (CheckBox2->Checked==true)
{
if (CheckBox8->Checked==true) for (int k=1; k<=km;k++) np[k]=Defekts_SL[5]*f[k];
else for (int k=1; k<=km;k++)  np[k]=Defekts_SL[5];
if (CheckBox9->Checked==true) for (int k=1; k<=km;k++) R0p[k]=Defekts_SL[4]*f[k];
else for (int k=1; k<=km;k++)  R0p[k]=Defekts_SL[4];
Gama=(1+Nu)/(3*(1-Nu));
for (int k=1; k<=km;k++)
{
Vcl=4/3.*M_PI*R0p[k]*R0p[k]*R0p[k];
DD_SfCl[k]=Gama*Defekts_SL[6]*Vcl*np[k];
}
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD_SfCl[k])) Dmax=fabs(DD_SfCl[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD_SfCl[k]/Dmax);
Edit244->Text=FloatToStr(Dmax);
}

//розрахунок для сферичних кластерів (точкові дефекти)
if (CheckBox26->Checked==true)
{
if (CheckBox69->Checked==true) for (int k=1; k<=km;k++) nptd[k]=Defekts_SL[11]*f[k];
else for (int k=1; k<=km;k++)  nptd[k]=Defekts_SL[11];
if (CheckBox70->Checked==true) for (int k=1; k<=km;k++) R0ptd[k]=Defekts_SL[10]*f[k];
else for (int k=1; k<=km;k++)  R0ptd[k]=Defekts_SL[10];
Gama=(1+Nu)/(3*(1-Nu));
for (int k=1; k<=km;k++)
{
Vcl=4/3.*M_PI*R0ptd[k]*R0ptd[k]*R0ptd[k];
DD_SfCltd[k]=Gama*Defekts_SL[12]*Vcl*nptd[k];
}
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD_SfCltd[k])) Dmax=fabs(DD_SfCltd[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD_SfCltd[k]/Dmax);
Edit253->Text=FloatToStr(Dmax);
}
//розрахунок для дископодібних кластерів
if (CheckBox4->Checked==true)
{
if (CheckBox10->Checked==true) for (int k=1; k<=km;k++) nd[k]=Defekts_SL[8]*f[k];
else for (int k=1; k<=km;k++)  nd[k]=Defekts_SL[8];
if (CheckBox11->Checked==true) for (int k=1; k<=km;k++) R0d[k]=Defekts_SL[7]*f[k];
else for (int k=1; k<=km;k++)  R0d[k]=Defekts_SL[7];
Gama=(1+Nu)/(3*(1-Nu));
for (int k=1; k<=km;k++)
{
hp=3.96*R0d[k]*exp(0.5966*log((0.89e-8/R0d[k])));                     //дані для кремнію
Vcl=M_PI*R0d[k]*R0d[k]*hp;
DD_DsCl[k]=Gama*Defekts_SL[9]*Vcl*nd[k];
}
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD_DsCl[k])) Dmax=fabs(DD_DsCl[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD_DsCl[k]/Dmax);
Edit258->Text=FloatToStr(Dmax);
}

for (int k=1; k<=km;k++) DD[k]=DD_L[k]+DD_L_an[k]+DD_SfCl[k]+DD_SfCltd[k]+DD_DsCl[k];
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
//for (int k=1; k<=km;k++) f[k]=fabs(DD[k]/Dmax);
Edit245->Text=FloatToStr(Dmax);


if (fitting==0)
{
Edit43->Text=FloatToStr(Dmax);
Edit44->Text=FloatToStr(L/1e-8);
Edit45->Text=IntToStr(km);
Edit223->Text=IntToStr(km_1);
Edit212->Text=IntToStr(km_2);
Edit224->Text=IntToStr(kmd);

for (int k=1; k<=km;k++)
{
z=dl*km-dl*k+dl/2;
Series5->AddXY(z/1e-8,DD[k],"",clFuchsia);
Series28->AddXY(z/1e-8,DDprof[k],"",clRed);
Series30->AddXY(z/1e-8,DDPL1[k],"",clRed);
Series31->AddXY(z/1e-8,DDPL2[k],"",clRed);
if (CheckBox52->Checked==true) Series27->AddXY(z/1e-8,DDd[k],"",clYellow);
}

if (CheckBox31->Checked==false)
  {
  if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
  if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
  }
if (CheckBox31->Checked==true)
{
DD0=(apl-a)/a;  //Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
if (CheckBox18->Checked==false) DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
//for (int k=1; k<=km;k++) DD[k]=(DD[k]+1.)*(DD0+1.)-1. ;
if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
}

//Мiтки одиниць dD/D на КДВ:
if (KDV_lich==1) for (int k=1; k<=km;k++)Series32->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==2) for (int k=1; k<=km;k++)Series52->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==3) for (int k=1; k<=km;k++)Series50->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
//  Z_shod = new double[2*km+2];
//  D_shod = new double[2*km+2];
DD[0]=0;
L=dl*km;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=L-dl*(k-1);
Z_shod[2*k]=L-dl*(k-1);
D_shod[2*k-1]=DD[k];
D_shod[2*k]=DD[k-1];
}
Z_shod[2*km+1]=0.;
D_shod[2*km+1]=DD[km];

for (int k=1; k<=2*km+1;k++)
{ Series33->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlack);
}
}
if (fitting==0 || fitting==1 || (fitting==10 && vse==1))
{                     //Для збереження стартових даних
kp=13;
DDparam0[1]=Dmax1;
DDparam0[2]=D01;
DDparam0[3]=L1/1e-8;
DDparam0[4]=Rp1/1e-8;
DDparam0[5]=D02;
DDparam0[6]=L2/1e-8;
DDparam0[7]=Rp2/1e-8;
DDparam0[8]=Dmin;
DDparam0[9]=Emin;  //!!!!!! При першому циклі записує Emin=0, а далі правильне (бо Profil() виконується швидше за RozrachDiduz_SL()
DDparam0[10]=km;
DDparam0[11]=dl/1e-8;
DDparam0[12]=Dmax;
DDparam0[13]=L/1e-8;
 for (int k=1; k<=km;k++)
 {
 DDstart[0][k]=DD[k];
 DDstart[1][k]=DDPL1[k];
 DDstart[2][k]=DDPL2[k];
}
}
if (fitting==10 && vse==2)
{                        //Для збереження результату
kp=13;
DDparam[1]=Dmax1;
DDparam[2]=D01;
DDparam[3]=L1/1e-8;
DDparam[4]=Rp1/1e-8;
DDparam[5]=D02;
DDparam[6]=L2/1e-8;
DDparam[7]=Rp2/1e-8;
DDparam[8]=Dmin;
DDparam[9]=Emin;
DDparam[10]=km;
DDparam[11]=dl/1e-8;
DDparam[12]=Dmax;
DDparam[13]=L/1e-8;
}
delete Z_shod, D_shod, DDd, DDPLk_1,DDPLk_2,DDprof,nL,R0;
delete R0, nL,R0_an, nL_an,R0p, np,R0ptd, nptd,R0d, nd, DD_L,DD_L_an, DD_SfCl, DD_SfCltd,DD_DsCl;
//for (int k=1; k<=km;k++) Memo9-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(km)+'\t'+FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k]));

Memo9->Lines->Add( "Profil пройшло");
}

//---------------------------------------------------------------------------
void TForm1::Profil(int &km, double *DD, double &dl)  //Розрахунок профілю ас. + спадна гаусіана
{
//double Z_shod [2*KM],D_shod [2*KM];
//double DDPLk_1[KM],DDPLk_2[KM];
  double *Z_shod, *D_shod, *DDd, *DDPLk_1, *DDPLk_2;
  Z_shod = new double[2*KM];
  D_shod = new double[2*KM];
  DDd    = new double[KM];
  DDPLk_1= new double[KM];
  DDPLk_2= new double[KM];
int kk,km_1,km_2,kmd;
double s1,s2,s3,ss,z,L;
double Dmax ,Dmax1, D01, L1, Rp1, D02, L2, Rp2,Dmin,DD0;
 //якщо не наближає чи наближає диф. фон в ПШ чи набл. профіль деф.):
if (fitting==0 || fitting==1) // fitting==1 без vse==1, бо тільки при старті набл. викор. Profil(), а Calc... сюди не заходить!!!
{
Dmax1=StrToFloat(Edit35->Text);
D01=StrToFloat(Edit36->Text);
L1=1e-8*StrToFloat(Edit37->Text);
Rp1=1e-8*StrToFloat(Edit38->Text);
D02=StrToFloat(Edit39->Text);
L2=1e-8*StrToFloat(Edit40->Text);
Rp2=1e-8*StrToFloat(Edit41->Text);
Dmin=StrToFloat(Edit42->Text);
dl=1e-8*StrToFloat(Edit33->Text);
}
if (fitting==10 || CheckBox50->Checked==true)
{	// Профіль та параметри профіля: 1) Dmax1; 2) D01; 3) L1; 4) Rp1; 5) D02; 6) L2; 7) Rp2  8) dl
Dmax1=PARAM[method_lich][1];
D01  =PARAM[method_lich][2];
L1   =PARAM[method_lich][3];
Rp1  =PARAM[method_lich][4];
D02  =PARAM[method_lich][5];
L2   =PARAM[method_lich][6];
Rp2  =PARAM[method_lich][7];
kEW  =PARAM[method_lich][8];
Dmin =PARAM[0][14];
}
if (CheckBox50->Checked==true) dl=1e-8*StrToFloat(Edit97->Text);

// Розрахунок масиву профіля та функції f:
      if (Dmax1!=Dmin) s1=(L1-Rp1)*(L1-Rp1)/LogN(M_E,Dmax1/Dmin); else s1=dl;
      if (Dmax1!=D01)  s2=Rp1*Rp1/LogN(M_E,Dmax1/D01); else s2=10000;
      if (D02!=Dmin)   s3=L2*(L2-2*Rp2)/LogN(M_E,D02/Dmin);  else s3=dl;
      ss=s2;
      kk=0;

      DDPLk_1[kk]=Dmax1;
      while (DDPLk_1[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      if (z>=Rp1) ss=s1;
      DDPLk_1[kk]=Dmax1*exp(-(z-Rp1)*(z-Rp1)/ss);
}
      km_1=kk-1;
for (int k=1; k<=km_1;k++)  DDPL1[k]=DDPLk_1[km_1-k+1];

      kk=0;
      DDPLk_2[kk]=D02;
      while (DDPLk_2[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      DDPLk_2[kk]=D02*exp(Rp2*Rp2/s3)*exp(-(z-Rp2)*(z-Rp2)/s3);
}
      km_2=kk-1;
for (int k=1; k<=km_2;k++)  DDPL2[k]=DDPLk_2[km_2-k+1];

if (km_2<km_1)
{
for (int k=km_1; k>km_1-km_2;k--) DDPL2[k]=DDPL2[k-(km_1-km_2)];
for (int k=km_1-km_2; k>=1;k--) DDPL2[k]=0;
for (int k=1; k<=km_1;k++) DD[k]=DDPL1[k]+DDPL2[k];
km=km_1;
}
if (km_2>=km_1)
{
for (int k=km_2; k>km_2-km_1;k--) DDPL1[k]=DDPL1[k-(km_2-km_1)];
for (int k=km_2-km_1; k>=1;k--) DDPL1[k]=0;
for (int k=1; k<=km_2;k++) DD[k]=DDPL1[k]+DDPL2[k];
km=km_2;
}


   if (CheckBox52->Checked==true)   // подвійна імплантація
   {
   Profil_double(kmd, DDd);
   if (kmd<km)
   {
   for (int k=km; k>km-kmd;k--) DDd[k]=DDd[k-(km-kmd)];
   for (int k=km-kmd; k>=1;k--) DDd[k]=0;
   for (int k=1; k<=km;k++) DD[k]=DD[k]+DDd[k];
   }
   if (kmd>=km)
   {
   for (int k=kmd; k>kmd-km;k--) {DD[k]=DD[k-(kmd-km)]; DDPL1[k]=DDPL1[k-(kmd-km)]; DDPL2[k]=DDPL2[k-(kmd-km)];}
   for (int k=kmd-km; k>=1;k--) {DD[k]=0; DDPL1[k]=0; DDPL2[k]=0;}
   for (int k=1; k<=kmd;k++) DD[k]=DD[k]+DDd[k];
   km=kmd;
   }
   }

L=km*dl;
//dl0=t-L;
for (int k=1; k<=km;k++)    Dl[k]=dl;
Dmax=0;
for (int k=1; k<=km;k++) if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
for (int k=1; k<=km;k++) f[k]=fabs(DD[k]/Dmax);


if (fitting==0)
{
Edit43->Text=FloatToStr(Dmax);
Edit44->Text=FloatToStr(L/1e-8);
Edit45->Text=IntToStr(km);
Edit223->Text=IntToStr(km_1);
Edit212->Text=IntToStr(km_2);
Edit224->Text=IntToStr(kmd);

for (int k=1; k<=km;k++)
{
z=dl*km-dl*k+dl/2;
Series5->AddXY(z/1e-8,DD[k],"",clFuchsia		);
Series30->AddXY(z/1e-8,DDPL1[k],"",clRed);
Series31->AddXY(z/1e-8,DDPL2[k],"",clRed);
if (CheckBox52->Checked==true) Series27->AddXY(z/1e-8,DDd[k],"",clYellow);
}

if (CheckBox31->Checked==false)
  {
  if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
  if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
  }
if (CheckBox31->Checked==true)
{
DD0=(apl-a)/a;  //Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
if (CheckBox18->Checked==false) DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
//for (int k=1; k<=km;k++) DD[k]=(DD[k]+1.)*(DD0+1.)-1. ;
if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
}

//Мiтки одиниць dD/D на КДВ:
if (KDV_lich==1) for (int k=1; k<=km;k++)Series32->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==2) for (int k=1; k<=km;k++)Series52->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==3) for (int k=1; k<=km;k++)Series50->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
//  Z_shod = new double[2*km+2];
//  D_shod = new double[2*km+2];
DD[0]=0;
L=dl*km;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=L-dl*(k-1);
Z_shod[2*k]=L-dl*(k-1);
D_shod[2*k-1]=DD[k];
D_shod[2*k]=DD[k-1];
}
Z_shod[2*km+1]=0;
D_shod[2*km+1]=DD[km];

for (int k=1; k<=2*km+1;k++)
{ Series33->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlack);
}
}
if (fitting==0 || fitting==1 || (fitting==10 && vse==1))
{                     //Для збереження стартових даних
kp=13;
DDparam0[1]=Dmax1;
DDparam0[2]=D01;
DDparam0[3]=L1/1e-8;
DDparam0[4]=Rp1/1e-8;
DDparam0[5]=D02;
DDparam0[6]=L2/1e-8;
DDparam0[7]=Rp2/1e-8;
DDparam0[8]=Dmin;
DDparam0[9]=Emin;  //!!!!!! При першому циклі записує Emin=0, а далі правильне (бо Profil() виконується швидше за RozrachDiduz_SL()
DDparam0[10]=km;
DDparam0[11]=dl/1e-8;
DDparam0[12]=Dmax;
DDparam0[13]=L/1e-8;
 for (int k=1; k<=km;k++)
 {
 DDstart[0][k]=DD[k];
 DDstart[1][k]=DDPL1[k];
 DDstart[2][k]=DDPL2[k];
}
}
if (fitting==10 && vse==2)
{                        //Для збереження результату
kp=13;
DDparam[1]=Dmax1;
DDparam[2]=D01;
DDparam[3]=L1/1e-8;
DDparam[4]=Rp1/1e-8;
DDparam[5]=D02;
DDparam[6]=L2/1e-8;
DDparam[7]=Rp2/1e-8;
DDparam[8]=Dmin;
DDparam[9]=Emin;
DDparam[10]=km;
DDparam[11]=dl/1e-8;
DDparam[12]=Dmax;
DDparam[13]=L/1e-8;
}
delete Z_shod, D_shod, DDd, DDPLk_1,DDPLk_2;
//for (int k=1; k<=km;k++) Memo9-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(km)+'\t'+FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k]));

Memo9->Lines->Add( "Profil пройшло");
}

//---------------------------------------------------------------------------
void TForm1::Profil_double(int &kmd, double *DDd)  // Розрахунок профілю ас. + спадна гаусіана
{                                                  // Подвійна імплантація
double *DDPLk_1,*DDPLk_2,*DDPL1,*DDPL2;
  DDPLk_1 = new double[KM];
  DDPLk_2 = new double[KM];
  DDPL1 = new double[KM];
  DDPL2 = new double[KM];
int kk,km_1,km_2;
double s1,s2,s3,ss,z,L;
double Dmax1, D01, L1, Rp1, D02, L2, Rp2, Dmin;

Dmax1=StrToFloat(Edit205->Text);
D01=StrToFloat(Edit206->Text);
L1=1e-8*StrToFloat(Edit207->Text);
Rp1=1e-8*StrToFloat(Edit208->Text);
D02=StrToFloat(Edit209->Text);
L2=1e-8*StrToFloat(Edit210->Text);
Rp2=1e-8*StrToFloat(Edit211->Text);
Dmin=StrToFloat(Edit42->Text);
dl=1e-8*StrToFloat(Edit33->Text);

// Розрахунок масиву профіля та функції f:
      if (Dmax1!=Dmin) s1=(L1-Rp1)*(L1-Rp1)/LogN(M_E,Dmax1/Dmin); else s1=dl;
      if (Dmax1!=D01)  s2=Rp1*Rp1/LogN(M_E,Dmax1/D01); else s2=10000;
      if (D02!=Dmin)   s3=L2*(L2-2*Rp2)/LogN(M_E,D02/Dmin);  else s3=dl;
      ss=s2;
      kk=0;

      DDPLk_1[kk]=Dmax1;
      while (DDPLk_1[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      if (z>=Rp1) ss=s1;
      DDPLk_1[kk]=Dmax1*exp(-(z-Rp1)*(z-Rp1)/ss);
Memo1->Lines->Add(IntToStr(kk)+'\t'+FloatToStr(z)+'\t'+FloatToStr(DDPLk_1[kk])+'\t'+FloatToStr(111)+'\t'+IntToStr(11111)+'\t'+IntToStr(22222));
}
      km_1=kk-1;
for (int k=1; k<=km_1;k++)  DDPL1[k]=DDPLk_1[km_1-k+1];

      kk=0;
      DDPLk_2[kk]=D02;
      while (DDPLk_2[kk]>Dmin)
{       kk=kk+1;
      z=dl*kk-dl/2;
      DDPLk_2[kk]=D02*exp(Rp2*Rp2/s3)*exp(-(z-Rp2)*(z-Rp2)/s3);
Memo1->Lines->Add(IntToStr(kk)+'\t'+FloatToStr(z)+'\t'+FloatToStr(DDPLk_2[kk])+'\t'+FloatToStr(333)+'\t'+IntToStr(11111)+'\t'+IntToStr(22222));
}
      km_2=kk-1;
for (int k=1; k<=km_2;k++)  DDPL2[k]=DDPLk_2[km_2-k+1];

if (km_2<km_1)
{
for (int k=km_1; k>km_1-km_2;k--) DDPL2[k]=DDPL2[k-(km_1-km_2)];
for (int k=km_1-km_2; k>=1;k--) DDPL2[k]=0;
for (int k=1; k<=km_1;k++) DDd[k]=DDPL1[k]+DDPL2[k];
kmd=km_1;
}
if (km_2>=km_1)
{
for (int k=km_2; k>km_2-km_1;k--) DDPL1[k]=DDPL1[k-(km_2-km_1)];
for (int k=km_2-km_1; k>=1;k--) DDPL1[k]=0;
for (int k=1; k<=km_2;k++) DDd[k]=DDPL1[k]+DDPL2[k];
kmd=km_2;
}

delete DDPL1, DDPL2, DDPLk_1, DDPLk_2;
}

//---------------------------------------------------------------------------
void TForm1::Profil_shod()         ///Розрахунок профілю  сходинками
{
//double Z_shod [2*KM],D_shod [2*KM];
  double *Z_shod, *D_shod;
  Z_shod = new double[2*KM];
  D_shod = new double[2*KM];
double ss,z,L,L_tmp,L_shod;
double Dmax, DD0 ;

if (CheckBox72->Checked==true) // "Початковий" профіль - 1 сходинка
{
Edit90->Text=IntToStr(1);
Memo5->Clear();
Memo5->Lines->Add(FloatToStr(0.002)+'\t'+FloatToStr(5000)) ;
}
if (CheckBox38->Checked==true) // "Початковий" профіль для сходинок з гаусіани
{
Profil(km,DD,dl);
for (int k=1; k<=km;k++) Memo6-> Lines->Add(IntToStr(k));
for (int k=1; k<=km;k++) Memo5-> Lines->Add(FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k]/1e-8));
/*
//char ss[20]="12345";
MessageBox(0,"Повідомл.","Шапка", MB_OK + MB_ICONEXCLAMATION);
char ss[20];
for (int k=1; k<=km;k++)
{
        sprintf(ss,"%3.6lf\t%.0lf", DD[k],Dl[k]/1e-8);
        Memo5-> Lines->Add( ss );
}           */
Edit90->Text=IntToStr(km);
}
km=StrToInt(Edit90->Text);

ReadMemo2stovp(Memo5,km,DD,Dl);    //  Зчитуємо профіль з Memo5
for (int k=1; k<=km;k++) Dl[k]=Dl[k]*1e-8;

Dmax=0;
L=0;
for (int k=1; k<=km;k++)
{
if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
    L=L+Dl[k];
}
//      dl0=t-L;
for (int k=1; k<=km;k++)   f[k]=fabs(DD[k]/Dmax);

Edit91->Text=FloatToStr(Dmax);
Edit92->Text=FloatToStr(L/1e-8);

L_tmp=0;
for (int k=1; k<=km;k++)
{
L_tmp=L_tmp+Dl[k];
z=L-(L_tmp-Dl[k]/2);
Series3->AddXY(z/1e-8,DD[k],"",clRed);
}

if (CheckBox31->Checked==false)
{
if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=DD[k]/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
}
if (CheckBox31->Checked==true)
{
DD0=(apl-a)/a;  //Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
if (CheckBox18->Checked==false) DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  DeltaTetaDDpl=DD0/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
//for (int k=1; k<=km;k++) DD[k]=(DD[k]+1.)*(DD0+1.)-1. ;
if (CheckBox18->Checked==false) for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
if (CheckBox18->Checked==true)  for (int k=1; k<=km;k++) DeltaTetaDD[k]=((DD[k]+1)*(DD0+1)-1)/b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi));
}
//Мiтки одиниць dD/D на КДВ:
if (KDV_lich==1) for (int k=1; k<=km;k++)Series32->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==2) for (int k=1; k<=km;k++)Series52->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);
if (KDV_lich==3) for (int k=1; k<=km;k++)Series50->AddXY((DeltaTetaDD[k])*180/M_PI*3600, 0.000005,"",clBlack);

//  Z_shod = new double[2*km+2];
//  D_shod = new double[2*km+2];
DD[0]=0;
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;

for (int k=1; k<=2*km+1;k++)
{ Series34->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlack);
}
delete Z_shod, D_shod;
}

//---------------------------------------------------------------------------
void TForm1::Xi_SL()
{
if (RadioButton59->Checked==true)   // Хі як в мон.
  {
  if (CheckBox31->Checked==false || (CheckBox31->Checked==true && RadioButton45->Checked==true)) // GGG
  for (int k=1; k<=km;k++)
    {
    //ChiRO[k]=ChiRO;
    ChiI0_a[k]=ChiI0;
    ModChiI0_a[k]=ModChiI0;
    ReChiRH_a[k] =ReChiRH;
    ModChiRH_a[k]=ModChiRH;
    ReChiIH_a[1][k] =ReChiIH[1];
    ModChiIH_a[1][k]=ModChiIH[1];
    ReChiIH_a[2][k] =ReChiIH[2];
    ModChiIH_a[2][k]=ModChiIH[2];
    }
  if (CheckBox31->Checked==true && RadioButton45->Checked==false) // YIG
  for (int k=1; k<=km;k++)
    {
    //ChiRO[k]=ChiRO;
    ChiI0_a[k]=ChiI0pl;
    ModChiI0_a[k]=ModChiI0pl;
    ReChiRH_a[k] =ReChiRHpl;
    ModChiRH_a[k]=ModChiRHpl;
    ReChiIH_a[1][k] =ReChiIHpl[1];
    ModChiIH_a[1][k]=ModChiIHpl[1];
    ReChiIH_a[2][k] =ReChiIHpl[2];
    ModChiIH_a[2][k]=ModChiIHpl[2];
    }
  }


if (RadioButton58->Checked==true)   // Хі проп. da
{
  if (CheckBox31->Checked==false || (CheckBox31->Checked==true && RadioButton45->Checked==true)) // GGG
  {
  for (int k=1; k<=km;k++)
    {
    //ChiRO[k]= -0.000353602+ 0.0000422263*Dl[k]*k -0.00000134461*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ChiI0_a[k]=ChiI0/(-0.0000344464+0.00000411312*12.383-0.000000130962*12.383*12.383)*
      (-0.0000344464+0.00000411312*a*1e8*(1.+DD[k])-0.000000130962*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiI0_a[k]=fabs(ChiI0_a[k]);
    }
  if (KDV_lich==1) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRH/(0.0000914273-0.0000101963*12.383+0.000000309731*12.383*12.383)*
      (0.0000914273-0.0000101963*a*1e8*(1.+DD[k])+0.000000309731*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIH[1]/(0.0000309409-0.00000368604*12.383+0.000000117155*12.383*12.383)*
      (0.0000309409-0.00000368604*a*1e8*(1.+DD[k])+0.000000117155*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIH[2]/(0.00000736383-0.000000559792*12.383+0.0000000105512*12.383*12.383)*
      (0.00000736383-0.000000559792*a*1e8*(1.+DD[k])+0.0000000105512*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    //Memo8->Lines->Add(FloatToStr(k)+'\t'+FloatToStr(a*1e8*(1.+DD[k]))+'\t'+FloatToStr(ModChiRH_a[k])+'\t'+FloatToStr(ModChiI0_a[k]) +'\t'+FloatToStr(ModChiIH_a[1][k]) +'\t'+FloatToStr(ModChiIH_a[2][k]) );
    }
  if (KDV_lich==2) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRH/(-0.0000547342+0.00000535002*12.383-0.000000147906*12.383*12.383)*
      (-0.0000547342+0.00000535002*a*1e8*(1.+DD[k])-0.000000147906*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIH[1]/(-0.0000313766+0.00000371064*12.383-0.000000117325*12.383*12.383)*
      (-0.0000313766+0.00000371064*a*1e8*(1.+DD[k])-0.000000117325*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIH[2]/(-0.0000652364+0.00000906133*12.383-0.000000317144*12.383*12.383)*
      (-0.0000652364+0.00000906133*a*1e8*(1.+DD[k])-0.000000317144*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    //Memo8->Lines->Add(FloatToStr(k)+'\t'+FloatToStr(a*1e8*(1.+DD[k]))+'\t'+FloatToStr(ModChiRH_a[k])+'\t'+FloatToStr(ModChiI0_a[k]) +'\t'+FloatToStr(ModChiIH_a[1][k]) +'\t'+FloatToStr(ModChiIH_a[2][k]) );
    }
  if (KDV_lich==3) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRH/(-0.0000773522+0.00000800574*12.383-0.000000230482*12.383*12.383)*
      (-0.0000773522+0.00000800574*a*1e8*(1.+DD[k])-0.000000230482*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIH[1]/(-0.0000322337+0.00000382429*12.383-0.000000121199*12.383*12.383)*
      (-0.0000322337+0.00000382429*a*1e8*(1.+DD[k])-0.000000121199*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIH[2]/(0.0000336289-0.00000489358*12.383+0.000000175659*12.383*12.383)*
      (0.0000336289-0.00000489358*a*1e8*(1.+DD[k])+0.000000175659*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    //Memo8->Lines->Add(FloatToStr(k)+'\t'+FloatToStr(a*1e8*(1.+DD[k]))+'\t'+FloatToStr(ModChiRH_a[k])+'\t'+FloatToStr(ModChiI0_a[k]) +'\t'+FloatToStr(ModChiIH_a[1][k]) +'\t'+FloatToStr(ModChiIH_a[2][k]) );
    }
  }

  if (CheckBox31->Checked==true && RadioButton45->Checked==false)  // YIG
  {
  for (int k=1; k<=km;k++)
    {
    //ChiRO[k]= -0.000290442+ 0.0000347021*a*1e8*(1.+DD[k])-0.00000110559*a*1e8*(1.+DD[k])*a*1e8*(1.+DD[k]));
    ChiI0_a[k]=ChiI0pl/(-0.0000192592+ 0.00000230102*12.376-0.0000000733075*12.376*12.376)*
      (-0.0000192592+ 0.00000230102*apl*1e8*(1.+DD[k])-0.0000000733075*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiI0_a[k]=fabs(ChiI0_a[k]);
    }
  if (KDV_lich==1) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRHpl/(0.0000656422-0.00000737446*12.376+0.000000225359*12.376*12.376)*
      (0.0000656422-0.00000737446*apl*1e8*(1.+DD[k])+0.000000225359*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIHpl[1]/(0.00000757104-0.000000899214*12.376+ 0.0000000285259*12.376*12.376)*
      (0.00000757104-0.000000899214*apl*1e8*(1.+DD[k])+ 0.0000000285259*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIHpl[2]/(0.00000176856-0.000000130623*12.376+0.00000000233606*12.376*12.376)*
      (0.00000176856-0.000000130623*apl*1e8*(1.+DD[k])+0.00000000233606*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    }
  if (KDV_lich==2) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRHpl/(-0.0000446087+0.00000457534*12.376-0.000000131937*12.376*12.376)*
      (-0.0000446087+0.00000457534*apl*1e8*(1.+DD[k])-0.000000131937*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIHpl[1]/(-0.0000137086+0.000001574*12.376-0.0000000487221*12.376*12.376)*
      (-0.0000137086+0.000001574*apl*1e8*(1.+DD[k])-0.0000000487221*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIHpl[2]/(-0.0000306025+0.00000422812*12.376-0.000000147221*12.376*12.376)*
      (-0.0000306025+0.00000422812*apl*1e8*(1.+DD[k])-0.000000147221*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    }
  if (KDV_lich==3) for (int k=1; k<=km;k++)
    {
    ReChiRH_a[k]=ReChiRHpl/(-0.0000740436+0.00000761197*12.376-0.000000218205*12.376*12.376)*
      (-0.0000740436+0.00000761197*apl*1e8*(1.+DD[k])-0.000000218205*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiRH_a[k]=fabs(ReChiRH_a[k]);
    ReChiIH_a[1][k]=ReChiIHpl[1]/(-0.0000313058+0.00000370373*12.376-0.000000117168*12.376*12.376)*
      (-0.0000313058+0.00000370373*apl*1e8*(1.+DD[k])-0.000000117168*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[1][k]=fabs(ReChiIH_a[1][k]);
    ReChiIH_a[2][k]=ReChiIHpl[2]/(0.0000330846-0.00000481353*12.376+0.000000172747*12.376*12.376)*
      (0.0000330846-0.00000481353*apl*1e8*(1.+DD[k])+0.000000172747*apl*1e8*(1.+DD[k])*apl*1e8*(1.+DD[k]));
    ModChiIH_a[2][k]=fabs(ReChiIH_a[2][k]);
    }
  }
}

  for (int k=1; k<=km;k++)
    {
    Mu0_a[k]=2*M_PI/Lambda*ModChiI0_a[k];
    Kapa_a[1][k]=ModChiIH_a[1][k]/ModChiRH_a[k];
    Kapa_a[2][k]=ModChiIH_a[2][k]/ModChiRH_a[k];
    p_a[1][k]=Kapa_a[1][k];
    p_a[2][k]=Kapa_a[2][k];
    g_a[1][k]=-ModChiI0_a[k]*(sqrt(b_as)+1/sqrt(b_as))/(2*C[1]*ModChiRH_a[k]);
    g_a[2][k]=-ModChiI0_a[k]*(sqrt(b_as)+1/sqrt(b_as))/(2*C[2]*ModChiRH_a[k]);
//Memo8->Lines->Add(FloatToStr(k)+'\t'+FloatToStr(Mu0_a[k])+'\t'+FloatToStr(Kapa_a[1][k])+'\t'+FloatToStr(Kapa_a[2][k]) +'\t'+FloatToStr(p_a[1][k]) +'\t'+FloatToStr(p_a[2][k]) );
    }


}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//void __fastcall TForm1::Button1Click(TObject *Sender)  // Розраховує  диф. складову
//{   RozrachDiduz();    }

void TForm1::RozrachDiduz()
{
  double L;
  double *R_dif_0, *R_dif_0pl, *R_dif_dl, *R_difPun_dl, *R_dif;
  R_dif_0  = new double[m1_teor+1];
  R_dif_0pl= new double[m1_teor+1];
  R_dif_dl = new double[m1_teor+1];
  R_difPun_dl = new double[m1_teor+1];
  R_dif    = new double[m1_teor+1];

  // Занулення всіх даних:
for (int i=0; i<=m1_teor; i++)
  {
  R_dif_0[i]=0;
  R_dif_0pl[i]=0;
  R_dif_dl[i]=0;
  }

if (CheckBox67->Checked==true)         // Приповерхневий порушений шар
  {
  if (fitting==0) if (RadioButton34->Checked==true) Profil_defects(km,DD,dl) ;
  if (fitting==0) if (RadioButton3->Checked==true)  Profil(km,DD,dl) ;
  if (fitting==0) if (RadioButton4->Checked==true)  Profil_shod() ;

  Xi_SL();

  if (RadioButton68->Checked==false)RozrachDiduz_SL(Defekts_SL, R_dif_dl);
  if (RadioButton68->Checked==true) RozrachDiduz_SL_Punegov(R_dif_dl);
  if (RadioButton68->Checked==true)       // !!!!!!! ТИМЧАСОВО  !!!!!!
    for (int n=nC1; n<=nC; n++)
      for (int i=0; i<=m1_teor; i++)
        Fabsj_SL[i][n]=1;

  Memo9->Lines->Add( "RozrachDiduz_SL пройшло");

  L=0;
  for (int k=1; k<=km;k++) L=L+Dl[k] ;
  dl0=t-L;
  }

if (CheckBox31->Checked==true)         // Гетероструктура
  {
  if (CheckBox67->Checked==false)
    {
    for (int n=nC1; n<=nC; n++)
      for (int i=0; i<=m1_teor; i++)
        Fabsj_SL[i][n]=1;
    L=0;
    }
  hpl0=hpl-L;
  dl0=t-hpl;
  RozrachDiduz0pl(Defekts_film, R_dif_0pl);
  Memo9->Lines->Add( "RozrachDiduz0pl пройшло");
  }

if (CheckBox68->Checked==true)         // Монокристал
  {
  if (CheckBox67->Checked==false && CheckBox31->Checked==false)
    {
    for (int n=nC1; n<=nC; n++)
      for (int i=0; i<=m1_teor; i++)
        Fabsj_PL[i][n]=1;
    dl0=t;
    }
  if (CheckBox67->Checked==true && CheckBox31->Checked==false)
    {
    for (int n=nC1; n<=nC; n++)
      for (int i=0; i<=m1_teor; i++)
        Fabsj_PL[i][n]=Fabsj_SL[i][n];
    }
  RozrachDiduz0(Defekts_mon, R_dif_0);
  Memo9->Lines->Add( "RozrachDiduz0 пройшло");
  }

if (KDV_lich==1 && CheckBox62->Checked==true) for (int i=0; i<=m1_teor; i++) R_dif_dl[i]=0;
if (KDV_lich==2 && CheckBox63->Checked==true) for (int i=0; i<=m1_teor; i++) R_dif_dl[i]=0;
if (KDV_lich==3 && CheckBox64->Checked==true) for (int i=0; i<=m1_teor; i++) R_dif_dl[i]=0;

for (int i=0; i<=m1_teor; i++) R_dif_[i][KDV_lich]=R_dif[i]=R_dif_0[i]+R_dif_0pl[i]+R_dif_dl[i];

if (fitting==0 || (fitting==1 && vse==2) || (fitting==10 && vse==2))
{
for (int i=0; i<=m1_teor; i++)
{
	 R_dif_0_[i][KDV_lich]=R_dif_0[i];
	 R_dif_0pl_[i][KDV_lich]=R_dif_0pl[i];
	 R_dif_dl_[i][KDV_lich]=R_dif_dl[i];
}
int op=-m10z;
int ok=m1z-m10z;
        for (int i=0; i<=m1_teor-(op+ok); i++)  //   Зсув КДВ до початку інформативної області
        {
	 R_dif_0_[i][KDV_lich]=R_dif_0_[i+ok][KDV_lich];
	 R_dif_0pl_[i][KDV_lich]=R_dif_0pl_[i+ok][KDV_lich];
	 R_dif_dl_[i][KDV_lich]=R_dif_dl_[i+ok][KDV_lich];
//         R_dif_[i][KDV_lich]=R_dif_[i+ok][KDV_lich];
        }    
}

if (fitting==0)
{
for (int i=0; i<=m1_teor; i++)
{

if (number_KDV==1)
	{
	if (CheckBox19->Checked==false)
	{
	Series9->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	Series15->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	}
	Series17->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);

	Series16->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif_0[i]);
	Series7->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif_dl[i]);
	}
if (number_KDV==2)
	{
	if (CheckBox19->Checked==false)
	{
	if (KDV_lich==2) 	Series9->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	if (KDV_lich==1) 	Series15->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	}
	}
if (number_KDV==3)
	{
	if (CheckBox19->Checked==false)
	{
	if (KDV_lich==3) 	Series46->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	if (KDV_lich==2) 	Series9->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	if (KDV_lich==1) 	Series15->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_dif[i],"",clPurple	);
	}
	}
}
}
if (CheckBox53->Checked==true) Memo8-> Lines->Add("Дифузне пораховано!");
  delete R_dif_0, R_dif_0pl, R_dif_dl, R_difPun_dl, R_dif;
}

//---------------------------------------------------------------------------
void TForm1::RozrachDiduz_SL(double *Defekts_SL, double *R_dif_dl)
{         // Розраховує  дифузну складову від  ПШ
//double Rint_dl [MM], RintP_dl [MM], Rintd_dl [MM];
//double LhD [KM],double Lhp [KM],double Lhpd [KM],double LhSum [KM];
  double *Rint_dl,*Rint_an_dl, *RintP_dl,*RintPtd_dl, *Rintd_dl;
  double *LhD,*LhD_an,*Lhp, *Lhpd, *Lhptd, *LhSum;
  Rint_dl = new double[m1_teor+1];
  Rint_an_dl = new double[m1_teor+1];
  RintP_dl= new double[m1_teor+1];
  RintPtd_dl= new double[m1_teor+1];
  Rintd_dl= new double[m1_teor+1];
  LhD  = new double[km+1];
  LhD_an  = new double[km+1];
  Lhp  = new double[km+1];
  Lhpd = new double[km+1];
  Lhptd  = new double[km+1];
  LhSum= new double[km+1];
double E444, E888, E880, Ekoef, stepin, Ekoef2, stepin2, H444;
double *zz, *Esum444, *L_vruchnu, *L_vruchnu2;
  zz = new double[km+1];
  Esum444 = new double[km+1];
  L_vruchnu = new double[km+1];
  L_vruchnu2 = new double[km+1];
double LhD_max,Lhp_max, Lhptd_max, Lhpd_max, LhSum_max;
double ED_min,Ep_min, Eptd_min, Epd_min, hp;
double **FabsjD_dl,**FabsjD_an_dl,**FabsjP_dl,**FabsjPd_dl,**FabsjPtd_dl;
FabsjD_dl = new double*[m1_teor+1];
FabsjD_an_dl = new double*[m1_teor+1];
FabsjP_dl = new double*[m1_teor+1];
FabsjPd_dl = new double*[m1_teor+1];
FabsjPtd_dl = new double*[m1_teor+1];
for(int i=0;i<m1_teor+1; i++)
{
    FabsjD_dl[i]    = new double[3];
    FabsjD_an_dl[i] = new double[3];
    FabsjP_dl[i]    = new double[3];
    FabsjPd_dl[i]   = new double[3];
    FabsjPtd_dl[i]  = new double[3];
}
double ***MuDSj,***MuDSj_an,***MuDSPj,***MuDSPjtd,***MuDSdj;
MuDSj    = new double**[m1_teor+1];
MuDSj_an = new double**[m1_teor+1];
MuDSPj   = new double**[m1_teor+1];
MuDSPjtd = new double**[m1_teor+1];
MuDSdj   = new double**[m1_teor+1];
for(int i=0;i<m1_teor+1; i++)
  {
  MuDSj[i]    = new double*[KM];
  MuDSj_an[i] = new double*[KM];
  MuDSPj[i]   = new double*[KM];
  MuDSPjtd[i] = new double*[KM];
  MuDSdj[i]   = new double*[KM];
  }
for(int i=0;i<m1_teor+1; i++)
  for(int k=1;k<KM; k++)
    {
    MuDSj[i][k]    = new double[3];
    MuDSj_an[i][k] = new double[3];
    MuDSPj[i][k]   = new double[3];
    MuDSPjtd[i][k] = new double[3];
    MuDSdj[i][k]   = new double[3];
    }
// Занулення всіх даних:
for (int i=0; i<=m1_teor; i++)
  {
  Rint_dl[i]=0;
  Rint_an_dl[i]=0;
  RintP_dl[i]=0;
  Rintd_dl[i]=0;
  RintPtd_dl[i]=0;
  R_dif_dl[i]=0;
  }
for (int k=1; k<=km;k++)
  {
  LhD[k]=0;
  LhD_an[k]=0;
  Lhp[k]=0;
  Lhpd[k]=0;
  Lhptd[k]=0;
  LhSum[k]=0;
  }

for (int n=nC1; n<=nC; n++)
  for (int i=0; i<=m1_teor; i++)
    {
    Fabsj_SL[i][n]=1;
    FabsjD_dl[i][n]=1;
    FabsjD_an_dl[i][n]=1;
    FabsjP_dl[i][n]=1;
    FabsjPd_dl[i][n]=1;
    FabsjPtd_dl[i][n]=1;
    }

//Розрахунок для дислокаційних петель
if (CheckBox1->Checked==true)
{
DifuzSL_Loop(Defekts_SL[1], Defekts_SL[2], LhD, MuDSj, FabsjD_dl, Rint_dl); // (R0_max, nL_max, LhD, Rint_dl);
}
//Розрахунок для дислокаційних петель із урахуванням анізотропії
if (CheckBox58->Checked==true)
{
DifuzSL_LoopAniz(Defekts_SL[13], Defekts_SL[14], LhD_an,MuDSj_an, FabsjD_an_dl, Rint_an_dl); // (R0_max, nL_max, LhD, Rint_dl);
}

//розрахунок для сферичних кластерів
if (CheckBox2->Checked==true)
{
DifuzSL_SferClaster(Defekts_SL[4], Defekts_SL[5],Defekts_SL[6],Lhp,MuDSPj,FabsjP_dl,RintP_dl); // (R0p_max, np_max,eps,Lhp,RintP_dl);
}

//розрахунок для дископодібних кластерів
if (CheckBox4->Checked==true)
{
DiduzSL_DiscClaster(Defekts_SL[7], Defekts_SL[8], Defekts_SL[9],Lhpd,MuDSdj,FabsjPd_dl,Rintd_dl); // (R0d_max, nd_max, epsd,Lhpd,Rintd_dl);
}
//розрахунок для дископодібних кластерів із урахуванням анізотропії
if (CheckBox85->Checked==true && CheckBox4->Checked==true)
{
DiduzSL_DiscClasterAniz(Defekts_SL[7], Defekts_SL[8], Defekts_SL[9],Lhpd,MuDSdj,FabsjPd_dl,Rintd_dl); // (R0d_max, nd_max, epsd,Lhpd,Rintd_dl);
}

//розрахунок для сферичних кластерів (точкові дефекти)
if (CheckBox26->Checked==true)
{
DifuzSL_SferClaster(Defekts_SL[10], Defekts_SL[11],Defekts_SL[12],Lhptd,MuDSPjtd,FabsjPtd_dl,RintPtd_dl); // (R0ptd_max, nptd_max,epstd,Lhp,RintP_dl);
}

//розрахунок дифузного фону  за Пунеговим
if (CheckBox83->Checked==true)
{
DifuzSL_PynByshKato(Defekts_SL[10], Defekts_SL[11],Defekts_SL[12],Lhptd,MuDSPjtd,FabsjPtd_dl,RintPtd_dl); // (R0ptd_max, nptd_max,epstd,Lhp,RintP_dl);
}

for (int n=nC1; n<=nC; n++)
  for (int i=0; i<=m1_teor; i++)
    Fabsj_SL[i][n]=FabsjD_dl[i][n]*FabsjD_an_dl[i][n]*FabsjP_dl[i][n]*FabsjPd_dl[i][n]*FabsjPtd_dl[i][n];

for (int i=0; i<=m1_teor; i++)
{
//DeltaTeta1=(TetaMin+i*ik);
R_dif_dl[i]=Rint_dl[i]+Rint_an_dl[i]+RintP_dl[i]+Rintd_dl[i]+RintPtd_dl[i];
//Series7->AddXY(DeltaTeta1,R_dif_dl[i],"",clPurple);
//Series18->AddXY(DeltaTeta1,RintP_dl[i],"",clGreen);
//Series19->AddXY(DeltaTeta1,Rint_dl[i],"",clGreen);
//Series20->AddXY(DeltaTeta1,Rintd_dl[i],"",clGreen);
}

for (int i=0; i<=m1_teor; i++)
  for (int n=nC1; n<=nC; n++)
    for (int k=1; k<=km;k++)
      MuDSsum_dl[i][k][n]=MuDSj[i][k][n]+MuDSj_an[i][k][n]+MuDSPj[i][k][n]+MuDSdj[i][k][n]+MuDSPjtd[i][k][n];


  for (int k=1; k<=km;k++) zz[k]=dl*km-dl*k+dl/2;
  for (int k=1; k<=km;k++) L_vruchnu[k]=1e-9;
  if (CheckBox3->Checked==true)     // 1- Аморфізацію задаємо вручну!!!
    {
    Ekoef=StrToFloat(Edit398->Text);
    stepin=StrToFloat(Edit399->Text);
    for (int k=1; k<=km;k++) Esum444[k]=1-exp(stepin*LogN(M_E,f[k]))*Ekoef;
    H444=sqrt(4*4+4*4+4*4)/a;        //=1/d
    if (KDV_lich==1)
      {
      for (int k=1; k<=km;k++) Esum[k]=Esum444[k];
      if (RadioButton69->Checked==true) for (int k=1; k<=km;k++) Series54->AddXY(zz[k],Esum[k],"",clOlive);
      E444=1;
      for (int k=1; k<=km; k++) if (E444>Esum[k]) E444=Esum[k];
      Edit131->Text=FloatToStr(E444);
      }
    if (KDV_lich==2)
      {
      for (int k=1; k<=km;k++) Esum[k]=exp(LogN(M_E,Esum444[k])*exp(3/2.*LogN(M_E,H/H444)));
      if (RadioButton70->Checked==true) for (int k=1; k<=km;k++) Series54->AddXY(zz[k],Esum[k],"",clOlive);
      E888=1;
      for (int k=1; k<=km; k++) if (E888>Esum[k]) E888=Esum[k];
      Edit396->Text=FloatToStr(E888);
      }
    if (KDV_lich==3)
      {
      for (int k=1; k<=km;k++) Esum[k]=exp(LogN(M_E,Esum444[k])*exp(3/2.*LogN(M_E,H/H444)));
      if (RadioButton71->Checked==true) for (int k=1; k<=km;k++) Series54->AddXY(zz[k],Esum[k],"",clOlive);
      E880=1;
      for (int k=1; k<=km; k++) if (E880>Esum[k]) E880=Esum[k];
      Edit397->Text=FloatToStr(E880);
      }
    for (int k=1; k<=km;k++) L_vruchnu[k]=-LogN(M_E,Esum[k]);
    }

  for (int k=1; k<=km;k++) L_vruchnu2[k]=1e-9;
  if (CheckBox88->Checked==true)     // 2- Аморфізацію задаємо вручну!!!
    {
    Ekoef2=StrToFloat(Edit407->Text);
    stepin2=StrToFloat(Edit408->Text);
    for (int k=1; k<=km;k++) Esum444[k]=1-exp(stepin2*LogN(M_E,f[k]))*Ekoef2;
    H444=sqrt(4*4+4*4+4*4)/a;        //=1/d
    if (KDV_lich==1)
      {
      for (int k=1; k<=km;k++) Esum[k]=Esum444[k];
      if (RadioButton69->Checked==true) for (int k=1; k<=km;k++) Series29->AddXY(zz[k],Esum[k],"",clOlive);
      E444=1;
      for (int k=1; k<=km; k++) if (E444>Esum[k]) E444=Esum[k];
      Edit409->Text=FloatToStr(E444);
      }
    if (KDV_lich==2)
      {
      for (int k=1; k<=km;k++) Esum[k]=exp(LogN(M_E,Esum444[k])*exp(3/2.*LogN(M_E,H/H444)));
      if (RadioButton70->Checked==true) for (int k=1; k<=km;k++) Series29->AddXY(zz[k],Esum[k],"",clOlive);
      E888=1;
      for (int k=1; k<=km; k++) if (E888>Esum[k]) E888=Esum[k];
      Edit410->Text=FloatToStr(E888);
      }
    if (KDV_lich==3)
      {
      for (int k=1; k<=km;k++) Esum[k]=exp(LogN(M_E,Esum[k])*exp(3/2.*LogN(M_E,H/H444)));
      if (RadioButton71->Checked==true) for (int k=1; k<=km;k++) Series29->AddXY(zz[k],Esum444[k],"",clOlive);
      E880=1;
      for (int k=1; k<=km; k++) if (E880>Esum[k]) E880=Esum[k];
      Edit411->Text=FloatToStr(E880);
      }
    for (int k=1; k<=km;k++) L_vruchnu2[k]=-LogN(M_E,Esum[k]);
    }

LhSum_max=0;
for (int k=1; k<=km;k++)
  {
  LhSum[k]=LhD[k]+LhD_an[k]+Lhp[k]+Lhpd[k]+Lhptd[k];
  if ((KDV_lich==1 && RadioButton69->Checked==true)||(KDV_lich==2 && RadioButton70->Checked==true)
        ||(KDV_lich==3 && RadioButton71->Checked==true))Series58->AddXY(zz[k],exp(-LhSum[k]),"",clPurple);
  LhSum[k]=LhSum[k]+L_vruchnu[k]+L_vruchnu2[k];
  if ((KDV_lich==1 && RadioButton69->Checked==true)||(KDV_lich==2 && RadioButton70->Checked==true)
        ||(KDV_lich==3 && RadioButton71->Checked==true))Series55->AddXY(zz[k],exp(-(L_vruchnu[k]+L_vruchnu2[k])),"",clMaroon);
  if ((KDV_lich==1 && RadioButton69->Checked==true)||(KDV_lich==2 && RadioButton70->Checked==true)
        ||(KDV_lich==3 && RadioButton71->Checked==true))Series57->AddXY(zz[k],exp(-LhSum[k]),"",clRed);
  if (LhSum_max<LhSum[k]) LhSum_max=LhSum[k];
  Esum[k]=exp(-LhSum[k]);
  }
  Emin=exp(-LhSum_max);

if (fitting==0)
{
  if (CheckBox1->Checked==true)
  {
  LhD_max=0;
  for (int k=1; k<=km;k++) if (LhD_max<LhD[k]) LhD_max=LhD[k];
  ED_min=exp(-LhD_max);
  if (KDV_lich==1)
    {
    Edit247->Text=FloatToStr(LhD_max);
    Edit248->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==2)
    {
    Edit363->Text=FloatToStr(LhD_max);
    Edit362->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==3)
    {
    Edit365->Text=FloatToStr(LhD_max);
    Edit364->Text=FloatToStr(ED_min);
    }
  }
  if (CheckBox58->Checked==true)
  {
  LhD_max=0;
  for (int k=1; k<=km;k++) if (LhD_max<LhD_an[k]) LhD_max=LhD_an[k];
  ED_min=exp(-LhD_max);
  if (KDV_lich==1)
    {
    Edit270->Text=FloatToStr(LhD_max);
    Edit271->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==2)
    {
    Edit281->Text=FloatToStr(LhD_max);
    Edit283->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==3)
    {
    Edit286->Text=FloatToStr(LhD_max);
    Edit287->Text=FloatToStr(ED_min);
    }
  }
  if (CheckBox2->Checked==true)
  {
  Lhp_max=0;
  for (int k=1; k<=km;k++) if (Lhp_max<Lhp[k]) Lhp_max=Lhp[k];
  Ep_min=exp(-Lhp_max);
  if (KDV_lich==1)
    {
    Edit246->Text=FloatToStr(Lhp_max);
    Edit249->Text=FloatToStr(Ep_min);
    }
  if (KDV_lich==2)
    {
    Edit366->Text=FloatToStr(Lhp_max);
    Edit367->Text=FloatToStr(Ep_min);
    }
  if (KDV_lich==3)
    {
    Edit368->Text=FloatToStr(Lhp_max);
    Edit369->Text=FloatToStr(Ep_min);
    }
  }
  if (CheckBox26->Checked==true)
  {
  Lhptd_max=0;
  for (int k=1; k<=km;k++) if (Lhptd_max<Lhptd[k]) Lhptd_max=Lhptd[k];
  Eptd_min=exp(-Lhptd_max);
  if (KDV_lich==1)
    {
    Edit254->Text=FloatToStr(Lhptd_max);
    Edit255->Text=FloatToStr(Eptd_min);
    }
  if (KDV_lich==2)
    {
    Edit370->Text=FloatToStr(Lhptd_max);
    Edit371->Text=FloatToStr(Eptd_min);
    }
  if (KDV_lich==3)
    {
    Edit372->Text=FloatToStr(Lhptd_max);
    Edit373->Text=FloatToStr(Eptd_min);
    }
  }
  if (CheckBox4->Checked==true)
  {
  Lhpd_max=0;
  for (int k=1; k<=km;k++) if (Lhpd_max<Lhpd[k]) Lhpd_max=Lhpd[k];
  Epd_min=exp(-Lhpd_max);
  if (KDV_lich==1)
    {
    Edit256->Text=FloatToStr(Lhpd_max);
    Edit257->Text=FloatToStr(Epd_min);
    }
  if (KDV_lich==2)
    {
    Edit374->Text=FloatToStr(Lhpd_max);
    Edit375->Text=FloatToStr(Epd_min);
    }
  if (KDV_lich==3)
    {
    Edit376->Text=FloatToStr(Lhpd_max);
    Edit377->Text=FloatToStr(Epd_min);
    }
  }
if (KDV_lich==1)
  {
  Edit17->Text=FloatToStr(LhSum_max);
  Edit18->Text=FloatToStr(Emin);
  }
if (KDV_lich==2)
  {
  Edit378->Text=FloatToStr(LhSum_max);
  Edit379->Text=FloatToStr(Emin);
  }
if (KDV_lich==3)
  {
  Edit380->Text=FloatToStr(LhSum_max);
  Edit381->Text=FloatToStr(Emin);
  }
}

if (fitting==0 || vse==1) for (int k=1;k<=km;k++) Esum_[k]=Esum[k];

delete Rint_dl, Rint_an_dl, RintP_dl, RintPtd_dl, Rintd_dl;
delete LhD, LhD_an, Lhp, Lhptd, Lhpd, LhSum,zz, Esum444, L_vruchnu, L_vruchnu2;
for(int i=0; i<m1_teor+1; i++)
{
  delete[] FabsjD_dl[i];
  delete[] FabsjD_an_dl[i];
  delete[] FabsjP_dl[i];
  delete[] FabsjPd_dl[i];
  delete[] FabsjPtd_dl[i];
}
delete[] FabsjD_dl;
delete[] FabsjD_an_dl;
delete[] FabsjP_dl;
delete[] FabsjPd_dl;
delete[] FabsjPtd_dl;

for(int i=0;i<m1_teor+1; i++)
  for(int k=1;k<KM; k++)
    {
    delete []MuDSj[i][k];
    delete []MuDSj_an[i][k];
    delete []MuDSPj[i][k];
    delete []MuDSPjtd[i][k];
    delete []MuDSdj[i][k];
    }
for(int i=0; i<m1_teor+1; i++)
  {
  delete[] MuDSj[i];
  delete[] MuDSj_an[i];
  delete[] MuDSPj[i];
  delete[] MuDSPjtd[i];
  delete[] MuDSdj[i];
  }
delete[] MuDSj;
delete[] MuDSj_an;
delete[] MuDSPj;
delete[] MuDSPjtd;
delete[] MuDSdj;
}

//---------------------------------------------------------------------------
void TForm1::RozrachDiduz0pl(double *Defekts_film, double *R_dif_0pl)
{         // Розраховує  дифузну складову   (ід. частина плівки)
//double Rint1pl[MM],Rint2pl[MM],RintP1pl[MM],RintP2pl[MM],RintPtdpl[MM],Rintdpl[MM];
//double MuDS1pl[MM],MuDS2pl[MM],MuDSP1pl[MM],MuDSP2pl[MM],MuDSPtdpl[MM],MuDSdpl[MM],MuDSsumpl[MM];   // !!!!!!! Поки що не використовується
  double *Rint1pl, *Rint2pl, *RintP1pl, *RintP2pl, *RintPtdpl, *Rintdpl;
  double *MuDS1pl, *MuDS2pl, *MuDSP1pl, *MuDSP2pl, *MuDSPtdpl, *MuDSdpl, *MuDSsumpl;
  Rint1pl = new double[m1_teor+1];
  Rint2pl= new double[m1_teor+1];
  RintP1pl= new double[m1_teor+1];
  RintP2pl  = new double[m1_teor+1];
  RintPtdpl  = new double[m1_teor+1];
  Rintdpl = new double[m1_teor+1];
  MuDS1pl= new double[m1_teor+1];
  MuDS2pl= new double[m1_teor+1];
  MuDSP1pl= new double[m1_teor+1];
  MuDSP2pl= new double[m1_teor+1];
  MuDSPtdpl= new double[m1_teor+1];
  MuDSdpl= new double[m1_teor+1];
  MuDSsumpl= new double[m1_teor+1];  // !!!!!!! Поки що не використовується
double  LhD01pl,LhD02pl,Lhp01pl,Lhp02pl,Lhp0tdpl,Lhpd0pl,LhSum0pl;
double **FabsjD1_pl,**FabsjD2_pl,**FabsjP1_pl,**FabsjP2_pl,**FabsjPd_pl,**FabsjPtd_pl;
FabsjD1_pl = new double*[m1_teor+1];
FabsjD2_pl = new double*[m1_teor+1];
FabsjP1_pl = new double*[m1_teor+1];
FabsjP2_pl = new double*[m1_teor+1];
FabsjPd_pl = new double*[m1_teor+1];
FabsjPtd_pl = new double*[m1_teor+1];
for(int i=0;i<m1_teor+1; i++)
{
    FabsjD1_pl[i]  = new double[3];
    FabsjD2_pl[i]  = new double[3];
    FabsjP1_pl[i]  = new double[3];
    FabsjP2_pl[i]  = new double[3];
    FabsjPd_pl[i]  = new double[3];
    FabsjPtd_pl[i] = new double[3];
}

// Занулення всіх даних:
for (int i=0; i<=m1_teor; i++)
{
Rint1pl[i]=0;
Rint2pl[i]=0;
RintP1pl[i]=0;
RintP2pl[i]=0;
RintPtdpl[i]=0;
Rintdpl[i]=0;
R_dif_0pl[i]=0;
MuDS1pl[i]=0;
MuDS2pl[i]=0;
MuDSP1pl[i]=0;
MuDSP2pl[i]=0;
MuDSPtdpl[i]=0;
MuDSdpl[i]=0;
MuDSsumpl[i]=0;
}
LhD01pl=0;
LhD02pl=0;
Lhp01pl=0;
Lhp02pl=0;
Lhp0tdpl=0;
Lhpd0pl=0;

for (int n=nC1; n<=nC; n++)
  for (int i=0; i<=m1_teor; i++)
    {
    Fabsj_PL[i][n]=1;
    FabsjD1_pl[i][n]=1;
    FabsjD2_pl[i][n]=1;
    FabsjP1_pl[i][n]=1;
    FabsjP2_pl[i][n]=1;
    FabsjPd_pl[i][n]=1;
    FabsjPtd_pl[i][n]=1;
    }

//Розрахунок для дислокаційних петель (ід. частина плівки)
if (CheckBox34->Checked==true)
{
Difuz0pl_Loop(Defekts_film[1],Defekts_film[2],LhD01pl,MuDS1pl,FabsjD1_pl,Rint1pl);  //(R001pl,nL01pl,LhD01pl,MuDS1pl, Rint1pl);
}

//Розрахунок для дислокаційних петель 2 (ід. частина плівки)
if (CheckBox35->Checked==true)
{
Difuz0pl_Loop(Defekts_film[4],Defekts_film[5],LhD02pl,MuDS2pl,FabsjD2_pl, Rint2pl);  //(R002pl,nL02pl,LhD02pl,MuDS2pl, Rint2pl);
}

//розрахунок для сферичних кластерів (ід. частина плівки)
if (CheckBox32->Checked==true)
{
Difuz0pl_SferClaster(Defekts_film[7],Defekts_film[8],Defekts_film[9],Lhp01pl,MuDSP1pl,FabsjP1_pl, RintP1pl); // (R0p01pl, np01pl, eps01pl, Lhp01pl,MuDSP1pl, RintP1pl);
}

//розрахунок для сферичних кластерів 2 (ід. частина плівки)
if (CheckBox33->Checked==true)
{
Difuz0pl_SferClaster(Defekts_film[10],Defekts_film[11],Defekts_film[12], Lhp02pl,MuDSP2pl,FabsjP2_pl, RintP2pl);  // (R0p02pl, np02pl, eps02pl, Lhp02pl,MuDSP2pl, RintP2pl);
}

//розрахунок для сферичних кластерів - точкові дефекти (ід. частина плівки)
if (CheckBox37->Checked==true)
{
Difuz0pl_SferClaster(Defekts_film[16],Defekts_film[17],Defekts_film[18],Lhp0tdpl,MuDSPtdpl,FabsjPtd_pl, RintPtdpl);  //(R0p0tdpl, np0tdpl, eps0tdpl, Lhp0tdpl,MuDSPtdpl, RintPtdpl);
}

//розрахунок для дископодібних кластерів (ід. частина плівки)
if (CheckBox36->Checked==true)
{
Diduz0pl_DiscClaster(Defekts_film[13],Defekts_film[14],Defekts_film[15], Lhpd0pl, MuDSdpl,FabsjPd_pl,Rintdpl);  //(R0d0pl,nd0pl,eps0dpl,Lhpd0pl,MuDSdpl,Rintdpl);
}

for (int n=nC1; n<=nC; n++)
  for (int i=0; i<=m1_teor; i++)
    Fabsj_PL[i][n]=FabsjD1_pl[i][n]*FabsjD2_pl[i][n]*FabsjP1_pl[i][n]*FabsjP2_pl[i][n]*FabsjPd_pl[i][n]*FabsjPtd_pl[i][n];

//TetaMin=-(m10)*ik;
for (int i=0; i<=m1_teor; i++)
{
//DeltaTeta1=(TetaMin+i*ik);
R_dif_0pl[i]=Rint1pl[i]+Rint2pl[i]+RintP1pl[i]+RintP2pl[i]+RintPtdpl[i]+Rintdpl[i];
//Series16->AddXY(DeltaTeta1,R_dif_0pl[i],"",clPurple);
//Series21->AddXY(DeltaTeta1,RintP[i],"",clGreen);
//Series27->AddXY(DeltaTeta1,RintP2[i],"",clGreen);
//Series29->AddXY(DeltaTeta1,RintPtd[i],"",clGreen);
//Series22->AddXY(DeltaTeta1,Rint[i],"",clGreen);
//Series28->AddXY(DeltaTeta1,Rint2[i],"",clGreen);
//Series23->AddXY(DeltaTeta1,Rintd[i],"",clGreen);
//MuDSsum[i]=MuDS1[i]+MuDS2[i]+MuDSP1[i]+MuDSP2[i]+MuDSPt[i]+MuDSd[i];
MuDSsumpl[i]=MuDS1pl[i]+MuDS2pl[i]+MuDSP1pl[i]+MuDSP2pl[i]+MuDSPtdpl[i]+MuDSdpl[i];
}
LhSum0pl=LhD01pl+LhD02pl+Lhp01pl+Lhp02pl+Lhp0tdpl+Lhpd0pl;
Esum0pl=exp(-LhSum0pl);
Edit191->Text=FloatToStr(LhSum0pl);
Edit190->Text=FloatToStr(Esum0pl);

  delete Rint1pl, Rint2pl, RintP1pl, RintP2pl, RintPtdpl, Rintdpl;
  delete MuDS1pl, MuDS2pl, MuDSP1pl, MuDSP2pl, MuDSPtdpl, MuDSdpl, MuDSsumpl;
for(int i=0; i<m1_teor+1; i++)
{
  delete[] FabsjD1_pl[i];
  delete[] FabsjD2_pl[i];
  delete[] FabsjP1_pl[i];
  delete[] FabsjP2_pl[i];
  delete[] FabsjPd_pl[i];
  delete[] FabsjPtd_pl[i];
}
delete[] FabsjD1_pl;
delete[] FabsjD2_pl;
delete[] FabsjP1_pl;
delete[] FabsjP2_pl;
delete[] FabsjPd_pl;
delete[] FabsjPtd_pl;
}

//---------------------------------------------------------------------------
void TForm1::RozrachDiduz0(double *Defekts_mon, double *R_dif_0)
{        // Розраховує  дифузну складову   (ід. частина монокристалу)
//double Rint1[MM],Rint2[MM],RintP1[MM],RintP2[MM],RintPtd[MM],Rintd[MM];
//double MuDS1[MM],MuDS2[MM],MuDSP1[MM],MuDSP2[MM],MuDSPtd[MM],MuDSd[MM];
  double *Rint1, *Rint2, *Rint_an, *RintP1, *RintP2, *RintPtd, *Rintd;
  double *MuDS1, *MuDS2, *MuDS_an, *MuDSP1, *MuDSP2, *MuDSPtd, *MuDSd;
  Rint1 = new double[m1_teor+1];
  Rint2= new double[m1_teor+1];
  Rint_an= new double[m1_teor+1];
  RintP1= new double[m1_teor+1];
  RintP2  = new double[m1_teor+1];
  RintPtd  = new double[m1_teor+1];
  Rintd = new double[m1_teor+1];
  MuDS1= new double[m1_teor+1];
  MuDS2= new double[m1_teor+1];
  MuDS_an= new double[m1_teor+1];
  MuDSP1= new double[m1_teor+1];
  MuDSP2= new double[m1_teor+1];
  MuDSPtd= new double[m1_teor+1];
  MuDSd= new double[m1_teor+1];
double  LhD01,LhD02,LhD0_an,Lhp01,Lhp02,Lhp0td,Lhpd0,LhSum0;

// Занулення всіх даних:
for (int i=0; i<=m1_teor; i++)
{
Rint1[i]=0;
Rint2[i]=0;
Rint_an[i]=0;
RintP1[i]=0;
RintP2[i]=0;
RintPtd[i]=0;
Rintd[i]=0;
R_dif_0[i]=0;
MuDS1[i]=0;
MuDS2[i]=0;
MuDS_an[i]=0;
MuDSP1[i]=0;
MuDSP2[i]=0;
MuDSPtd[i]=0;
MuDSd[i]=0;
MuDSsum[i]=0;
}
LhD01=0;
LhD02=0;
LhD0_an=0;
Lhp01=0;
Lhp02=0;
Lhp0td=0;
Lhpd0=0;

//Розрахунок для дислокаційних петель (ід. частина монокристалу)
if (CheckBox13->Checked==true)
{
Difuz0_Loop(Defekts_mon[1],Defekts_mon[2],LhD01,MuDS1, Rint1); //(R001,nL02,LhD01,MuDS1, Rint1);
}

//Розрахунок для дислокаційних петель 2 (ід. частина монокристалу)
if (CheckBox17->Checked==true)
{
Difuz0_Loop(Defekts_mon[4],Defekts_mon[5],LhD02,MuDS2, Rint2);  //(R002,nL02,LhD02,MuDS2, Rint2);
}

//Розрахунок для дислокаційних петель із урахуванням анізотропії (ід. частина монокристалу)
if (CheckBox55->Checked==true)
{
Difuz0_LoopAniz(Defekts_mon[19],Defekts_mon[20],LhD0_an,MuDS_an, Rint_an); //(R002,nL02,LhD02,MuDS2, Rint2);
}

//розрахунок для сферичних кластерів (ід. частина монокристалу)
if (CheckBox12->Checked==true)
{
Difuz0_SferClaster(Defekts_mon[7],Defekts_mon[8],Defekts_mon[9],Lhp01,MuDSP1, RintP1); //(R0p01,np01,eps01,Lhp01,MuDSP1, RintP1);
}

//розрахунок для сферичних кластерів 2 (ід. частина монокристалу)
if (CheckBox15->Checked==true)
{
Difuz0_SferClaster(Defekts_mon[10],Defekts_mon[11],Defekts_mon[12],Lhp02,MuDSP2, RintP2); //(R0p01,np01,eps01,Lhp01,MuDSP1, RintP1);
}

//розрахунок для сферичних кластерів - точкові дефекти (ід. частина монокристалу)
if (CheckBox14->Checked==true)
{
Difuz0_SferClaster(Defekts_mon[16],Defekts_mon[17],Defekts_mon[18],Lhp0td,MuDSPtd, RintPtd); //(R0p0td,np0td,eps0td,Lhp0td,MuDSPtd, RintPtd);
}

//розрахунок для дископодібних кластерів (ід. частина монокристалу)
if (CheckBox16->Checked==true)
{
Diduz0_DiscClaster(Defekts_mon[13],Defekts_mon[14],Defekts_mon[15],Lhpd0,MuDSd, Rintd); //(R0d0,nd0,eps0d,Lhpd0,MuDSd, Rintd);
}

for (int i=0; i<=m1_teor; i++)
{
//DeltaTeta1=(TetaMin+i*ik);
R_dif_0[i]=Rint1[i]+Rint2[i]+Rint_an[i]+RintP1[i]+RintP2[i]+RintPtd[i]+Rintd[i];
//Series16->AddXY(DeltaTeta1,R_dif_0[i],"",clPurple);
//Series21->AddXY(DeltaTeta1,RintP[i],"",clGreen);
//Series27->AddXY(DeltaTeta1,RintP2[i],"",clGreen);
//Series29->AddXY(DeltaTeta1,RintPtd[i],"",clGreen);
//Series22->AddXY(DeltaTeta1,Rint[i],"",clGreen);
//Series28->AddXY(DeltaTeta1,Rint2[i],"",clGreen);
//Series23->AddXY(DeltaTeta1,Rintd[i],"",clGreen);
MuDSsum[i]=MuDS1[i]+MuDS2[i]+MuDS_an[i]+MuDSP1[i]+MuDSP2[i]+MuDSPtd[i]+MuDSd[i];
}

LhSum0=LhD01+LhD02+LhD0_an+Lhp01+Lhp02+Lhp0td+Lhpd0;
Esum0=exp(-LhSum0);
Edit59->Text=FloatToStr(LhSum0);
Edit58->Text=FloatToStr(Esum0);

  delete Rint1, Rint2, Rint_an, RintP1, RintP2, RintPtd, Rintd;
  delete MuDS1, MuDS2, MuDS_an, MuDSP1, MuDSP2, MuDSPtd, MuDSd;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void TForm1::RozrachDiduz_SL_Punegov(double *R_dif_dl)
{         // Розраховує  дифузну складову від  ПШ ша Пунеговим
  double  *LhSum;
  LhSum= new double[km+1];
double LhD_max,Lhp_max, Lhptd_max, Lhpd_max, LhSum_max;
double ED_min,Ep_min, Eptd_min, Epd_min, hp;
// Занулення всіх даних:
for (int i=0; i<=m1_teor; i++)
  {
  R_dif_dl[i]=0;
  }
for (int k=1; k<=km;k++)
  {
  LhSum[k]=0;
  }




double R[3];
//double L,hpl0;   // DDpd[KM],
  double *DDpd;
  DDpd   = new double[KM];
complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);
double      *x0i_a,*eta0_a;
  x0i_a   = new double[KM];
  eta0_a  = new double[KM];
complex< double>  xhp_a[3][KM],xhn_a[3][KM],eta_a,sigmasp_a,sigmasn_a;
double d,  TAU[KM], kTau,expcsr,expcsi,expcsi1;
complex< double>  RO[KM], TS[3];
long double       expcsr1;
         //x0r0=0;
         //x0r=0;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

for (int k=1; k<=km;k++)
{
//         x0i0=ChiI0_a[k];
      x0i_a[k]=ChiI0_a[k];
      xhp_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);   //для  центр.-сим. крист.
      xhn_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);
      xhp_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);   //для  центр.-сим. крист.
      xhn_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);
      eta0_a[k]=M_PI*x0i_a[k]*(1+b_as)/(Lambda*gamma0);
}

/*
if (CheckBox3->Checked==true)     // !!!!!! ТУТ Однакове для всіх рефлексів
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);
  d=a/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) Memo3-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(TAU[k])+'\t'+FloatToStr(Esum[k]));
   */
   
if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
        DD0=(apl-a)/a;
 for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1)*(DD0+1)-1 ;
//      L=0;
// for (int k=1; k<=km;k++) L=L+Dl[k] ;
//      hpl0=hpl-L;
  d=apl/sqrt(h*h+k*k+l*l);
}
if (CheckBox31->Checked==false)
{
for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
}

kTau=1e-8*StrToFloat(Edit395->Text);    //см
if (CheckBox84->Checked==true) for (int k=1; k<=km;k++) TAU[k]=f[k]*kTau;
if (CheckBox84->Checked==false) for (int k=1; k<=km;k++) TAU[k]=kTau;

      eta00=M_PI*x0i0*(1+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

//Memo9-> Lines->Add("  Обчислення теор. когер. КДВ  ");

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=+(eta0*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));  // тут змінено спереду знак

       for (int n=nC1; n<=nC; n++)
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
//Memo9-> Lines->Add(FloatToStr(real(sigmasp))+'\t'+FloatToStr(imag(sigmasp))+'\t'+FloatToStr(abs(5555)));



//      Обчислення амплітуди від заданого профілю:
/*  if (CheckBox67->Checked==false) goto m102ps;  */  // якщо пор. шару немає
        for (int k=1; k<=km;k++)
{
      eta_a=-(eta0_a[k]*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
      sigmasp_a=M_PI*xhp_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn_a=M_PI*xhn_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));
}

TS[n]=0.;
for (int k=1; k<=km;k++)
  {
  TS[n]=TS[n]+Dl[k]*(1.-Esum[k]*Esum[k])*TAU[k]*exp(-(TAU[k]*eta-2.*M_PI*DDpd[k]/d*TAU[k])*(TAU[k]*eta-2.*M_PI*DDpd[k]/d*TAU[k])/M_PI);
///  TS[n]=TS[n]+Dl[k]*(1.-Esum[k]*Esum[k])*TAU[k]*exp(-(TAU[k]*(real(eta)+2.*M_PI*DD[k]/d))*(TAU[k]*(real(eta)+2.*M_PI*DD[k]/d))/M_PI);
  }
TS[n]=2.*abs(sigmasp_a)*abs(sigmasp_a)*TS[n];

/*c       Обчислення дифузноi складовоi для заданого профiлю dD/D(z) :
          TK(n)=0.
          do k=2,km+1
      TK(n)=TK(n)+(1-EW(k)**2)*TAU(k)*exp(-(TAU(k)*(eta-2*pi*DD(k)/dpl))**2/pi)
          end do
        TS(n)=2*(abs(sigmasp))**2*DL(2)*TK(n)
  */



R[n]=abs(TS[n]);

//Memo9->Lines->Add( "RozrachKogerTT km end пройшло");
}
if (RadioButton1->Checked==true)  R_dif_dl[i]=R[1];
if (RadioButton55->Checked==true) R_dif_dl[i]=R[1];
if (RadioButton2->Checked==true)  R_dif_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) R_dif_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
//Memo9->Lines->Add( "RozrachKogerTT R_cogerTT[i] end пройшло");
}

//for (int k=1; k<=km;k++)
//           Memo3-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(Dl[k])+'\t'+FloatToStr(TAU[k]));

  delete DDpd, x0i_a, eta0_a;
//Memo9->Lines->Add( "RozrachKogerTT 3 пройшло");



LhSum_max=0;
for (int k=1; k<=km;k++)
  {
//  LhSum[k]=LhD[k]+LhD_an[k]+Lhp[k]+Lhpd[k]+Lhptd[k];
  if (LhSum_max<LhSum[k]) LhSum_max=LhSum[k];
  Esum[k]=exp(-LhSum[k]);
  }
Emin=exp(-LhSum_max);
/*
if (fitting==0)
{
  if (CheckBox1->Checked==true)
  {
  LhD_max=0;
  for (int k=1; k<=km;k++) if (LhD_max<LhD[k]) LhD_max=LhD[k];
  ED_min=exp(-LhD_max);
  if (KDV_lich==1)
    {
    Edit247->Text=FloatToStr(LhD_max);
    Edit248->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==2)
    {
    Edit363->Text=FloatToStr(LhD_max);
    Edit362->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==3)
    {
    Edit365->Text=FloatToStr(LhD_max);
    Edit364->Text=FloatToStr(ED_min);
    }
  }
  if (CheckBox58->Checked==true)
  {
  LhD_max=0;
  for (int k=1; k<=km;k++) if (LhD_max<LhD_an[k]) LhD_max=LhD_an[k];
  ED_min=exp(-LhD_max);
  if (KDV_lich==1)
    {
    Edit270->Text=FloatToStr(LhD_max);
    Edit271->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==2)
    {
    Edit281->Text=FloatToStr(LhD_max);
    Edit283->Text=FloatToStr(ED_min);
    }
  if (KDV_lich==3)
    {
    Edit286->Text=FloatToStr(LhD_max);
    Edit287->Text=FloatToStr(ED_min);
    }
  }
  if (CheckBox2->Checked==true)
  {
  Lhp_max=0;
  for (int k=1; k<=km;k++) if (Lhp_max<Lhp[k]) Lhp_max=Lhp[k];
  Ep_min=exp(-Lhp_max);
  if (KDV_lich==1)
    {
    Edit246->Text=FloatToStr(Lhp_max);
    Edit249->Text=FloatToStr(Ep_min);
    }
  if (KDV_lich==2)
    {
    Edit366->Text=FloatToStr(Lhp_max);
    Edit367->Text=FloatToStr(Ep_min);
    }
  if (KDV_lich==3)
    {
    Edit368->Text=FloatToStr(Lhp_max);
    Edit369->Text=FloatToStr(Ep_min);
    }
  }
  if (CheckBox26->Checked==true)
  {
  Lhptd_max=0;
  for (int k=1; k<=km;k++) if (Lhptd_max<Lhptd[k]) Lhptd_max=Lhptd[k];
  Eptd_min=exp(-Lhptd_max);
  if (KDV_lich==1)
    {
    Edit254->Text=FloatToStr(Lhptd_max);
    Edit255->Text=FloatToStr(Eptd_min);
    }
  if (KDV_lich==2)
    {
    Edit370->Text=FloatToStr(Lhptd_max);
    Edit371->Text=FloatToStr(Eptd_min);
    }
  if (KDV_lich==3)
    {
    Edit372->Text=FloatToStr(Lhptd_max);
    Edit373->Text=FloatToStr(Eptd_min);
    }
  }
  if (CheckBox4->Checked==true)
  {
  Lhpd_max=0;
  for (int k=1; k<=km;k++) if (Lhpd_max<Lhpd[k]) Lhpd_max=Lhpd[k];
  Epd_min=exp(-Lhpd_max);
  if (KDV_lich==1)
    {
    Edit256->Text=FloatToStr(Lhpd_max);
    Edit257->Text=FloatToStr(Epd_min);
    }
  if (KDV_lich==2)
    {
    Edit374->Text=FloatToStr(Lhpd_max);
    Edit375->Text=FloatToStr(Epd_min);
    }
  if (KDV_lich==3)
    {
    Edit376->Text=FloatToStr(Lhpd_max);
    Edit377->Text=FloatToStr(Epd_min);
    }
  }
if (KDV_lich==1)
  {
  Edit17->Text=FloatToStr(LhSum_max);
  Edit18->Text=FloatToStr(Emin);
  }
if (KDV_lich==2)
  {
  Edit378->Text=FloatToStr(LhSum_max);
  Edit379->Text=FloatToStr(Emin);
  }
if (KDV_lich==3)
  {
  Edit380->Text=FloatToStr(LhSum_max);
  Edit381->Text=FloatToStr(Emin);
  }
} */

if (fitting==0 || vse==1) for (int k=1;k<=km;k++) Esum_[k]=Esum[k];

delete  LhSum;
}

//---------------------------------------------------------------------------


//---------------------------------------------------------------------------

void TForm1::DifuzSL_Loop(double R0_max, double nL_max, double *LhD,double ***MuDSj,double **FabsjD_dl, double *Rint_dl)
{   //Функція для розрахунку за дислокаційними петлями (профіль)
double R [3];
//double Mu [MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM],MuDSpr[MM];
  double *Mu, *pMut, *Jh1, *JhSW1,*JSW1, *J;
  Mu    = new double[KM];
  pMut  = new double[KM];
  Jh1   = new double[KM];
  JhSW1 = new double[KM];
  JSW1  = new double[KM];
  J     = new double[KM];
//double MuDSj[KM],Mu00j[KM],Fabsj[KM],R0[KM],nL[KM],EL[KM];
  double *Mu00j,*Fabsj,*R0,*nL,*EL;
  Mu00j  = new double[KM];
  Fabsj  = new double[KM];
  R0  = new double[KM];  //При заміні KM на km+1  повільно працює ????!!!!
  nL  = new double[KM];
  EL  = new double[KM];
double z,v,u,r;
double m0, B21, b2, b3, b4,Beta,k0,Ref1,Km1;
long double B11;
double MuLj;

for (int k=1; k<=km;k++)
{
  if (CheckBox6->Checked==true) nL[k]=nL_max*f[k];
    else  nL[k]=nL_max;
  if (CheckBox7->Checked==true) R0[k]=R0_max*f[k];
    else  R0[k]=R0_max;
  LhD[k]=koefLh*nL[k]*R0[k]*R0[k]*R0[k]*exp(1.5*LogN(M_E,(H*b)));
  EL[k]=exp(-LhD[k]);
}

for (int i=0; i<=m1_teor; i++)
{
Rint_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

//if (n==1) Memo9->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(z)+'\t'+FloatToStr(Jh1[k])+'\t'+FloatToStr(JhSW1[k])+'\t'+FloatToStr(JSW1[k])+'\t'+FloatToStr(m0)+'\t'+FloatToStr(Mu[k])+'\t'+FloatToStr(Km1)+'\t'+FloatToStr(k0)+'\t'+FloatToStr(B11)+'\t'+FloatToStr(B21)+'\t'+FloatToStr(b2)+'\t'+FloatToStr(b3)+'\t'+FloatToStr(b4));

for (int k=1; k<=km;k++)
{
//z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
z=(DeltaTeta[i]+DeltaTetaDD[k])*Sin2Teta/(C[n]*ModChiRH_a[k])*sqrt(b_as);
v=2*(z*g_a[n][k]/(EL[k]*EL[k])-p_a[n][k]);
//v=2*(z*g_a[n]/(EL[k]*EL[k])-p[n]);
u=(z*z-g_a[n][k]*g_a[n][k])/(EL[k]*EL[k])+Kapa_a[n][k]*Kapa_a[n][k]-1;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+r*EL[k]/fabs(g_a[n][k]));
pMut[k]=(1-exp(-2*Mu[k]*Dl[k]))/(2*Mu[k]*Dl[k]);
//Memo8->Lines->Add(FloatToStr(i)+'\t'+FloatToStr((1-exp(-2*Mu[1]*Dl[1]))/(2*Mu[1]*Dl[1])));
m0=(M_PI*VelKom/4.)*(H2Pi*ModChiRH_a[k]/Lambda)*(H2Pi*ModChiRH_a[k]/Lambda);
B11=(4/15.)*(M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2=B11+0.5*B21*CosTeta*CosTeta;
b3=B21*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4=B21*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDD[k]);
//k0=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]);
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
//if (i==30) Memo9->Lines->Add(FloatToStr(Km1)+'\t'+FloatToStr(R0[k])+'\t'+FloatToStr(EL[k])+'\t'+FloatToStr(H)+'\t'+FloatToStr(b)+'\t'+FloatToStr(m0)+'\t'+FloatToStr(VelKom)+'\t'+FloatToStr(Km1)+'\t'+FloatToStr(k0)+'\t'+FloatToStr(R0_max)+'\t'+FloatToStr(nL_max)+'\t'+FloatToStr(f[k])+'\t'+FloatToStr(b3)+'\t'+FloatToStr(b4));
if (fabs(k0)<=Km1)
{
Jh1[k]=b2*LogN(M_E,(Km1*Km1+Mu[k]*Mu[k])/(k0*k0+Mu[k]*Mu[k]))+(b3*k0*k0+b4*Mu[k]*Mu[k])*(1/(Km1*Km1+Mu[k]*Mu[k])-1/(k0*k0+Mu[k]*Mu[k]));//область хуаня
JhSW1[k]=(Km1*Km1/(Km1*Km1+Mu[k]*Mu[k]))*(b2-0.5*((b3*k0*k0+b4*Mu[k]*Mu[k])/(Km1*Km1+Mu[k]*Mu[k])));///область стокса вілсона
J[k]=Jh1[k]+JhSW1[k];
}
if (fabs(k0)>Km1)
{
JSW1[k]=(Km1*Km1/(k0*k0+Mu[k]*Mu[k]))*(b2-0.5*((b3*k0*k0+b4*Mu[k]*Mu[k])/(k0*k0+Mu[k]*Mu[k])));
J[k]=JSW1[k];
}

MuDSj[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*J[k];
Mu00j[k]=MuDSj[i][k][n]*pMut[k];
//Memo8->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(Mu0_a[k])+'\t'+FloatToStr(Mu[k])+'\t'+FloatToStr(MuDSj[i][k][n])+'\t'+FloatToStr(LhD[k])+'\t'+FloatToStr(EL[k]) );
   //Memo8->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(Mu0_a[k])+'\t'+FloatToStr(n)+'\t'+FloatToStr(MuDSj[i][k][n]) );
//Memo9->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(Jh1[k])+'\t'+FloatToStr(JhSW1[k])+'\t'+FloatToStr(JSW1[k])+'\t'+FloatToStr(m0)+'\t'+FloatToStr(Mu[k])+'\t'+FloatToStr(Km1)+'\t'+FloatToStr(k0)+'\t'+FloatToStr(B11)+'\t'+FloatToStr(B21)+'\t'+FloatToStr(b2)+'\t'+FloatToStr(b3)+'\t'+FloatToStr(b4));
   //if (i==30) Memo9->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(Jh1[k])+'\t'+FloatToStr(JhSW1[k])+'\t'+FloatToStr(JSW1[k])+'\t'+FloatToStr(m0)+'\t'+FloatToStr(Mu[k])+'\t'+FloatToStr(Km1)+'\t'+FloatToStr(k0)+'\t'+FloatToStr(B11)+'\t'+FloatToStr(B21)+'\t'+FloatToStr(b2)+'\t'+FloatToStr(b3)+'\t'+FloatToStr(b4));
}
for (int k=0; k<=km;k++)
  {
  Fabsj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    MuLj=(Mu0_a[jk]+MuDSj[i][jk][n])*(b_as+1)/(2*gamma0);
    Fabsj[k]=Fabsj[k]*exp(-MuLj*Dl[jk]);
    }
  }
FabsjD_dl[i][n]=Fabsj[0];

for (int k=1; k<=km;k++)
//R[n]= R[n]+Fabsj[k]*Mu00j[k]*Dl[k]/(gamma0)*(1+sin((2*Km1*Ref1)*sqrt(2*Km1*Ref1)));
//R[n]= R[n]+Fabsj[k]*MuDSj[i][k][n]*Dl[k]/(gamma0)*(1+sin((2*Km1*Ref1)*sqrt(2*Km1*Ref1)));
//R[n]= R[n]+Fabsj[k]*Mu00j[k]*Dl[k]/(gamma0);
R[n]= R[n]+Fabsj[k]*Mu00j[k]*Dl[k]/(gamma0);

//DeltaTeta1=(TetaMin+i*ik);
//if (n==1) Series35->AddXY(DeltaTeta1,R[1],"",clBlue);
//if (n==2) Series36->AddXY(DeltaTeta1,R[2],"",clBlack);

}
if (RadioButton1->Checked==true)  Rint_dl[i]=R[1];
if (RadioButton55->Checked==true) Rint_dl[i]=R[1];
if (RadioButton2->Checked==true)  Rint_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rint_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
//Memo9->Lines->Add( "DifuzSL_Loop пройшло");
  delete Mu, pMut , Jh1, JhSW1, JSW1, J;
  delete Mu00j, Fabsj,R0,nL,EL;
};

//---------------------------------------------------------------------------
void TForm1::DifuzSL_LoopAniz(double R0_max, double nL_max, double *LhD_an, double ***MuDSj_an, double **FabsjD_an_dl, double *Rint_an_dl)
{    ///Функція для розрахунку за дислокаційними петлями з урах. анізотропії(профіль)
double R [3], L_ext[3];
//double Mu [MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM],MuDSpr[MM];
  double *Mu, *pMut;
  Mu    = new double[KM];
  pMut  = new double[KM];
  double  *Jh1,*JSW1, *J,*JHss,*JHSWss,*JHhh,*JHSWhh, *JhSW1;
  Jh1   = new double[KM];
  JhSW1 = new double[KM];
  JSW1  = new double[KM];
  J     = new double[KM];
  JHss     = new double[3];
  JHSWss     = new double[5];
  JHhh     = new double[3];
  JHSWhh     = new double[5];
//double MuDSj[KM],Mu00j[KM],Fabsj[KM],R0[KM],nL[KM],EL[KM];
  double /* *MuDSj,*/ *Mu00j,*Fabsj,*R0,*nL,*EL;
//  MuDSj  = new double[KM];
  Mu00j  = new double[KM];
  Fabsj  = new double[KM];
  R0  = new double[KM];  //При заміні KM на km+1  повільно працює ????!!!!
  nL  = new double[KM];
  EL  = new double[KM];
double z,v,u,r;
double m0, B21, b2_, Beta,Kc1,Ref1,Km1,Koef,B11_,D;
long double B11;
double MuLj;
double k0,k0j,mu,LL,y,WW,K;

K=2*M_PI/Lambda;

for (int k=1; k<=km;k++)
{
if (CheckBox39->Checked==true) nL[k]=nL_max*f[k];
else  nL[k]=nL_max;
if (CheckBox61->Checked==true) R0[k]=R0_max*f[k];
else  R0[k]=R0_max;
LhD_an[k]=koefLh*nL[k]*R0[k]*R0[k]*R0[k]*exp(1.5*LogN(M_E,(H*b)));
EL[k]=exp(-LhD_an[k]);
}

for (int i=0; i<=m1_teor; i++)
{
Rint_an_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

for (int k=1; k<=km;k++)
{
//z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
z=(DeltaTeta[i]+DeltaTetaDD[k])*Sin2Teta/(C[n]*ModChiRH_a[k])*sqrt(b_as);
v=2*(z*g_a[n][k]/(EL[k]*EL[k])-p_a[n][k]);
//v=2*(z*g_a[n]/(EL[k]*EL[k])-p[n]);
u=(z*z-g_a[n][k]*g_a[n][k])/(EL[k]*EL[k])+Kapa_a[n][k]*Kapa_a[n][k]-1;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+r*EL[k]/fabs(g_a[n][k]));
pMut[k]=(1-exp(-2*Mu[k]*Dl[k]))/(2*Mu[k]*Dl[k]);
//Memo8->Lines->Add(FloatToStr(i)+'\t'+FloatToStr((1-exp(-2*Mu[1]*Dl[1]))/(2*Mu[1]*Dl[1])));
m0=(M_PI*VelKom/4.)*(H2Pi*ModChiRH_a[k]/Lambda)*(H2Pi*ModChiRH_a[k]/Lambda);

if(RadioButton48->Checked==true)     // Молодкін Дедерікс
{
B11=(4/15.)*(M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2_=B11+0.5*B21*CosTeta*CosTeta;
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2*M_PI/L_ext[n];///для петель
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
Koef=b2_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)));
MuDSj_an[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*Koef;
}

if(RadioButton49->Checked==true)     // Уляна Дедерікс
{
B11_=(M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2.*M_PI/L_ext[n];///для петель
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2.*M_PI/Ref1;///для петель
Koef=0.5*B11_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)))*D_loop;
MuDSj_an[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*Koef;
}

if(RadioButton63->Checked==true)     // Уляна Молодкін
{
k0j=k0=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDD[k]);
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
mu=Mu[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+r*EL[k]/fabs(g_a[n][k]));
LL=(M_PI*M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
WW=-H2Pi*H2Pi*(1.-sin(tb-(DeltaTeta[i]-DeltaTetaDD[k]))/sin(tb)/(2*C[n]*K*K*ModChiRH_a[k]*EL[k]));
//Memo10->Lines->Add("Poch");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(k)+'\t'+FloatToStr(n)+'\t'+FloatToStr(LL));
//Memo10->Lines->Add(FloatToStr(Km1)+'\t'+FloatToStr(k0j)+'\t'+FloatToStr(mu));
JHss[1]=0;
JHss[2]=0;
JHSWss[1]=0;
JHSWss[2]=0;
JHSWss[3]=0;
JHSWss[4]=0;
JHhh[1]=0;
JHhh[2]=0;
JHSWhh[1]=0;
JHSWhh[2]=0;
JHSWhh[3]=0;
JHSWhh[4]=0;
if (KDV_lich==1) if (fabs(k0)<=Km1)    //   (444)
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
JHss[ii] = LL*(1.610367869e-9*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.411591867e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+4.028754665e-11*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.014377332e-10*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.208626399e-9*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.678488360e-10*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.339244180e-9*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.403546508e-8*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.469551201e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.823183734e-10*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.403546508e-8*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-3.530287964e-10*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.469551201e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.841374335e-10*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/sqrt(k0j*k0j+mu*mu)
-0.1066935346*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.020337683*LogN(M_E,(k0j*k0j+y+mu*mu))
-0.8710312545*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.2360257074*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.2175022110*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.2420393372*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.266472077*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.9041015950e-1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6228371117*k0j*k0j/(k0j*k0j+y+mu*mu)
-1.786488457*mu*mu/(k0j*k0j+y+mu*mu)
-3.530287964e-10*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+9.662207216e-9*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+9.662207216e-9*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.680543553e-10*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.608326132e-9*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.608326132e-9*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.208626399e-9*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+6.774540279e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.258180093e-10*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.129090046e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+6.774540279e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.879847529e-9*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.170593168e-10*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+4.302355901e-9*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-2.879847529e-9*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.302355901e-9*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));
    }
Jh1[k]=JHss[2]-JHss[1];

 y=Km1*Km1-k0*k0;                //  (444)
JHSWss[3] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//y=infinity;            //  (444)
JHSWss[4] = LL*(
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));


JhSW1[k]=JHSWss[4]-JHSWss[3];
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

if (KDV_lich==1) if (fabs(k0)>Km1)     //   (444)
{
y=0;
//Memo10->Lines->Add("JHSWss[ii]");
JHSWss[1] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//y=infinity;             // (444)
JHSWss[2] = LL*(
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));

J[k]=JHSWss[2]-JHSWss[1];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }
//Memo10->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(Jh1[k])+'\t'+FloatToStr(JhSW1[k])+'\t'+FloatToStr(JSW1[k]));



if (KDV_lich==2) if (fabs(k0)<=Km1)    //  початок (888)
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
JHss[ii] = LL*(2.172706350e-8*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
+2.172706350e-8*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+6.883741810e-9*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+6.883741810e-9*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+5.883360794e-10*pow(k0j,5.)*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+2.941680397e-9*pow(k0j,5.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+1.765008238e-8*pow(k0j,5.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+1.765008238e-8*pow(k0j,5.)*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-3.649021572e-9*k0j*pow(mu,4.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
-2.189412943e-8*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-2.189412943e-8*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
+1.337140790e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+7.242354499e-10*k0j*pow(mu,6.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+3.621177250e-9*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+6.685703951e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+4.011422371e-8*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+4.011422371e-8*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-8.792025079e-10*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/sqrt(k0j*k0j+mu*mu)
+0.5970633902*log(k0j*k0j+y+mu*mu)
+1.375449835*k0j*k0j*mu*mu/pow((k0j*k0j+y+mu*mu),2.)
+0.3515374728*pow(k0j,4.)*mu*mu/pow((k0j*k0j+y+mu*mu),3.)
-0.3625395381*k0j*k0j*pow(mu,4.)/pow((k0j*k0j+y+mu*mu),3.)
-0.5224069516*k0j*k0j/(k0j*k0j+y+mu*mu)
-3.818913814*mu*mu/(k0j*k0j+y+mu*mu)
-0.5258252004*pow(k0j,4.)/pow((k0j*k0j+y+mu*mu),2.)
+2.405315597*pow(mu,4.)/pow((k0j*k0j+y+mu*mu),2.)
+0.2607561584*pow(k0j,6.)/pow((k0j*k0j+y+mu*mu),3.)
-0.4648846632*pow(mu,6.)/pow((k0j*k0j+y+mu*mu),3.)
+4.107100718e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.137493960e-9*pow(k0j,5.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
+4.107100718e-9*pow(k0j,3.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.656374259e-9*pow(k0j,3.)*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
-2.793824555e-8*pow(k0j,3.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-2.793824555e-8*pow(k0j,3.)*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
-7.370778260e-10*pow(k0j,7.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-6.824963758e-9*pow(k0j,5.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-6.824963758e-9*pow(k0j,5.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
-2.456926087e-11*pow(k0j,7.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
-1.228463043e-10*pow(k0j,7.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
-7.370778260e-10*pow(k0j,7.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu)));

    }
Jh1[k]=JHss[2]-JHss[1];

 y=Km1*Km1-k0*k0;                //  (888)
JHSWss[3] = LL*(1.029588139e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.2636531046*Km1*Km1*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.053550359e-9*Km1*Km1*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.2719046536*Km1*Km1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+4.412520596e-10*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.059176278e-9*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.842694565e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+0.9169665566*Km1*Km1*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-8.599241303e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.552124753e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.1955671188*Km1*Km1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2612034758*Km1*Km1*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.002855593e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.679992766e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.339996383e-8*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.299620652e-10*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.758405016e-9*Km1*Km1*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
-2.432681048e-9*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.216340524e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.104249506e-9*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.909456907*Km1*Km1*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.3486634974*Km1*Km1*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.603543732*Km1*Km1*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.3505501336*Km1*Km1*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.583293065e-10*Km1*Km1*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.791646532e-9*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.5970633902*Km1*Km1/(pow(k0j,2.)+y+pow(mu,2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.441870905e-9*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.431765875e-10*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.534824075e-9*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.267412037e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//y=infinity;            //  (888)
JHSWss[4] = LL*(
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));


JhSW1[k]=JHSWss[4]-JHSWss[3];
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

if (KDV_lich==2) if (fabs(k0)>Km1)     //   (888)
{
y=0;
//Memo10->Lines->Add("JHSWss[ii]");
JHSWss[1] = LL*(1.029588139e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.2636531046*Km1*Km1*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.053550359e-9*Km1*Km1*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.2719046536*Km1*Km1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+4.412520596e-10*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.059176278e-9*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.842694565e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+0.9169665566*Km1*Km1*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-8.599241303e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.552124753e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.1955671188*Km1*Km1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2612034758*Km1*Km1*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.002855593e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.679992766e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.339996383e-8*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.299620652e-10*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.758405016e-9*Km1*Km1*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
-2.432681048e-9*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.216340524e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.104249506e-9*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.909456907*Km1*Km1*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.3486634974*Km1*Km1*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.603543732*Km1*Km1*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.3505501336*Km1*Km1*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.583293065e-10*Km1*Km1*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.791646532e-9*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.5970633902*Km1*Km1/(pow(k0j,2.)+y+pow(mu,2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.441870905e-9*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.431765875e-10*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.534824075e-9*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.267412037e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//y=infinity;             // (888)
JHSWss[2] = LL*(
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

J[k]=JHSWss[2]-JHSWss[1];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }


if (KDV_lich==3) if (fabs(k0)<=Km1)   // для (880)  початок !!!!
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
//Memo10->Lines->Add(FloatToStr(pow(k0j,4.))+'\t'+FloatToStr(pow(mu,2.))+'\t'+FloatToStr(pow((pow(k0j,2.)+y+pow(mu,2.)),3.))+'\t'+FloatToStr(5555));

JHss[ii]= LL*(-0.1003223772*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+2.259615968*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-1.359957593e-8*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.4565870160*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)   
+3.740538411e-10*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.870269206e-9*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.677744575e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+4.677744575e-9*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))  
+1.122161523e-8*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.122161523e-8*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.359957593e-8*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.))) 
-2.266595988e-9*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))  
-6.333570612e-10*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/sqrt(pow(k0j,2.)+pow(mu,2.))  
+4.268683320e-8*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))  
-3.153037175*pow(mu,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+0.7786040683e-1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+1.422894440e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+7.114472200e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.256548304e-10*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.628274152e-9*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.576964491e-8*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.813880352e-8*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+5.164381221e-9*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.356467254e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.813880352e-8*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.5537198861*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-3.000124914e-8*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.1638651777*log(pow(k0j,2.)+y+pow(mu,2.))
+1.271293451e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.740733831*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-1.642389477e-8*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.000208190e-9*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-3.000124914e-8*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.061267923*pow(k0j,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+4.268683320e-8*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.164381221e-9*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+1.576964491e-8*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.737315795e-9*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.642389477e-8*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-0.2785251273*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)  );

//Memo10->Lines->Add("JHss[ii]end JHhh[ii]start ");
                                                    //  (880)
JHhh[ii]= LL*(-0.1027197499*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.7262938991*log(pow(k0j,2.)+y+pow(mu,2.))
+0.1216176877*pow(k0j,2.)/(pow(k0j,2.)+y+pow(mu,2.))
-1.068649806*pow(mu,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+0.7896675942*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+0.4064006075*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1701078364*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.1518005181*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2842049176e-1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.4283722467*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-5.055063235e-10*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/sqrt(pow(k0j,2.)+pow(mu,2.))
-7.318134490e-10*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-4.390880694e-9*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.390880694e-9*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.575578488e-10*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.287789244e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.556055694e-10*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.556055694e-10*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.923770500e-10*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.961885250e-9*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.177131150e-8*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.177131150e-8*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.102941940e-9*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+2.102941940e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+4.260092823e-11*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.990525003e-12*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-7.726735463e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.726735463e-9*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.328286762e-10*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+6.641433811e-10*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.984860287e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.984860287e-9*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.936578712e-11*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.161947227e-10*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.161947227e-10*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.971575010e-11*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+9.782526477e-10*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+9.782526477e-10*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.952625016e-12*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-5.971575010e-11*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.))));
//Memo10->Lines->Add("JHhh[ii]end ");

    }
//Jh1[k]=JHss[2]-JHss[1];
if(RadioButton64->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]))+0.5*WW/sqrt(1.+WW*WW)*((JHss[2]-JHss[1])-(JHhh[2]-JHhh[1]));
if(RadioButton65->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]))-0.5*WW/sqrt(1.+WW*WW)*((JHss[2]-JHss[1])-(JHhh[2]-JHhh[1]));
if(RadioButton66->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]));
if(RadioButton67->Checked==true)
  Jh1[k]=(JHss[2]-JHss[1]);

//goto nn10;
//Memo10->Lines->Add("JHSWss[3] ");

 y=Km1*Km1-k0*k0;                  // (880)
JHSWss[3]= LL*(1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.224763539e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.449527078e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.067170830e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-0.1638651777*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.490065270e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.980130540e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.942411228e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+1.839791906e-9*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.805403808e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+1.309188444e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+6.545942219e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))   
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.555319960e-9*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))             
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.))) 
+2.338872288e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))  
-1.266714122e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+1.506410646*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.7524178289e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.3424402620*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)       
+2.582190610e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+9.534700881e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+9.198959531e-9*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.511063992e-9*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.666736063e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.576518587*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)     
-0.2088938455*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-1.030633962*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.160489221*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.5839530512e-1*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.3691465907*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-1.824877196e-9*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.124385982e-9*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.333472127e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))   
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[3] ");
                                       //   (880)
JHSWhh[3]= LL*(-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+4.891263239e-10*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.931683866e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-9.014524707e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.420030941e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.942827875e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.840061882e-11*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+0.2131536882e-1*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+9.962150717e-11*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.649003668e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.324501834e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.373319675e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-6.866598375e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.291052475e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-4.507262353e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7703981241e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2855814978*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.051470970e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.011012647e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.6080884386e-1*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.5343249031*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1275808773*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.1138503886*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.5264450628*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2709337383*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.878756326e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.492893752e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-6.966837511e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.483418756e-11*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7262938991*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.439378163e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.455262373e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWss[4] ");

//y=infinity;                           //   (880)
JHSWss[4]= LL*(
1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[4] ");
                                      //  (880)
JHSWhh[4]= LL*(
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));


//JhSW1[k]=JHSWss[4]-JHSWss[3];
if(RadioButton64->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]))+0.5*WW/sqrt(1.+WW*WW)*((JHSWss[4]-JHSWss[3])-(JHSWhh[4]-JHSWhh[3]));
if(RadioButton65->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]))-0.5*WW/sqrt(1.+WW*WW)*((JHSWss[4]-JHSWss[3])-(JHSWhh[4]-JHSWhh[3]));
if(RadioButton66->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]));
if(RadioButton67->Checked==true)
  JhSW1[k]=(JHSWss[4]-JHSWss[3]);
//nn10:
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(212121)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

//goto nn11;

if (KDV_lich==3) if (fabs(k0)>Km1)    //   (880)
{
y=0;
//Memo10->Lines->Add("JHSWss[1]<");

JHSWss[1] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//Memo10->Lines->Add("JHSWhh[2]< ");
                                          //   (880)
JHSWhh[1]= LL*(-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+4.891263239e-10*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.931683866e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-9.014524707e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.420030941e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.942827875e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.840061882e-11*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+0.2131536882e-1*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+9.962150717e-11*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.649003668e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.324501834e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.373319675e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-6.866598375e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.291052475e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-4.507262353e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7703981241e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2855814978*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.051470970e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.011012647e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.6080884386e-1*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.5343249031*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1275808773*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.1138503886*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.5264450628*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2709337383*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.878756326e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.492893752e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-6.966837511e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.483418756e-11*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7262938991*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.439378163e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.455262373e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWss[2]< inf ");
                                           //   (880)
//y=infinity;
JHSWss[2]= LL*(
1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[2]< inf ");
                                            // (880)
JHSWhh[2]= LL*(
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//J[k]=JHSWss[2]-JHSWss[1];
if(RadioButton64->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]))+0.5*WW/sqrt(1.+WW*WW)*((JHSWss[2]-JHSWss[1])-(JHSWhh[2]-JHSWhh[1]));
if(RadioButton65->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]))-0.5*WW/sqrt(1.+WW*WW)*((JHSWss[2]-JHSWss[1])-(JHSWhh[2]-JHSWhh[1]));
if(RadioButton66->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]));
if(RadioButton67->Checked==true)
  J[k]=(JHSWss[2]-JHSWss[1]);
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }

//nn11:


MuDSj_an[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*J[k];
//Memo10->Lines->Add("MuDSj_an[i][k][n]");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(k)+'\t'+FloatToStr(n)+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(MuDSj_an[i][k][n]));
}

if(CheckBox66->Checked==true) pMut[k]=1;
Mu00j[k]=MuDSj_an[i][k][n]*pMut[k];
}
for (int k=0; k<=km;k++)
{
  Fabsj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    MuLj=(Mu0_a[jk]+MuDSj_an[i][jk][n])*(b_as+1)/(2*gamma0);
    Fabsj[k]=Fabsj[k]*exp(-MuLj*Dl[jk]);
    }
}
FabsjD_an_dl[i][n]=Fabsj[0];

for (int k=1; k<=km;k++)
//R[n]= R[n]+Fabsj[k]*MuDSj_an[i][k][n]*Dl[k]/(gamma0);
R[n]= R[n]+Fabsj[k]*Mu00j[k]*Dl[k]/(gamma0);

//DeltaTeta1=(TetaMin+i*ik);
//if (n==1) Series35->AddXY(DeltaTeta1,R[1],"",clBlue);;
//if (n==2) Series36->AddXY(DeltaTeta1,R[2],"",clBlack);
//if (CheckBox53->Checked==true)
//Memo8->Lines->Add(FloatToStr(R[i])+'\t'+FloatToStr(MuDSpr[i])+'\t'+FloatToStr(pMut[i])+'\t'+FloatToStr(Fabsj0*dl0/(gamma0)*(1.+sin((2.*Km1*Ref1)*sqrt(2.*Km1*Ref1))))+'\t'+FloatToStr(Koef)+'\t'+FloatToStr(L_ext[1])+'\t'+FloatToStr(m0));

}
//if (RadioButton1->Checked==true) Rint_an_dl[i]=R[1];
//if (RadioButton2->Checked==true) Rint_an_dl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rint_an_dl[i]=R[1];
if (RadioButton55->Checked==true) Rint_an_dl[i]=R[1];
if (RadioButton2->Checked==true)  Rint_an_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rint_an_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
//Memo10->Lines->Add("END ");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(Rint_an_dl[i])+'\t'+FloatToStr(444)+'\t'+FloatToStr(555)+'\t'+FloatToStr(666));
}
  delete  Jh1, JSW1, J,JHss,JHSWss,JHhh,JHSWhh, JhSW1;
  delete Mu,pMut,Mu00j;
  delete Fabsj,R0,nL,EL;
};

//---------------------------------------------------------------------------
void TForm1::DifuzSL_SferClaster(double R0p_max,double np_max,double eps,double *Lhp,double***MuDSPj,double**FabsjP_dl,double *RintP_dl)//функція для сферичних кластерів (профіль)
{
double R [3];
//double MuP[MM],pMutP[MM],Jh1P[MM],JhSW1P[MM],JSW1P[MM],JP[MM],Mu00P[MM],MuDSPpr[MM];        ,
  double *MuP, *pMutP, *Jh1P, *JhSW1P,*JSW1P, *JP;
  MuP    = new double[KM];
  pMutP  = new double[KM];
  Jh1P   = new double[KM];
  JhSW1P = new double[KM];
  JSW1P  = new double[KM];
  JP     = new double[KM];
//double MuDSPj[KM],Mu00Pj[KM],FabsPj[KM],R0p[KM],np[KM],EP[KM];
  double *Mu00Pj, *FabsPj,*R0p,*np,*EP;
  Mu00Pj  = new double[KM];
  FabsPj  = new double[KM];
  R0p = new double[KM];
  np  = new double[KM];
  EP  = new double[KM];
double zP, vP,uP,rP,AKl;
double m0P, B22, b2P, b3P, b4P,BetaP,k0P,Ref1P,Km1P;
long double B12;
double MuPj;
double Gama, n0,Alfa0,hh,Eta;

for (int k=1; k<=km;k++)
{
if (CheckBox8->Checked==true) np[k]=np_max*f[k];
else  np[k]=np_max;
if (CheckBox9->Checked==true) R0p[k]=R0p_max*f[k];
else  R0p[k]=R0p_max;
n0=(4/3.)*M_PI*R0p[k]*R0p[k]*R0p[k]/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Eta=Alfa0*exp((1./3.)*LogN(M_E,n0))*hh;
Lhp[k]=(np[k]*VelKom)*n0*exp((3/2.)*LogN(M_E,Eta));
EP[k]=exp(-Lhp[k]);
}

for (int i=0; i<=m1_teor; i++)
{
RintP_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

for (int k=1; k<=km;k++)
{
//zP=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
zP=(DeltaTeta[i]+DeltaTetaDD[k])*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
vP=2*(zP*g[n]/(EP[k]*EP[k])-p[n]);
uP=(zP*zP-g[n]*g[n])/(EP[k]*EP[k])+Kapa[n]*Kapa[n]-1;
rP=sqrt(0.5*(sqrt(uP*uP+vP*vP)-uP));
MuP[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+rP*EP[k]/fabs(g[n]));
pMutP[k]=(1-exp(-2*MuP[k]*Dl[k]))/(2*MuP[k]*Dl[k]);
m0P=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);
B12=0;
AKl=Gama*eps*R0p[k]*R0p[k]*R0p[k];
B22=(4*M_PI*AKl/VelKom)*(4*M_PI*AKl/VelKom);
b2P=B12+0.5*B22*CosTeta*CosTeta;
b3P=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4P=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0P=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDD[k]);
Ref1P=sqrt(H*fabs(AKl))*EP[k];///для сф. класт
Km1P=2*M_PI/(Ref1P);///для сф. класт
if (fabs(k0P)<=Km1P)
{
Jh1P[k]=b2P*LogN(M_E,(Km1P*Km1P+MuP[k]*MuP[k])/(k0P*k0P+MuP[k]*MuP[k]))+(b3P*k0P*k0P+b4P*MuP[k]*MuP[k])*(1/(Km1P*Km1P+MuP[k]*MuP[k])-1/(k0P*k0P+MuP[k]*MuP[k]));//область хуаня
JhSW1P[k]=(Km1P*Km1P/(Km1P*Km1P+MuP[k]*MuP[k]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[k]*MuP[k])/(Km1P*Km1P+MuP[k]*MuP[k])));///область стокса вілсона
JP[k]=Jh1P[k]+JhSW1P[k];
}
if (fabs(k0P)>Km1P)
{
JSW1P[k]=(Km1P*Km1P/(k0P*k0P+MuP[k]*MuP[k]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[k]*MuP[k])/(k0P*k0P+MuP[k]*MuP[k])));
JP[k]=JSW1P[k];
}

MuDSPj[i][k][n]=(np[k]*VelKom)*C[n]*C[n]*EP[k]*EP[k]*m0P*JP[k];
Mu00Pj[k]=MuDSPj[i][k][n]*pMutP[k];
}
for (int k=0; k<=km;k++)
  {
  FabsPj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    MuPj=(Mu0_a[jk]+MuDSPj[i][jk][n])*(b_as+1)/(2*gamma0);
    FabsPj[k]=FabsPj[k]*exp(-MuPj*Dl[jk]);
    }
  }
FabsjP_dl[i][n]=FabsPj[0];
for (int k=1; k<=km;k++)
R[n]= R[n]+FabsPj[k]*Mu00Pj[k]*Dl[k]/(gamma0);
//R[n]= R[n]+FabsPj[k]*MuDSPj[i][k][n]*Dl[k]/(gamma0);
}
//if (RadioButton1->Checked==true) RintP_dl[i]=R[1];
//if (RadioButton2->Checked==true) RintP_dl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  RintP_dl[i]=R[1];
if (RadioButton55->Checked==true) RintP_dl[i]=R[1];
if (RadioButton2->Checked==true)  RintP_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) RintP_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete MuP,pMutP, Jh1P, JhSW1P, JSW1P, JP;
  delete Mu00Pj, FabsPj,R0p,np,EP;
};

//---------------------------------------------------------------------------
void TForm1::DiduzSL_DiscClaster(double R0d_max, double nd_max, double epsd,double *Lhpd,double ***MuDSdj, double **FabsjPd_dl,double *Rintd_dl)//функція для дискових кластерів (профіль)
{
double R [3];
//double Mud [MM],pMutd[MM],Jh1d[MM],JhSW1d[MM],JSW1d[MM],Jd[MM],Mu00d[MM],MuDSdpr[MM];
  double *Mud, *pMutd, *Jh1d, *JhSW1d,*JSW1d, *Jd;
  Mud    = new double[KM];
  pMutd  = new double[KM];
  Jh1d   = new double[KM];
  JhSW1d = new double[KM];
  JSW1d  = new double[KM];
  Jd     = new double[KM];
//double MuDSdj[KM],Mu00dj[KM],Fabsdj[KM],R0d[KM],nd[KM],Ed[KM];
  double   *Mu00dj, *Fabsdj,*R0d,*nd,*Ed;
  Mu00dj  = new double[KM];
  Fabsdj  = new double[KM];
  R0d = new double[KM];
  nd  = new double[KM];
  Ed  = new double[KM];
double zd, vd,ud,rd,AKld;
double m0d, B22, b2d, b3d, b4d,Betad,k0d,Ref1d,Km1d;
long double B12;
double Mudj;
double Gama, hp,n0d,Alfa0,hh,Etad;

for (int k=1; k<=km;k++)
{
if (CheckBox10->Checked==true) nd[k]=nd_max*f[k];
else  nd[k]=nd_max;
if (CheckBox11->Checked==true) R0d[k]=R0d_max*f[k];
else  R0d[k]=R0d_max;
hp=3.96*R0d[k]*exp(0.5966*log((0.89e-8/R0d[k])));                     //дані для кремнію
if (fitting==0) Edit76->Text=FloatToStr(hp*1e8);
n0d=M_PI*R0d[k]*R0d[k]*hp/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*epsd*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Etad=Alfa0*exp((1/3.)*LogN(M_E,n0d))*hh;
Lhpd[k]=(nd[k]*VelKom)*n0d*exp((3/2.)*LogN(M_E,Etad));
Ed[k]=exp(-Lhpd[k]);
}
for (int i=0; i<=m1_teor; i++)
{
Rintd_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

for (int k=1; k<=km;k++)
{
//zd=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
zd=(DeltaTeta[i]+DeltaTetaDD[k])*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
vd=2*(zd*g[n]/(Ed[k]*Ed[k])-p[n]);
ud=(zd*zd-g[n]*g[n])/(Ed[k]*Ed[k])+Kapa[n]*Kapa[n]-1;
rd=sqrt(0.5*(sqrt(ud*ud+vd*vd)-ud));
Mud[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+rd*Ed[k]/fabs(g[n]));
pMutd[k]=(1-exp(-2*Mud[k]*Dl[k]))/(2*Mud[k]*Dl[k]);
m0d=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);
AKld=3*Gama*epsd*R0d[k]*R0d[k]*hp/4.;
B12=(4*M_PI*AKld/VelKom)*(4*M_PI*AKld/VelKom);
B22=(4*M_PI*AKld/VelKom)*(4*M_PI*AKld/VelKom);
b2d=B12+0.5*B22*CosTeta*CosTeta;
b3d=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4d=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0d=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDD[k]);
Ref1d=sqrt(H*fabs(AKld))*Ed[k];///для диск. класт
Km1d=2*M_PI/(Ref1d);///для диск. класт
if (fabs(k0d)<=Km1d)
{
Jh1d[k]=b2d*LogN(M_E,(Km1d*Km1d+Mud[k]*Mud[k])/(k0d*k0d+Mud[k]*Mud[k]))+(b3d*k0d*k0d+b4d*Mud[k]*Mud[k])*(1/(Km1d*Km1d+Mud[k]*Mud[k])-1/(k0d*k0d+Mud[k]*Mud[k]));//область хуаня
JhSW1d[k]=(Km1d*Km1d/(Km1d*Km1d+Mud[k]*Mud[k]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[k]*Mud[k])/(Km1d*Km1d+Mud[k]*Mud[k])));///область стокса вілсона
Jd[k]=Jh1d[k]+JhSW1d[k];
}
if (fabs(k0d)>Km1d)
{
JSW1d[k]=(Km1d*Km1d/(k0d*k0d+Mud[k]*Mud[k]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[k]*Mud[k])/(k0d*k0d+Mud[k]*Mud[k])));
Jd[k]=JSW1d[k];
}
MuDSdj[i][k][n]=(nd[k]*VelKom)*C[n]*C[n]*Ed[k]*Ed[k]*m0d*Jd[k];
Mu00dj[k]=MuDSdj[i][k][n]*pMutd[k];
}
for (int k=0; k<=km;k++)
{
  Fabsdj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    Mudj=(Mu0_a[jk]+MuDSdj[i][jk][n])*(b_as+1)/(2*gamma0);
    Fabsdj[k]=Fabsdj[k]*exp(-Mudj*Dl[jk]);
    }
}
FabsjPd_dl[i][n]=Fabsdj[0];
for (int k=1; k<=km;k++)
//R[n]= R[n]+Fabsdj[k]*MuDSdj[i][k][n]*Dl[k]/(gamma0);
R[n]= R[n]+Fabsdj[k]*Mu00dj[k]*Dl[k]/(gamma0);
}
//if (RadioButton1->Checked==true) Rintd_dl[i]=R[1];
//if (RadioButton2->Checked==true) Rintd_dl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintd_dl[i]=R[1];
if (RadioButton55->Checked==true) Rintd_dl[i]=R[1];
if (RadioButton2->Checked==true)  Rintd_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintd_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mud, pMutd, Jh1d, JhSW1d, JSW1d, Jd;
  delete Mu00dj, Fabsdj,R0d,nd,Ed;
};

/////////////////////////////////////
//---------------------------------------------------------------------------
void TForm1::DiduzSL_DiscClasterAniz(double R0d_max, double nd_max, double epsd,double *Lhpd,double ***MuDSdj, double **FabsjPd_dl,double *Rintd_dl)//функція для дискових кластерів (anizotrop) (профіль)
//void TForm1::DifuzSL_LoopAniz(double R0_max, double nL_max, double *LhD_an, double ***MuDSj_an, double **FabsjD_an_dl, double *Rint_an_dl)
{    ///Функція для розрахунку за дислокаційними петлями з урах. анізотропії(профіль)
double R [3], L_ext[3];
//double Mu [MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM],MuDSpr[MM];
  double *Mu, *pMut;
  Mu    = new double[KM];
  pMut  = new double[KM];
  double  *Jh1,*JSW1, *J,*JHss,*JHSWss,*JHhh,*JHSWhh, *JhSW1;
  Jh1   = new double[KM];
  JhSW1 = new double[KM];
  JSW1  = new double[KM];
  J     = new double[KM];
  JHss     = new double[3];
  JHSWss     = new double[5];
  JHhh     = new double[3];
  JHSWhh     = new double[5];
//double MuDSj[KM],Mu00j[KM],Fabsj[KM],R0[KM],nL[KM],EL[KM];
  double /* *MuDSj,*/ *Mu00dj,*Fabsdj,*R0d,*nd,*Ed;
//  MuDSj  = new double[KM];
  Mu00dj  = new double[KM];
  Fabsdj  = new double[KM];
  R0d  = new double[KM];  //При заміні KM на km+1  повільно працює ????!!!!
  nd  = new double[KM];
  Ed  = new double[KM];
double z,v,u,r;
double m0, B21, b2_, Beta,Kc1,Ref1,Km1,Koef,B11_,D;
long double B11;
double Mudj;
double Gama, hp,n0d,Alfa0,hh,Etad,         Dd;
double k0,k0j,mu,LL,y,WW,K;

K=2*M_PI/Lambda;

for (int k=1; k<=km;k++)
{
if (CheckBox10->Checked==true) nd[k]=nd_max*f[k];
else  nd[k]=nd_max;
if (CheckBox11->Checked==true) R0d[k]=R0d_max*f[k];
else  R0d[k]=R0d_max;
//LhD_an[k]=koefLh*nL[k]*R0[k]*R0[k]*R0[k]*exp(1.5*LogN(M_E,(H*b)));
//EL[k]=exp(-LhD_an[k]);
hp=3.96*R0d[k]*exp(0.5966*log((0.89e-8/R0d[k])));                     //???? ??? ???????
if (fitting==0) Edit76->Text=FloatToStr(hp*1e8);
n0d=M_PI*R0d[k]*R0d[k]*hp/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*epsd*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Etad=Alfa0*exp((1/3.)*LogN(M_E,n0d))*hh;
if (CheckBox86->Checked==false) Lhpd[k]=(nd[k]*VelKom)*n0d*exp((3/2.)*LogN(M_E,Etad));
if (CheckBox86->Checked==true)
{
      Dd=b*epsd;
  if (KDV_lich==1) koefLh=StrToFloat(Edit220->Text);
  if (KDV_lich==2) koefLh=StrToFloat(Edit220->Text);
  if (KDV_lich==3) koefLh=StrToFloat(Edit394->Text);
 Lhpd[k]=koefLh*nd[k]*R0d[k]*R0d[k]*R0d[k]*exp(1.5*LogN(M_E,(H*Dd)));
}
Ed[k]=exp(-Lhpd[k]);
}

for (int i=0; i<=m1_teor; i++)
{
Rintd_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

for (int k=1; k<=km;k++)
{
//z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
z=(DeltaTeta[i]+DeltaTetaDD[k])*Sin2Teta/(C[n]*ModChiRH_a[k])*sqrt(b_as);
v=2*(z*g_a[n][k]/(Ed[k]*Ed[k])-p_a[n][k]);
//v=2*(z*g_a[n]/(EL[k]*EL[k])-p[n]);
u=(z*z-g_a[n][k]*g_a[n][k])/(Ed[k]*Ed[k])+Kapa_a[n][k]*Kapa_a[n][k]-1;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+r*Ed[k]/fabs(g_a[n][k]));
pMut[k]=(1-exp(-2*Mu[k]*Dl[k]))/(2*Mu[k]*Dl[k]);
//Memo8->Lines->Add(FloatToStr(i)+'\t'+FloatToStr((1-exp(-2*Mu[1]*Dl[1]))/(2*Mu[1]*Dl[1])));
m0=(M_PI*VelKom/4.)*(H2Pi*ModChiRH_a[k]/Lambda)*(H2Pi*ModChiRH_a[k]/Lambda);
/*
if(RadioButton48->Checked==true)     // Молодкін Дедерікс
{
B11=(4/15.)*(M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2_=B11+0.5*B21*CosTeta*CosTeta;
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2*M_PI/L_ext[n];///для петель
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
Koef=b2_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)));
MuDSj_an[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*Koef;
}

if(RadioButton49->Checked==true)     // Уляна Дедерікс
{
B11_=(M_PI*b*R0[k]*R0[k]/VelKom)*(M_PI*b*R0[k]*R0[k]/VelKom);
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2.*M_PI/L_ext[n];///для петель
Ref1=R0[k]*EL[k]*sqrt(H*b);///для петель
Km1=2.*M_PI/Ref1;///для петель
Koef=0.5*B11_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)))*D_loop;
MuDSj_an[i][k][n]=(nL[k]*VelKom)*C[n]*C[n]*EL[k]*EL[k]*m0*Koef;
}
  */
if(RadioButton63->Checked==true)     // Уляна Молодкін
{
k0j=k0=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDD[k]);
Ref1=R0d[k]*Ed[k]*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
mu=Mu[k]=(0.5*Mu0_a[k]/gamma0)*(1+b_as)/2.*(1+r*Ed[k]/fabs(g_a[n][k]));
LL=(M_PI*M_PI*b*R0d[k]*R0d[k]/VelKom)*(M_PI*b*R0d[k]*R0d[k]/VelKom);
WW=-H2Pi*H2Pi*(1.-sin(tb-(DeltaTeta[i]-DeltaTetaDD[k]))/sin(tb)/(2*C[n]*K*K*ModChiRH_a[k]*Ed[k]));
//Memo10->Lines->Add("Poch");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(k)+'\t'+FloatToStr(n)+'\t'+FloatToStr(LL));
//Memo10->Lines->Add(FloatToStr(Km1)+'\t'+FloatToStr(k0j)+'\t'+FloatToStr(mu));
JHss[1]=0;
JHss[2]=0;
JHSWss[1]=0;
JHSWss[2]=0;
JHSWss[3]=0;
JHSWss[4]=0;
JHhh[1]=0;
JHhh[2]=0;
JHSWhh[1]=0;
JHSWhh[2]=0;
JHSWhh[3]=0;
JHSWhh[4]=0;
if (KDV_lich==1) if (fabs(k0)<=Km1)    //   (444)
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
JHss[ii] = LL*(1.610367869e-9*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.411591867e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+4.028754665e-11*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.014377332e-10*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.208626399e-9*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.678488360e-10*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.339244180e-9*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.403546508e-8*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.469551201e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.823183734e-10*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.403546508e-8*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-3.530287964e-10*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.469551201e-9*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.841374335e-10*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/sqrt(k0j*k0j+mu*mu)
-0.1066935346*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.020337683*LogN(M_E,(k0j*k0j+y+mu*mu))
-0.8710312545*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.2360257074*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.2175022110*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.2420393372*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.266472077*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.9041015950e-1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6228371117*k0j*k0j/(k0j*k0j+y+mu*mu)
-1.786488457*mu*mu/(k0j*k0j+y+mu*mu)
-3.530287964e-10*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+9.662207216e-9*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+9.662207216e-9*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.680543553e-10*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.608326132e-9*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.608326132e-9*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.208626399e-9*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+6.774540279e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.258180093e-10*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.129090046e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+6.774540279e-9*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.879847529e-9*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.170593168e-10*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+4.302355901e-9*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-2.879847529e-9*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.302355901e-9*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));
    }
Jh1[k]=JHss[2]-JHss[1];

 y=Km1*Km1-k0*k0;                //  (444)
JHSWss[3] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//y=infinity;            //  (444)
JHSWss[4] = LL*(
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));


JhSW1[k]=JHSWss[4]-JHSWss[3];
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

if (KDV_lich==1) if (fabs(k0)>Km1)     //   (444)
{
y=0;
//Memo10->Lines->Add("JHSWss[ii]");
JHSWss[1] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//y=infinity;             // (444)
JHSWss[2] = LL*(
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*M_PI/2./((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu)));

J[k]=JHSWss[2]-JHSWss[1];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }
//Memo10->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(Jh1[k])+'\t'+FloatToStr(JhSW1[k])+'\t'+FloatToStr(JSW1[k]));



if (KDV_lich==2) if (fabs(k0)<=Km1)    //  початок (888)
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
JHss[ii] = LL*(2.172706350e-8*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
+2.172706350e-8*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+6.883741810e-9*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+6.883741810e-9*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+5.883360794e-10*pow(k0j,5.)*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+2.941680397e-9*pow(k0j,5.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+1.765008238e-8*pow(k0j,5.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+1.765008238e-8*pow(k0j,5.)*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-3.649021572e-9*k0j*pow(mu,4.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
-2.189412943e-8*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-2.189412943e-8*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
+1.337140790e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+7.242354499e-10*k0j*pow(mu,6.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
+3.621177250e-9*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+6.685703951e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
+4.011422371e-8*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu))
+4.011422371e-8*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-8.792025079e-10*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/sqrt(k0j*k0j+mu*mu)
+0.5970633902*log(k0j*k0j+y+mu*mu)
+1.375449835*k0j*k0j*mu*mu/pow((k0j*k0j+y+mu*mu),2.)
+0.3515374728*pow(k0j,4.)*mu*mu/pow((k0j*k0j+y+mu*mu),3.)
-0.3625395381*k0j*k0j*pow(mu,4.)/pow((k0j*k0j+y+mu*mu),3.)
-0.5224069516*k0j*k0j/(k0j*k0j+y+mu*mu)
-3.818913814*mu*mu/(k0j*k0j+y+mu*mu)
-0.5258252004*pow(k0j,4.)/pow((k0j*k0j+y+mu*mu),2.)
+2.405315597*pow(mu,4.)/pow((k0j*k0j+y+mu*mu),2.)
+0.2607561584*pow(k0j,6.)/pow((k0j*k0j+y+mu*mu),3.)
-0.4648846632*pow(mu,6.)/pow((k0j*k0j+y+mu*mu),3.)
+4.107100718e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.137493960e-9*pow(k0j,5.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
+4.107100718e-9*pow(k0j,3.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.656374259e-9*pow(k0j,3.)*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),2.))
-2.793824555e-8*pow(k0j,3.)*mu*mu*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-2.793824555e-8*pow(k0j,3.)*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
-7.370778260e-10*pow(k0j,7.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),3.)*sqrt(k0j*k0j+mu*mu))
-6.824963758e-9*pow(k0j,5.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*(k0j*k0j+y+mu*mu))
-6.824963758e-9*pow(k0j,5.)*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/(pow((4.*k0j*k0j+4.*mu*mu),2.)*sqrt(k0j*k0j+mu*mu))
-2.456926087e-11*pow(k0j,7.)*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*pow((k0j*k0j+y+mu*mu),3.))
-1.228463043e-10*pow(k0j,7.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),2.)*pow((k0j*k0j+y+mu*mu),2.))
-7.370778260e-10*pow(k0j,7.)*sqrt(y)/(pow((4.*k0j*k0j+4.*mu*mu),3.)*(k0j*k0j+y+mu*mu)));

    }
Jh1[k]=JHss[2]-JHss[1];

 y=Km1*Km1-k0*k0;                //  (888)
JHSWss[3] = LL*(1.029588139e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.2636531046*Km1*Km1*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.053550359e-9*Km1*Km1*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.2719046536*Km1*Km1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+4.412520596e-10*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.059176278e-9*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.842694565e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+0.9169665566*Km1*Km1*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-8.599241303e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.552124753e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.1955671188*Km1*Km1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2612034758*Km1*Km1*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.002855593e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.679992766e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.339996383e-8*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.299620652e-10*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.758405016e-9*Km1*Km1*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
-2.432681048e-9*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.216340524e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.104249506e-9*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.909456907*Km1*Km1*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.3486634974*Km1*Km1*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.603543732*Km1*Km1*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.3505501336*Km1*Km1*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.583293065e-10*Km1*Km1*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.791646532e-9*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.5970633902*Km1*Km1/(pow(k0j,2.)+y+pow(mu,2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.441870905e-9*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.431765875e-10*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.534824075e-9*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.267412037e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//y=infinity;            //  (888)
JHSWss[4] = LL*(
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));


JhSW1[k]=JHSWss[4]-JHSWss[3];
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

if (KDV_lich==2) if (fabs(k0)>Km1)     //   (888)
{
y=0;
//Memo10->Lines->Add("JHSWss[ii]");
JHSWss[1] = LL*(1.029588139e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.2636531046*Km1*Km1*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.053550359e-9*Km1*Km1*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.2719046536*Km1*Km1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+4.412520596e-10*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.059176278e-9*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.842694565e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+0.9169665566*Km1*Km1*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-8.599241303e-11*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.552124753e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+0.1955671188*Km1*Km1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2612034758*Km1*Km1*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.002855593e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.679992766e-9*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.339996383e-8*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.299620652e-10*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.758405016e-9*Km1*Km1*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
-2.432681048e-9*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.216340524e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.104249506e-9*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.909456907*Km1*Km1*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.3486634974*Km1*Km1*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+1.603543732*Km1*Km1*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.3505501336*Km1*Km1*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.583293065e-10*Km1*Km1*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.791646532e-9*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.5970633902*Km1*Km1/(pow(k0j,2.)+y+pow(mu,2.))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.441870905e-9*Km1*Km1*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.431765875e-10*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+2.534824075e-9*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.267412037e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//y=infinity;             // (888)
JHSWss[2] = LL*(
+1.232130215e-8*Km1*Km1*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.312748517e-8*Km1*Km1*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.177528834e-8*Km1*Km1*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.274987919e-8*Km1*Km1*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.758405016e-9*Km1*Km1*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.298043144e-8*Km1*Km1*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.579772391e-9*Km1*Km1*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.065122543e-8*Km1*Km1*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403997830e-7*Km1*Km1*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+7.604472224e-8*Km1*Km1*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

J[k]=JHSWss[2]-JHSWss[1];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }


if (KDV_lich==3) if (fabs(k0)<=Km1)   // для (880)  початок !!!!
  {
  for (int ii=1; ii<=2; ii++)
    {
    if (ii==1) y=0;
    if (ii==2) y=Km1*Km1-k0*k0;
//Memo10->Lines->Add("JHss[ii]");
//Memo10->Lines->Add(FloatToStr(1111)+'\t'+FloatToStr(i)+'\t'+FloatToStr(ii)+'\t'+FloatToStr(y));
//Memo10->Lines->Add(FloatToStr(pow(k0j,4.))+'\t'+FloatToStr(pow(mu,2.))+'\t'+FloatToStr(pow((pow(k0j,2.)+y+pow(mu,2.)),3.))+'\t'+FloatToStr(5555));

JHss[ii]= LL*(-0.1003223772*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+2.259615968*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-1.359957593e-8*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-0.4565870160*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)   
+3.740538411e-10*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.870269206e-9*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.677744575e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+4.677744575e-9*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))  
+1.122161523e-8*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.122161523e-8*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.359957593e-8*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.))) 
-2.266595988e-9*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))  
-6.333570612e-10*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/sqrt(pow(k0j,2.)+pow(mu,2.))  
+4.268683320e-8*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))  
-3.153037175*pow(mu,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+0.7786040683e-1*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+1.422894440e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+7.114472200e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+5.256548304e-10*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.628274152e-9*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.576964491e-8*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.813880352e-8*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+5.164381221e-9*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.356467254e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.813880352e-8*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.5537198861*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-3.000124914e-8*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.1638651777*log(pow(k0j,2.)+y+pow(mu,2.))
+1.271293451e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.740733831*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-1.642389477e-8*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.000208190e-9*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-3.000124914e-8*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.061267923*pow(k0j,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+4.268683320e-8*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.164381221e-9*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+1.576964491e-8*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.737315795e-9*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.642389477e-8*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-0.2785251273*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)  );

//Memo10->Lines->Add("JHss[ii]end JHhh[ii]start ");
                                                    //  (880)
JHhh[ii]= LL*(-0.1027197499*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.7262938991*log(pow(k0j,2.)+y+pow(mu,2.))
+0.1216176877*pow(k0j,2.)/(pow(k0j,2.)+y+pow(mu,2.))
-1.068649806*pow(mu,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+0.7896675942*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+0.4064006075*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1701078364*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.1518005181*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2842049176e-1*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.4283722467*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-5.055063235e-10*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/sqrt(pow(k0j,2.)+pow(mu,2.))
-7.318134490e-10*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-4.390880694e-9*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.390880694e-9*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-2.575578488e-10*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.287789244e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.556055694e-10*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.556055694e-10*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.923770500e-10*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.961885250e-9*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.177131150e-8*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.177131150e-8*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.102941940e-9*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+2.102941940e-9*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+4.260092823e-11*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.990525003e-12*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-7.726735463e-9*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-7.726735463e-9*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.328286762e-10*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+6.641433811e-10*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.984860287e-9*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.984860287e-9*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.936578712e-11*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.161947227e-10*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.161947227e-10*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.971575010e-11*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+9.782526477e-10*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+9.782526477e-10*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-9.952625016e-12*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-5.971575010e-11*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.))));
//Memo10->Lines->Add("JHhh[ii]end ");

    }
//Jh1[k]=JHss[2]-JHss[1];
if(RadioButton64->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]))+0.5*WW/sqrt(1.+WW*WW)*((JHss[2]-JHss[1])-(JHhh[2]-JHhh[1]));
if(RadioButton65->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]))-0.5*WW/sqrt(1.+WW*WW)*((JHss[2]-JHss[1])-(JHhh[2]-JHhh[1]));
if(RadioButton66->Checked==true)
  Jh1[k]=0.5*((JHss[2]-JHss[1])+(JHhh[2]-JHhh[1]));
if(RadioButton67->Checked==true)
  Jh1[k]=(JHss[2]-JHss[1]);

//goto nn10;
//Memo10->Lines->Add("JHSWss[3] ");

 y=Km1*Km1-k0*k0;                  // (880)
JHSWss[3]= LL*(1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.224763539e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.449527078e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.067170830e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-0.1638651777*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+2.490065270e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+4.980130540e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.942411228e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+1.839791906e-9*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.805403808e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+1.309188444e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+6.545942219e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))   
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-7.555319960e-9*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))             
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.))) 
+2.338872288e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))  
-1.266714122e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+1.506410646*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-0.7524178289e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.3424402620*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)       
+2.582190610e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+9.534700881e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+9.198959531e-9*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.511063992e-9*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.666736063e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.576518587*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)     
-0.2088938455*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-1.030633962*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
+1.160489221*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.5839530512e-1*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.3691465907*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-1.824877196e-9*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-9.124385982e-9*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-3.333472127e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))   
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[3] ");
                                       //   (880)
JHSWhh[3]= LL*(-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+4.891263239e-10*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.931683866e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-9.014524707e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.420030941e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.942827875e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.840061882e-11*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+0.2131536882e-1*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+9.962150717e-11*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.649003668e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.324501834e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.373319675e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-6.866598375e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.291052475e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-4.507262353e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7703981241e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2855814978*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.051470970e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.011012647e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.6080884386e-1*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.5343249031*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1275808773*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.1138503886*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.5264450628*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2709337383*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.878756326e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.492893752e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-6.966837511e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.483418756e-11*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7262938991*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.439378163e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.455262373e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWss[4] ");

//y=infinity;                           //   (880)
JHSWss[4]= LL*(
1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[4] ");
                                      //  (880)
JHSWhh[4]= LL*(
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));


//JhSW1[k]=JHSWss[4]-JHSWss[3];
if(RadioButton64->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]))+0.5*WW/sqrt(1.+WW*WW)*((JHSWss[4]-JHSWss[3])-(JHSWhh[4]-JHSWhh[3]));
if(RadioButton65->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]))-0.5*WW/sqrt(1.+WW*WW)*((JHSWss[4]-JHSWss[3])-(JHSWhh[4]-JHSWhh[3]));
if(RadioButton66->Checked==true)
  JhSW1[k]=0.5*((JHSWss[4]-JHSWss[3])+(JHSWhh[4]-JHSWhh[3]));
if(RadioButton67->Checked==true)
  JhSW1[k]=(JHSWss[4]-JHSWss[3]);
//nn10:
J[k]=Jh1[k]+JhSW1[k];
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(212121)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHss[1])+'\t'+FloatToStr(JHss[2])+'\t'+FloatToStr(JHss[3])+'\t'+FloatToStr(JHss[4]));
  }

//goto nn11;

if (KDV_lich==3) if (fabs(k0)>Km1)    //   (880)
{
y=0;
//Memo10->Lines->Add("JHSWss[1]<");

JHSWss[1] = LL*(0.3114185559*Km1*Km1*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1631266582*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.073578580e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.637470926e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+5.367892898e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8932442287*Km1*Km1*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.1815295029*Km1*Km1*mu*mu*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.6027343967e-1*Km1*Km1*k0j*k0j*k0j*k0j/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.8443147178*Km1*Km1*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.187354630e-9*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+3.220735739e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-3.508866270e-10*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.912412778e-8*Km1*Km1*k0j*k0j*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.439923764e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-5.361087106e-9*Km1*Km1*k0j*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-2.117387800e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-9.881143068e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-4.940571534e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.020337683*Km1*Km1/(k0j*k0j+y+mu*mu)
+1.693635070e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.903630325e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+2.371089098e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+3.951815163e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.5806875030*Km1*Km1*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-0.8002015094e-1*Km1*Km1*k0j*k0j*k0j*k0j*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+0.1770192805*Km1*Km1*k0j*k0j*mu*mu*mu*mu/((k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+3.021565999e-11*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+4.780395445e-10*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-8.639542586e-9*Km1*Km1*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+7.682748671e-10*Km1*Km1*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+2.390197723e-9*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
+1.434118634e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.787029035e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-8.935145177e-10*Km1*Km1*k0j*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+1.410064133e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
+7.050320663e-10*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-1.765143982e-10*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)*(k0j*k0j+y+mu*mu))
-2.964342920e-8*Km1*Km1*k0j*k0j*k0j*k0j*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu))
-1.059086389e-9*Km1*Km1*k0j*mu*mu*atan(sqrt(y)/sqrt(k0j*k0j+mu*mu))/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*sqrt(k0j*k0j+mu*mu))
+4.230192398e-9*Km1*Km1*k0j*mu*mu*mu*mu*mu*mu*sqrt(y)/((4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(4.*k0j*k0j+4.*mu*mu)*(k0j*k0j+y+mu*mu)));

//Memo10->Lines->Add("JHSWhh[2]< ");
                                          //   (880)
JHSWhh[1]= LL*(-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+4.891263239e-10*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.931683866e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-9.014524707e-10*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.420030941e-10*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.942827875e-10*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.840061882e-11*pow(Km1,2.)*pow(k0j,5.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
+0.2131536882e-1*pow(Km1,2.)*pow(k0j,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+9.962150717e-11*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
+4.649003668e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
+2.324501834e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-1.373319675e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-6.866598375e-9*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
+1.291052475e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-4.507262353e-9*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7703981241e-1*pow(Km1,2.)*pow(k0j,4.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.2855814978*pow(Km1,2.)*pow(k0j,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*(pow(k0j,2.)+y+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*(pow(k0j,2.)+y+pow(mu,2.)))
+1.051470970e-9*pow(Km1,2.)*pow(k0j,3.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.011012647e-9*pow(Km1,2.)*k0j*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*(pow(k0j,2.)+y+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+0.6080884386e-1*pow(Km1,2.)*pow(k0j,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.5343249031*pow(Km1,2.)*pow(mu,2.)/pow((pow(k0j,2.)+y+pow(mu,2.)),2.)
-0.1275808773*pow(Km1,2.)*pow(k0j,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
-0.1138503886*pow(Km1,2.)*pow(mu,6.)/pow((pow(k0j,2.)+y+pow(mu,2.)),4.)
+0.5264450628*pow(Km1,2.)*pow(mu,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
+0.2709337383*pow(Km1,2.)*pow(k0j,4.)/pow((pow(k0j,2.)+y+pow(mu,2.)),3.)
-4.878756326e-10*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-1.492893752e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/((4.*pow(k0j,2.)+4.*pow(mu,2.))*pow((pow(k0j,2.)+y+pow(mu,2.)),4.))
-6.966837511e-12*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),3.))
-3.483418756e-11*pow(Km1,2.)*k0j*pow(mu,6.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-0.7262938991*pow(Km1,2.)/(pow(k0j,2.)+y+pow(mu,2.))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.439378163e-9*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*(pow(k0j,2.)+y+pow(mu,2.)))
+6.455262373e-11*pow(Km1,2.)*k0j*pow(mu,4.)*sqrt(y)/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*pow((pow(k0j,2.)+y+pow(mu,2.)),2.))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*atan(sqrt(y)/sqrt(pow(k0j,2.)+pow(mu,2.)))/(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWss[2]< inf ");
                                           //   (880)
//y=infinity;
JHSWss[2]= LL*(
1.334858123e-7*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.494039162e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+3.927565332e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.533191976e-8*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.403323373e-8*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.000041638e-7*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.549314366e-8*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.266714122e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
-5.474631589e-8*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+5.519375719e-8*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//Memo10->Lines->Add("JHSWhh[2]< inf ");
                                            // (880)
JHSWhh[2]= LL*(
+3.873157424e-10*pow(Km1,2.)*k0j*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+2.934757943e-9*pow(Km1,2.)*k0j*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.090051253e-10*pow(Km1,2.)*k0j*pow(mu,6.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-4.119959025e-8*pow(Km1,2.)*pow(k0j,7.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+8.520185647e-10*pow(Km1,2.)*pow(k0j,5.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.011012647e-9*pow(Km1,2.)*k0j*M_PI/2./((4.*pow(k0j,2.)+4.*pow(mu,2.))*sqrt(pow(k0j,2.)+pow(mu,2.)))
+6.308825821e-9*pow(Km1,2.)*pow(k0j,3.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),2.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
+1.394701100e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,4.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-2.704357412e-8*pow(Km1,2.)*pow(k0j,5.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),4.)*sqrt(pow(k0j,2.)+pow(mu,2.)))
-1.463626898e-8*pow(Km1,2.)*pow(k0j,3.)*pow(mu,2.)*M_PI/2./(pow((4.*pow(k0j,2.)+4.*pow(mu,2.)),3.)*sqrt(pow(k0j,2.)+pow(mu,2.))));

//J[k]=JHSWss[2]-JHSWss[1];
if(RadioButton64->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]))+0.5*WW/sqrt(1.+WW*WW)*((JHSWss[2]-JHSWss[1])-(JHSWhh[2]-JHSWhh[1]));
if(RadioButton65->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]))-0.5*WW/sqrt(1.+WW*WW)*((JHSWss[2]-JHSWss[1])-(JHSWhh[2]-JHSWhh[1]));
if(RadioButton66->Checked==true)
  J[k]=0.5*((JHSWss[2]-JHSWss[1])+(JHSWhh[2]-JHSWhh[1]));
if(RadioButton67->Checked==true)
  J[k]=(JHSWss[2]-JHSWss[1]);
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(y)+'\t'+FloatToStr(JHSWss[1])+'\t'+FloatToStr(JHSWss[2]));
  }

//nn11:


MuDSdj[i][k][n]=(nd[k]*VelKom)*C[n]*C[n]*Ed[k]*Ed[k]*m0*J[k];
//Memo10->Lines->Add("MuDSj_an[i][k][n]");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(k)+'\t'+FloatToStr(n)+'\t'+FloatToStr(J[k])+'\t'+FloatToStr(MuDSj_an[i][k][n]));
}

if(CheckBox66->Checked==true) pMut[k]=1;
Mu00dj[k]=MuDSdj[i][k][n]*pMut[k];
}
for (int k=0; k<=km;k++)
{
  Fabsdj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    Mudj=(Mu0_a[jk]+MuDSdj[i][jk][n])*(b_as+1)/(2*gamma0);
    Fabsdj[k]=Fabsdj[k]*exp(-Mudj*Dl[jk]);
    }
}
FabsjPd_dl[i][n]=Fabsdj[0];

for (int k=1; k<=km;k++)
//R[n]= R[n]+Fabsj[k]*MuDSj_an[i][k][n]*Dl[k]/(gamma0);
R[n]= R[n]+Fabsdj[k]*Mu00dj[k]*Dl[k]/(gamma0);

//DeltaTeta1=(TetaMin+i*ik);
//if (n==1) Series35->AddXY(DeltaTeta1,R[1],"",clBlue);;
//if (n==2) Series36->AddXY(DeltaTeta1,R[2],"",clBlack);
//if (CheckBox53->Checked==true)
//Memo8->Lines->Add(FloatToStr(R[i])+'\t'+FloatToStr(MuDSpr[i])+'\t'+FloatToStr(pMut[i])+'\t'+FloatToStr(Fabsj0*dl0/(gamma0)*(1.+sin((2.*Km1*Ref1)*sqrt(2.*Km1*Ref1))))+'\t'+FloatToStr(Koef)+'\t'+FloatToStr(L_ext[1])+'\t'+FloatToStr(m0));

}
//if (RadioButton1->Checked==true) Rint_an_dl[i]=R[1];
//if (RadioButton2->Checked==true) Rint_an_dl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintd_dl[i]=R[1];
if (RadioButton55->Checked==true) Rintd_dl[i]=R[1];
if (RadioButton2->Checked==true)  Rintd_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintd_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
//Memo10->Lines->Add("END ");
//Memo10->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(Rint_an_dl[i])+'\t'+FloatToStr(444)+'\t'+FloatToStr(555)+'\t'+FloatToStr(666));
}
  delete  Jh1, JSW1, J,JHss,JHSWss,JHhh,JHSWhh, JhSW1;
  delete Mu,pMut,Mu00dj;
  delete Fabsdj,R0d,nd,Ed;
};



//---------------------------------------------------------------------------
void TForm1::DifuzSL_PynByshKato(double R0p_max,double np_max,double eps,double *Lhp,double***MuDSPj,double**FabsjP_dl,double *RintP_dl)//функція для сферичних кластерів (профіль)
{
double R [3];
//double MuP[MM],pMutP[MM],Jh1P[MM],JhSW1P[MM],JSW1P[MM],JP[MM],Mu00P[MM],MuDSPpr[MM];        ,
  double *MuP, *pMutP, *Jh1P, *JhSW1P,*JSW1P, *JP;
  MuP    = new double[KM];
  pMutP  = new double[KM];
  Jh1P   = new double[KM];
  JhSW1P = new double[KM];
  JSW1P  = new double[KM];
  JP     = new double[KM];
//double MuDSPj[KM],Mu00Pj[KM],FabsPj[KM],R0p[KM],np[KM],EP[KM];
  double *Mu00Pj, *FabsPj,*R0p,*np,*EP;
  Mu00Pj  = new double[KM];
  FabsPj  = new double[KM];
  R0p = new double[KM];
  np  = new double[KM];
  EP  = new double[KM];
double zP, vP,uP,rP,AKl;
double m0P, B22, b2P, b3P, b4P,BetaP,k0P,Ref1P,Km1P;
long double B12;
double MuPj;
double Gama, n0,Alfa0,hh,Eta;

for (int k=1; k<=km;k++)
{
if (CheckBox8->Checked==true) np[k]=np_max*f[k];
else  np[k]=np_max;
if (CheckBox9->Checked==true) R0p[k]=R0p_max*f[k];
else  R0p[k]=R0p_max;
n0=(4/3.)*M_PI*R0p[k]*R0p[k]*R0p[k]/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Eta=Alfa0*exp((1./3.)*LogN(M_E,n0))*hh;
Lhp[k]=(np[k]*VelKom)*n0*exp((3/2.)*LogN(M_E,Eta));
EP[k]=exp(-Lhp[k]);
}

for (int i=0; i<=m1_teor; i++)
{
RintP_dl[i]=0;

for (int n=nC1; n<=nC; n++)
{
R[n]=0;

for (int k=1; k<=km;k++)
  {
//  TS(n)=TS(n)+dl*(1-Esum[k]**2)*TAU[k]*
//  exp(-(TAU[k]*(REAL(eta)+2*M_PI*DD[k]/dpl)*(TAU[k]*(REAL(eta)+2*M_PI*DD[k]/dpl))/M_PI);


}
/*for (int k=0; k<=km;k++)
  {
  FabsPj[k]=1;
  for (int jk=k+1; jk<=km;jk++)
    {
    MuPj=(Mu0_a[jk]+MuDSPj[i][jk][n])*(b_as+1)/(2*gamma0);
    FabsPj[k]=FabsPj[k]*exp(-MuPj*Dl[jk]);
    }
  }  */
FabsjP_dl[i][n]=FabsPj[0];
for (int k=1; k<=km;k++)
R[n]= R[n]+FabsPj[k]*Mu00Pj[k]*Dl[k]/(gamma0);
//R[n]= R[n]+FabsPj[k]*MuDSPj[i][k][n]*Dl[k]/(gamma0);
}
//if (RadioButton1->Checked==true) RintP_dl[i]=R[1];
//if (RadioButton2->Checked==true) RintP_dl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  RintP_dl[i]=R[1];
if (RadioButton55->Checked==true) RintP_dl[i]=R[1];
if (RadioButton2->Checked==true)  RintP_dl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) RintP_dl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete MuP,pMutP, Jh1P, JhSW1P, JSW1P, JP;
  delete Mu00Pj, FabsPj,R0p,np,EP;
};

//---------------------------------------------------------------------------
void TForm1::Difuz0pl_Loop(double R00pl,double nL0pl,double &LhD0pl,double *MuDSpl,double **FabsjD_pl,double *Rintpl)
{    //Функція для розрахунку за дислокаційними петлями (ід. частина плівки)
double R [3];
//double Mu[MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM];
  double *Mu, *pMut, *Jh1, *JhSW1,*JSW1, *J, *Mu00;
  Mu    = new double[m1_teor+1];
  pMut  = new double[m1_teor+1];
  Jh1   = new double[m1_teor+1];
  JhSW1 = new double[m1_teor+1];
  JSW1  = new double[m1_teor+1];
  J     = new double[m1_teor+1];
  Mu00  = new double[m1_teor+1];
double z,v,u,r;
double m0, B21, b2, b3, b4,Beta,k0,Ref1,Km1;
long double B11;
double  Fabsjpl;
//  MuDSjpl  = new double[2];    // бо плівка одна
  //Mu00jpl  = new double[2];    // бо плівка одна
//  Fabsjpl  = new double[2];    // бо плівка одна
double MuLj;
double EL0pl;

LhD0pl=koefLh*(nL0pl)*R00pl*R00pl*R00pl*exp(1.5*LogN(M_E,(H*b)));
EL0pl=exp(-LhD0pl);

for (int i=0; i<=m1_teor; i++)
{
Rintpl[i]=0;

for (int n=nC1; n<=nC; n++)
{
z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRHpl)*sqrt(b_as);
v=2*(z*g_pl[n]/(EL0pl*EL0pl)-p_pl[n]);
u=(z*z-g_pl[n]*g_pl[n])/(EL0pl*EL0pl)+Kapa_pl[n]*Kapa_pl[n]-1.;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[i]=(0.5*Mu0_pl/gamma0)*(1+b_as)/2.*(1+r*EL0pl/fabs(g_pl[n]));
pMut[i]=(1-exp(-2*Mu[i]*hpl0))/(2*Mu[i]*hpl0);
m0=(M_PI*VelKompl/4.)*(H2Pi*ModChiRHpl/Lambda)*(H2Pi*ModChiRHpl/Lambda);
B11=(4/15.)*(M_PI*b*R00pl*R00pl/VelKompl)*(M_PI*b*R00pl*R00pl/VelKompl);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2=B11+0.5*B21*CosTeta*CosTeta;
b3=B21*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4=B21*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDDpl);
Ref1=R00pl*EL0pl*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
if (fabs(k0)<=Km1)
{
Jh1[i]=b2*LogN(M_E,(Km1*Km1+Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i]))+(b3*k0*k0+b4*Mu[i]*Mu[i])*(1/(Km1*Km1+Mu[i]*Mu[i])-1/(k0*k0+Mu[i]*Mu[i]));//область хуаня
JhSW1[i]=(Km1*Km1/(Km1*Km1+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(Km1*Km1+Mu[i]*Mu[i])));///область стокса вілсона
J[i]=Jh1[i]+JhSW1[i];
}
if (fabs(k0)>Km1)
{
JSW1[i]=(Km1*Km1/(k0*k0+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i])));
J[i]=JSW1[i];
}

MuDSpl[i]=(nL0pl*VelKompl)*C[n]*C[n]*EL0pl*EL0pl*m0*J[i];
Mu00[i]=MuDSpl[i]*pMut[i];
Fabsjpl=Fabsj_SL[i][n];

MuLj=(Mu0_pl+MuDSpl[i])*(b_as+1)/(2*gamma0);
FabsjD_pl[i][n]=  Fabsj_SL[i][n]*exp(-MuLj*hpl0);

R[n]= Fabsjpl*Mu00[i]*hpl0/(gamma0);
}
//if (RadioButton1->Checked==true) Rintpl[i]=R[1];
//if (RadioButton2->Checked==true) Rintpl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintpl[i]=R[1];
if (RadioButton55->Checked==true) Rintpl[i]=R[1];
if (RadioButton2->Checked==true)  Rintpl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintpl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mu, pMut, Jh1, JhSW1, JSW1, J, Mu00;
};

//---------------------------------------------------------------------------
/*void TForm1::Difuz0pl_Loop(double R00pl,double nL0pl,double &LhD0pl,double *MuDSpl,double *Rintpl) ///Функція для розрахунку по дислокаційним петлям 1 (ід. частина плівки)
{
double R [3];
//double Mu[MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM];
  double *Mu, *pMut, *Jh1, *JhSW1,*JSW1, *J, *Mu00;
  Mu    = new double[m1_teor+1];
  pMut  = new double[m1_teor+1];
  Jh1   = new double[m1_teor+1];
  JhSW1 = new double[m1_teor+1];
  JSW1  = new double[m1_teor+1];
  J     = new double[m1_teor+1];
  Mu00  = new double[m1_teor+1];
double z,v,u,r;
double m0, B21, b2, b3, b4,Beta,k0,Ref1,Km1;
long double B11;
double *MuDSjpl, *Mu00jpl, *Fabsjpl;
  MuDSjpl  = new double[2];    // бо плівка одна
  Mu00jpl  = new double[2];    // бо плівка одна
  Fabsjpl  = new double[2];    // бо плівка одна
double MuLj;
double EL0pl;

LhD0pl=koefLh*(nL0pl)*R00pl*R00pl*R00pl*exp(1.5*LogN(M_E,(H*b)));
EL0pl=exp(-LhD0pl);

for (int i=0; i<=m1_teor; i++)
{
Rintpl[i]=0.;

for (int n=nC1; n<=nC; n++)
{
z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRHpl)*sqrt(b_as);
v=2.*(z*g_pl[n]/(EL0pl*EL0pl)-p_pl[n]);
u=(z*z-g_pl[n]*g_pl[n])/(EL0pl*EL0pl)+Kapa_pl[n]*Kapa_pl[n]-1.;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[i]=(0.5*Mu0_pl/gamma0)*(1.+b_as)/2.*(1.+r*EL0pl/fabs(g_pl[n]));
pMut[i]=(1.-exp(-2.*Mu[i]*hpl0))/(2.*Mu[i]*hpl0);
m0=(M_PI*VelKompl/4.)*(H2Pi*ModChiRHpl/Lambda)*(H2Pi*ModChiRHpl/Lambda);
B11=(4./15.)*(M_PI*b*R00pl*R00pl/VelKompl)*(M_PI*b*R00pl*R00pl/VelKompl);
Beta=0.25*(3.*Nu*Nu+6.*Nu-1)/((1.-Nu)*(1.-Nu));
B21=Beta*B11;
b2=B11+0.5*B21*CosTeta*CosTeta;
b3=B21*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4=B21*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0=(2.*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDDpl);
Ref1=R00pl*EL0pl*sqrt(H*b);///для петель
Km1=2.*M_PI/Ref1;///для петель
if (fabs(k0)<=Km1)
{
Jh1[i]=b2*LogN(M_E,(Km1*Km1+Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i]))+(b3*k0*k0+b4*Mu[i]*Mu[i])*(1./(Km1*Km1+Mu[i]*Mu[i])-1./(k0*k0+Mu[i]*Mu[i]));//область хуаня
JhSW1[i]=(Km1*Km1/(Km1*Km1+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(Km1*Km1+Mu[i]*Mu[i])));///область стокса вілсона
J[i]=Jh1[i]+JhSW1[i];
}
if (fabs(k0)>Km1)
{
JSW1[i]=(Km1*Km1/(k0*k0+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i])));
J[i]=JSW1[i];
}

MuDSpl[i]=(nL0pl*VelKompl)*C[n]*C[n]*EL0pl*EL0pl*m0*J[i];
Fabsjpl=Fabsj_SL[i][n];

MuLj=(Mu0_pl+MuDSpl[i])*(b_as+1)/(2*gamma0);
FabsjD_pl[i][n]=  Fabsj_SL[i][n]*exp(-MuLj*hpl0);

R[n]= Fabsjpl*Mu00[i]*hpl0/(gamma0);
}
//if (RadioButton1->Checked==true) Rintpl[i]=R[1];
//if (RadioButton2->Checked==true) Rintpl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintpl[i]=R[1];
if (RadioButton55->Checked==true) Rintpl[i]=R[1];
if (RadioButton2->Checked==true)  Rintpl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintpl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mu, pMut, Jh1, JhSW1, JSW1, J, Mu00;
  delete MuDSjpl, Mu00jpl, Fabsjpl;
};        */

//---------------------------------------------------------------------------
void TForm1::Difuz0pl_SferClaster(double R0p0pl,double np0pl,double eps0pl,double &Lhp0pl,double *MuDSPpl,double **FabsjP_pl,double *RintPpl)//функція для сферичних кластерів 1 (ід. частина плівки)
{
double R [3];
//double MuP [MM],pMutP[MM],Jh1P[MM],JhSW1P[MM],JSW1P[MM],JP[MM],Mu00P[MM];
  double *MuP, *pMutP, *Jh1P, *JhSW1P,*JSW1P, *JP, *Mu00P;
  MuP    = new double[m1_teor+1];
  pMutP  = new double[m1_teor+1];
  Jh1P   = new double[m1_teor+1];
  JhSW1P = new double[m1_teor+1];
  JSW1P  = new double[m1_teor+1];
  JP     = new double[m1_teor+1];
  Mu00P  = new double[m1_teor+1];
double zP, vP,uP,rP,AKl;
double m0P, B22, b2P, b3P, b4P,BetaP,k0P,Ref1P,Km1P;
long double B12;
double FabsPjpl;
//  MuDSPjpl  = new double[2];    // бо плівка одна
//  Mu00Pjpl  = new double[2];    // бо плівка одна
//  FabsPjpl  = new double[2];    // бо плівка одна
double MuLj;
double EL0pl;

double Gama, hh,Alfa0,EP0pl;
double n00,Eta0;

n00=(4/3.)*M_PI*R0p0pl*R0p0pl*R0p0pl/VelKompl;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps0pl*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*apl/M_PI;
Eta0=Alfa0*exp((1/3.)*LogN(M_E,n00))*hh;
Lhp0pl=(np0pl*VelKompl)*n00*exp((3/2.)*LogN(M_E,Eta0));
EP0pl=exp(-Lhp0pl);

for (int i=0; i<=m1_teor; i++)
{
RintPpl[i]=0;

for (int n=nC1; n<=nC; n++)
{
zP=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRHpl)*sqrt(b_as);
vP=2*(zP*g_pl[n]/(EP0pl*EP0pl)-p_pl[n]);
uP=(zP*zP-g_pl[n]*g_pl[n])/(EP0pl*EP0pl)+Kapa_pl[n]*Kapa_pl[n]-1;
rP=sqrt(0.5*(sqrt(uP*uP+vP*vP)-uP));
MuP[i]=(0.5*Mu0_pl/gamma0)*(1+b_as)/2.*(1+rP*EP0pl/fabs(g_pl[n]));
pMutP[i]=(1-exp(-2*MuP[i]*hpl0))/(2*MuP[i]*hpl0);
m0P=(M_PI*VelKompl/4.)*(H2Pi*ModChiRHpl/Lambda)*(H2Pi*ModChiRHpl/Lambda);
B12=0;
AKl=Gama*eps0pl*R0p0pl*R0p0pl*R0p0pl;
B22=(4*M_PI*AKl/VelKompl)*(4*M_PI*AKl/VelKompl);
b2P=B12+0.5*B22*CosTeta*CosTeta;
b3P=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4P=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0P=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDDpl);
Ref1P=sqrt(H*fabs(AKl))*EP0pl;///для сф. класт
Km1P=2*M_PI/(Ref1P);///для сф. класт
if (fabs(k0P)<=Km1P)
{
Jh1P[i]=b2P*LogN(M_E,(Km1P*Km1P+MuP[i]*MuP[i])/(k0P*k0P+MuP[i]*MuP[i]))+(b3P*k0P*k0P+b4P*MuP[i]*MuP[i])*(1/(Km1P*Km1P+MuP[i]*MuP[i])-1/(k0P*k0P+MuP[i]*MuP[i]));//область хуаня
JhSW1P[i]=(Km1P*Km1P/(Km1P*Km1P+MuP[i]*MuP[i]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[i]*MuP[i])/(Km1P*Km1P+MuP[i]*MuP[i])));///область стокса вілсона
JP[i]=Jh1P[i]+JhSW1P[i];//+JhSWoscP[i];
}
if (fabs(k0P)>Km1P)
{
JSW1P[i]=(Km1P*Km1P/(k0P*k0P+MuP[i]*MuP[i]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[i]*MuP[i])/(k0P*k0P+MuP[i]*MuP[i])));
JP[i]=JSW1P[i];
}

MuDSPpl[i]=(np0pl*VelKompl)*C[n]*C[n]*EP0pl*EP0pl*m0P*JP[i];
Mu00P[i]=MuDSPpl[i]*pMutP[i];
FabsPjpl=Fabsj_SL[i][n];

    MuLj=(Mu0_pl+MuDSPpl[i])*(b_as+1)/(2*gamma0);
    FabsjP_pl[i][n]=Fabsj_SL[i][n]*exp(-MuLj*hpl0);

R[n]= FabsPjpl*Mu00P[i]*hpl0/(gamma0);
}
//if (RadioButton1->Checked==true) RintPpl[i]=R[1];
//if (RadioButton2->Checked==true) RintPpl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  RintPpl[i]=R[1];
if (RadioButton55->Checked==true) RintPpl[i]=R[1];
if (RadioButton2->Checked==true)  RintPpl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) RintPpl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete MuP, pMutP, Jh1P, JhSW1P, JSW1P, JP, Mu00P;
};

//---------------------------------------------------------------------------
void TForm1::Diduz0pl_DiscClaster(double R0d0pl, double nd0pl,double eps0dpl,double &Lhpd0pl, double *MuDSdpl,double **FabsjPd_pl,double *Rintdpl)//функція для дискових кластерів (ід. частина плівки)
{
double R [3];
//double Mud [MM],pMutd[MM],Jh1d[MM],JhSW1d[MM],JSW1d[MM],Jd[MM],Mu00d[MM];
  double *Mud, *pMutd, *Jh1d, *JhSW1d,*JSW1d, *Jd, *Mu00d;
  Mud    = new double[m1_teor+1];
  pMutd  = new double[m1_teor+1];
  Jh1d   = new double[m1_teor+1];
  JhSW1d = new double[m1_teor+1];
  JSW1d  = new double[m1_teor+1];
  Jd     = new double[m1_teor+1];
  Mu00d  = new double[m1_teor+1];
double zd,vd,ud,rd,AKld;
double m0d, B22, b2d, b3d, b4d,Betad,k0d,Ref1d,Km1d;
long double B12;
double FabsPdjpl;
//  MuDSPdjpl  = new double[2];    // бо плівка одна
//  Mu00Pdjpl  = new double[2];    // бо плівка одна
//  FabsPdjpl  = new double[2];    // бо плівка одна
double MuLj;
double Gama, hp0, hh,Alfa0,Ed0pl;
double n00,Eta0;

hp0=3.96*R0d0pl*exp(0.5966*log((0.89e-8/R0d0pl)));
if (fitting==0) Edit77->Text=FloatToStr(hp0*1e8);
n00=M_PI*R0d0pl*R0d0pl*hp0/VelKompl;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps0dpl*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*apl/M_PI;
Eta0=Alfa0*exp((1/3.)*LogN(M_E,n00))*hh;
Lhpd0pl=(nd0pl*VelKompl)*n00*exp((3/2.)*LogN(M_E,Eta0));
Ed0pl=exp(-Lhpd0pl);

for (int i=0; i<=m1_teor; i++)
{
Rintdpl[i]=0;

for (int n=nC1; n<=nC; n++)
{
zd=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRHpl)*sqrt(b_as);
vd=2*(zd*g_pl[n]/(Ed0pl*Ed0pl)-p_pl[n]);
ud=(zd*zd-g_pl[n]*g_pl[n])/(Ed0pl*Ed0pl)+Kapa_pl[n]*Kapa_pl[n]-1;
rd=sqrt(0.5*(sqrt(ud*ud+vd*vd)-ud));
Mud[i]=(0.5*Mu0_pl/gamma0)*(1+b_as)/2.*(1+rd*Ed0pl/fabs(g_pl[n]));
pMutd[i]=(1-exp(-2*Mud[i]*hpl0))/(2*Mud[i]*hpl0);
m0d=(M_PI*VelKompl/4.)*(H2Pi*ModChiRHpl/Lambda)*(H2Pi*ModChiRHpl/Lambda);
AKld=3*Gama*eps0dpl*R0d0pl*R0d0pl*hp0/4.;
B12=(4*M_PI*AKld/VelKompl)*(4*M_PI*AKld/VelKompl);
B22=(4*M_PI*AKld/VelKompl)*(4*M_PI*AKld/VelKompl);
b2d=B12+0.5*B22*CosTeta*CosTeta;
b3d=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4d=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0d=(2*M_PI/Lambda)*Sin2Teta*(DeltaTeta[i]-DeltaTetaDDpl);
Ref1d=sqrt(H*fabs(AKld))*Ed0pl;///для диск. класт
Km1d=2*M_PI/(Ref1d);///для диск. класт
if (fabs(k0d)<=Km1d)
{
Jh1d[i]=b2d*LogN(M_E,(Km1d*Km1d+Mud[i]*Mud[i])/(k0d*k0d+Mud[i]*Mud[i]))+(b3d*k0d*k0d+b4d*Mud[i]*Mud[i])*(1/(Km1d*Km1d+Mud[i]*Mud[i])-1/(k0d*k0d+Mud[i]*Mud[i]));//область хуаня
JhSW1d[i]=(Km1d*Km1d/(Km1d*Km1d+Mud[i]*Mud[i]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[i]*Mud[i])/(Km1d*Km1d+Mud[i]*Mud[i])));///область стокса вілсона
Jd[i]=Jh1d[i]+JhSW1d[i];
}
if (fabs(k0d)>Km1d)
{
JSW1d[i]=(Km1d*Km1d/(k0d*k0d+Mud[i]*Mud[i]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[i]*Mud[i])/(k0d*k0d+Mud[i]*Mud[i])));
Jd[i]=JSW1d[i];
}

MuDSdpl[i]=(nd0pl*VelKompl)*C[n]*C[n]*Ed0pl*Ed0pl*m0d*Jd[i];
Mu00d[i]=MuDSdpl[i]*pMutd[i];
FabsPdjpl=Fabsj_SL[i][n];

    MuLj=(Mu0_pl+MuDSdpl[i])*(b_as+1)/(2*gamma0);
    FabsjPd_pl[i][n]=Fabsj_SL[i][n]*exp(-MuLj*hpl0);

R[n]=  FabsPdjpl*Mu00d[i]*hpl0/(gamma0);
}
//if (RadioButton1->Checked==true) Rintdpl[i]=R[1];
//if (RadioButton2->Checked==true) Rintdpl[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintdpl[i]=R[1];
if (RadioButton55->Checked==true) Rintdpl[i]=R[1];
if (RadioButton2->Checked==true)  Rintdpl[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintdpl[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mud, pMutd, Jh1d, JhSW1d, JSW1d, Jd, Mu00d;
};

//---------------------------------------------------------------------------
void TForm1::Difuz0_Loop(double R00, double nL0, double &LhD0,double *MuDS, double *Rint)
{   //Функція для розрахунку за дислокаційними петлями (ід. частина монокристалу)
double R[3];
//double Mu[MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM];
  double *Mu, *pMut, *Jh1, *JhSW1,*JSW1, *J, *Mu00;
  Mu    = new double[m1_teor+1];
  pMut  = new double[m1_teor+1];
  Jh1   = new double[m1_teor+1];
  JhSW1 = new double[m1_teor+1];
  JSW1  = new double[m1_teor+1];
  J     = new double[m1_teor+1];
  Mu00  = new double[m1_teor+1];
double z,v,u,r;
double m0, B21, b2, b3, b4,Beta,k0,Ref1,Km1;
long double B11;
double EL0;

LhD0=koefLh*(nL0)*R00*R00*R00*exp(1.5*LogN(M_E,(H*b)));
EL0=exp(-LhD0);
//Memo8->Lines->Add(FloatToStr(a)+'\t'+FloatToStr(b)+'\t'+FloatToStr(H)+'\t'+FloatToStr(EL0)+'\t'+FloatToStr(VelKom)+'\t'+'\t'+FloatToStr(R00)+'\t'+FloatToStr(nL0));
//Memo8->Lines->Add(FloatToStr(SinTeta)+'\t'+FloatToStr(CosTeta)+'\t'+FloatToStr(cos(psi))+'\t'+FloatToStr(psi));

for (int i=0; i<=m1_teor; i++)
{
Rint[i]=0;

for (int n=nC1; n<=nC; n++)
{
z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
v=2*(z*g[n]/(EL0*EL0)-p[n]);
u=(z*z-g[n]*g[n])/(EL0*EL0)+Kapa[n]*Kapa[n]-1.;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[i]=(0.5*Mu0/gamma0)*(1+b_as)/2.*(1+r*EL0/fabs(g[n]));
//Memo9->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(n)+'\t'+FloatToStr(Mu[i]));
pMut[i]=(1-exp(-2*Mu[i]*dl0))/(2*Mu[i]*dl0);
m0=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);
B11=(4/15.)*(M_PI*b*R00*R00/VelKom)*(M_PI*b*R00*R00/VelKom);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2=B11+0.5*B21*CosTeta*CosTeta;
b3=B21*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4=B21*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0=(2*M_PI/Lambda)*Sin2Teta*DeltaTeta[i];
Ref1=R00*EL0*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
if (fabs(k0)<=Km1)
{
Jh1[i]=b2*LogN(M_E,(Km1*Km1+Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i]))+(b3*k0*k0+b4*Mu[i]*Mu[i])*(1/(Km1*Km1+Mu[i]*Mu[i])-1/(k0*k0+Mu[i]*Mu[i]));//область хуаня
JhSW1[i]=(Km1*Km1/(Km1*Km1+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(Km1*Km1+Mu[i]*Mu[i])));///область стокса вілсона
J[i]=Jh1[i]+JhSW1[i];
}
if (fabs(k0)>Km1)
{
JSW1[i]=(Km1*Km1/(k0*k0+Mu[i]*Mu[i]))*(b2-0.5*((b3*k0*k0+b4*Mu[i]*Mu[i])/(k0*k0+Mu[i]*Mu[i])));
J[i]=JSW1[i];
}
//Memo9->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(J[i])+'\t'+FloatToStr(Jh1[i])+'\t'+FloatToStr(JhSW1[i])+'\t'+FloatToStr(JSW1[i])+'\t'+FloatToStr(m0)+'\t'+FloatToStr(Mu[i])+'\t'+FloatToStr(Km1)+'\t'+FloatToStr(k0)+'\t'+FloatToStr(B11)+'\t'+FloatToStr(B21)+'\t'+FloatToStr(b2)+'\t'+FloatToStr(b3)+'\t'+FloatToStr(b4));

MuDS[i]=(nL0*VelKom)*C[n]*C[n]*EL0*EL0*m0*J[i];
Mu00[i]=MuDS[i]*pMut[i];

R[n]= Fabsj_PL[i][n]*Mu00[i]*dl0/(gamma0);
}
//if (RadioButton1->Checked==true) Rint[i]=R[1];
//if (RadioButton2->Checked==true) Rint[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rint[i]=R[1];
if (RadioButton55->Checked==true) Rint[i]=R[1];
if (RadioButton2->Checked==true)  Rint[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rint[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}

//if (CheckBox53->Checked==true)for (int i=0; i<=m1_teor; i++)
//Memo8->Lines->Add(FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(Rint[i])+'\t'+FloatToStr(MuDS[i])+'\t'+FloatToStr(J[i])+'\t'+FloatToStr(Jh1[i])+'\t'+FloatToStr(JhSW1[i])+'\t'+FloatToStr(JSW1[i])+'\t'+FloatToStr(m0)+'\t'+FloatToStr(Mu[i])+'\t'+FloatToStr((1+sin((2*Km1*Ref1)*sqrt(2*Km1*Ref1)))));

  delete Mu, pMut, Jh1, JhSW1, JSW1, J, Mu00;
};

//---------------------------------------------------------------------------
void TForm1::Difuz0_LoopAniz(double R00, double nL0, double &LhD0,double *MuDS, double *Rint)
{    //Функція для розрахунку за дислокаційними петлями з урах. анізотропії (ід. частина монокристалу)
double R[3],L_ext[3];
//double Mu[MM],pMut[MM],Jh1[MM],JhSW1[MM],JSW1[MM],J[MM],Mu00[MM];
  double *Mu;
  Mu    = new double[m1_teor+1];
double z,v,u,r,m0;
double B21, b2_,Beta,Kc1,Ref1,Km1,Koef,B11_;
long double B11;
double EL0;

LhD0=koefLh*(nL0)*R00*R00*R00*exp(1.5*LogN(M_E,(H*b)));
EL0=exp(-LhD0);

for (int i=0; i<=m1_teor; i++)
{
Rint[i]=0.;

for (int n=nC1; n<=nC; n++)
{
z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
v=2*(z*g[n]/(EL0*EL0)-p[n]);
u=(z*z-g[n]*g[n])/(EL0*EL0)+Kapa[n]*Kapa[n]-1;
r=sqrt(0.5*(sqrt(u*u+v*v)-u));
Mu[i]=(0.5*Mu0/gamma0)*(1+b_as)/2.*(1+r*EL0/fabs(g[n]));
//pMut[i]=(1-exp(-2*Mu[i]*dl0))/(2*Mu[i]*dl0);
m0=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);

if(CheckBox56->Checked==true)     // Молодкін Дедерікс
{
B11=(4/15.)*(M_PI*b*R00*R00/VelKom)*(M_PI*b*R00*R00/VelKom);
Beta=0.25*(3*Nu*Nu+6*Nu-1)/((1-Nu)*(1-Nu));
B21=Beta*B11;
b2_=B11+0.5*B21*CosTeta*CosTeta;
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2*M_PI/L_ext[n];///для петель
Ref1=R00*EL0*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
Koef=b2_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)));
MuDS[i]=(nL0*VelKom)*C[n]*C[n]*EL0*EL0*m0*Koef;
}
if(CheckBox57->Checked==true)     // Уляна Дедерікс
{
B11_=(M_PI*b*R00*R00/VelKom)*(M_PI*b*R00*R00/VelKom);
L_ext[1]=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH);
L_ext[2]=Lambda*sqrt(gamma0*fabs(gammah))/(C[2]*ModChiRH);
Kc1=2*M_PI/L_ext[n];///для петель
Ref1=R00*EL0*sqrt(H*b);///для петель
Km1=2*M_PI/Ref1;///для петель
Koef=0.5*B11_*LogN(M_E,(Km1*Km1/(Kc1*Kc1)))*D_loop;
MuDS[i]=(nL0*VelKom)*C[n]*C[n]*EL0*EL0*m0*Koef;
}

//if(CheckBox61->Checked==true) pMut[i]=1;
//Mu00[i]=MuDS[i]*pMut[i];

R[n]=  Fabsj_PL[i][n]*MuDS[i]*dl0/(gamma0);
//if (CheckBox53->Checked==true)
//Memo8->Lines->Add(FloatToStr(R[i])+'\t'+FloatToStr(MuDS[i])+'\t'+FloatToStr(pMut[i])+'\t'+FloatToStr(Fabsj0*dl0/(gamma0)*(1.+sin((2.*Km1*Ref1)*sqrt(2.*Km1*Ref1))))+'\t'+FloatToStr(Koef)+'\t'+FloatToStr(L_ext[1])+'\t'+FloatToStr(m0));
}
//if (RadioButton1->Checked==true) Rint[i]=R[1];
//if (RadioButton2->Checked==true) Rint[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rint[i]=R[1];
if (RadioButton55->Checked==true) Rint[i]=R[1];
if (RadioButton2->Checked==true)  Rint[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rint[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mu;
};

//------------------------------------------
void TForm1::Difuz0_SferClaster(double R0p0,double np0,double eps0,double &Lhp0,double *MuDSP,double *RintP)//функція для сферичних кластерів 1 (ід. частина монокристалу)
{
double R [3];
//double MuP [MM],pMutP[MM],Jh1P[MM],JhSW1P[MM],JSW1P[MM],JP[MM],Mu00P[MM];
  double *MuP, *pMutP, *Jh1P, *JhSW1P,*JSW1P, *JP, *Mu00P;
  MuP    = new double[m1_teor+1];
  pMutP  = new double[m1_teor+1];
  Jh1P   = new double[m1_teor+1];
  JhSW1P = new double[m1_teor+1];
  JSW1P  = new double[m1_teor+1];
  JP     = new double[m1_teor+1];
  Mu00P  = new double[m1_teor+1];
double zP, vP,uP,rP,AKl;
double m0P, B22, b2P, b3P, b4P,BetaP,k0P,Ref1P,Km1P;
long double B12;
//double MuPj;
double Gama, hh,Alfa0,EP0;
double n00,Eta0;

n00=(4/3.)*M_PI*R0p0*R0p0*R0p0/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps0*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Eta0=Alfa0*exp((1/3.)*LogN(M_E,n00))*hh;
Lhp0=(np0*VelKom)*n00*exp((3/2.)*LogN(M_E,Eta0));
EP0=exp(-Lhp0);

for (int i=0; i<=m1_teor; i++)
{
RintP[i]=0;

for (int n=nC1; n<=nC; n++)
{
zP=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
vP=2*(zP*g[n]/(EP0*EP0)-p[n]);
uP=(zP*zP-g[n]*g[n])/(EP0*EP0)+Kapa[n]*Kapa[n]-1;
rP=sqrt(0.5*(sqrt(uP*uP+vP*vP)-uP));
MuP[i]=(0.5*Mu0/gamma0)*(1+b_as)/2.*(1+rP*EP0/fabs(g[n]));
pMutP[i]=(1-exp(-2*MuP[i]*dl0))/(2*MuP[i]*dl0);
m0P=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);
B12=0;
AKl=Gama*eps0*R0p0*R0p0*R0p0;
B22=(4*M_PI*AKl/VelKom)*(4*M_PI*AKl/VelKom);
b2P=B12+0.5*B22*CosTeta*CosTeta;
b3P=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4P=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0P=(2*M_PI/Lambda)*Sin2Teta*DeltaTeta[i];
Ref1P=sqrt(H*fabs(AKl))*EP0;///для сф. класт
Km1P=2*M_PI/(Ref1P);///для сф. класт
if (fabs(k0P)<=Km1P)
{
Jh1P[i]=b2P*LogN(M_E,(Km1P*Km1P+MuP[i]*MuP[i])/(k0P*k0P+MuP[i]*MuP[i]))+(b3P*k0P*k0P+b4P*MuP[i]*MuP[i])*(1/(Km1P*Km1P+MuP[i]*MuP[i])-1/(k0P*k0P+MuP[i]*MuP[i]));//область хуаня
JhSW1P[i]=(Km1P*Km1P/(Km1P*Km1P+MuP[i]*MuP[i]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[i]*MuP[i])/(Km1P*Km1P+MuP[i]*MuP[i])));///область стокса вілсона
JP[i]=Jh1P[i]+JhSW1P[i];//+JhSWoscP[i];
}
if (fabs(k0P)>Km1P)
{
JSW1P[i]=(Km1P*Km1P/(k0P*k0P+MuP[i]*MuP[i]))*(b2P-0.5*((b3P*k0P*k0P+b4P*MuP[i]*MuP[i])/(k0P*k0P+MuP[i]*MuP[i])));
JP[i]=JSW1P[i];
}

MuDSP[i]=(np0*VelKom)*C[n]*C[n]*EP0*EP0*m0P*JP[i];
Mu00P[i]=MuDSP[i]*pMutP[i];
R[n]= Fabsj_PL[i][n]*Mu00P[i]*dl0/(gamma0);
}
//if (RadioButton1->Checked==true) RintP[i]=R[1];
//if (RadioButton2->Checked==true) RintP[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  RintP[i]=R[1];
if (RadioButton55->Checked==true) RintP[i]=R[1];
if (RadioButton2->Checked==true)  RintP[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) RintP[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete MuP, pMutP, Jh1P, JhSW1P, JSW1P, JP, Mu00P;
};

//------------------------------------------
void TForm1::Diduz0_DiscClaster(double R0d0, double nd0,double eps0d,double &Lhpd0, double *MuDSd,double *Rintd)//функція для дискових кластерів (ід. частина монокристалу)
{
double R [3];
//double Mud[MM],pMutd[MM],Jh1d[MM],JhSW1d[MM],JSW1d[MM],Jd[MM],Mu00d[MM];
  double *Mud, *pMutd, *Jh1d, *JhSW1d,*JSW1d, *Jd, *Mu00d;
  Mud    = new double[m1_teor+1];
  pMutd  = new double[m1_teor+1];
  Jh1d   = new double[m1_teor+1];
  JhSW1d = new double[m1_teor+1];
  JSW1d  = new double[m1_teor+1];
  Jd     = new double[m1_teor+1];
  Mu00d  = new double[m1_teor+1];
double zd,vd,ud,rd,AKld;
double m0d, B22, b2d, b3d, b4d,Betad,k0d,Ref1d,Km1d;
long double B12;
//double Mudj;
double Gama, hp0, hh,Alfa0,Ed0;
double n00,Eta0;

hp0=3.96*R0d0*exp(0.5966*log((0.89e-8/R0d0)));
if (fitting==0) Edit49->Text=FloatToStr(hp0*1e8);
n00=M_PI*R0d0*R0d0*hp0/VelKom;
Gama=(1+Nu)/(3*(1-Nu));
Alfa0=Gama*eps0d*exp((1/3.)*LogN(M_E,(6*M_PI*M_PI/160.)));
hh=0.5*H2Pi*a/M_PI;
Eta0=Alfa0*exp((1/3.)*LogN(M_E,n00))*hh;
Lhpd0=(nd0*VelKom)*n00*exp((3/2.)*LogN(M_E,Eta0));
Ed0=exp(-Lhpd0);

for (int i=0; i<=m1_teor; i++)
{
Rintd[i]=0;

for (int n=nC1; n<=nC; n++)
{
zd=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);
vd=2*(zd*g[n]/(Ed0*Ed0)-p[n]);
ud=(zd*zd-g[n]*g[n])/(Ed0*Ed0)+Kapa[n]*Kapa[n]-1;
rd=sqrt(0.5*(sqrt(ud*ud+vd*vd)-ud));
Mud[i]=(0.5*Mu0/gamma0)*(1+b_as)/2.*(1+rd*Ed0/fabs(g[n]));
pMutd[i]=(1-exp(-2*Mud[i]*dl0))/(2*Mud[i]*dl0);
m0d=(M_PI*VelKom/4.)*(H2Pi*ModChiRH/Lambda)*(H2Pi*ModChiRH/Lambda);
AKld=3*Gama*eps0d*R0d0*R0d0*hp0/4;
B12=(4*M_PI*AKld/VelKom)*(4*M_PI*AKld/VelKom);
B22=(4*M_PI*AKld/VelKom)*(4*M_PI*AKld/VelKom);
b2d=B12+0.5*B22*CosTeta*CosTeta;
b3d=B22*(0.5*CosTeta*CosTeta-SinTeta*SinTeta);
b4d=B22*(0.5*CosTeta*CosTeta-cos(psi)*cos(psi));
k0d=(2*M_PI/Lambda)*Sin2Teta*DeltaTeta[i];
Ref1d=sqrt(H*fabs(AKld))*Ed0;///для диск. класт
Km1d=2*M_PI/(Ref1d);///для диск. класт
if (fabs(k0d)<=Km1d)
{
Jh1d[i]=b2d*LogN(M_E,(Km1d*Km1d+Mud[i]*Mud[i])/(k0d*k0d+Mud[i]*Mud[i]))+(b3d*k0d*k0d+b4d*Mud[i]*Mud[i])*(1/(Km1d*Km1d+Mud[i]*Mud[i])-1/(k0d*k0d+Mud[i]*Mud[i]));//область хуаня
JhSW1d[i]=(Km1d*Km1d/(Km1d*Km1d+Mud[i]*Mud[i]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[i]*Mud[i])/(Km1d*Km1d+Mud[i]*Mud[i])));///область стокса вілсона
Jd[i]=Jh1d[i]+JhSW1d[i];
}
if (fabs(k0d)>Km1d)
{
JSW1d[i]=(Km1d*Km1d/(k0d*k0d+Mud[i]*Mud[i]))*(b2d-0.5*((b3d*k0d*k0d+b4d*Mud[i]*Mud[i])/(k0d*k0d+Mud[i]*Mud[i])));
Jd[i]=JSW1d[i];
}

MuDSd[i]=(nd0*VelKom)*C[n]*C[n]*Ed0*Ed0*m0d*Jd[i];
Mu00d[i]=MuDSd[i]*pMutd[i];

R[n]= Fabsj_PL[i][n]*Mu00d[i]*dl0/(gamma0);
}
//if (RadioButton1->Checked==true) Rintd[i]=R[1];
//if (RadioButton2->Checked==true) Rintd[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rintd[i]=R[1];
if (RadioButton55->Checked==true) Rintd[i]=R[1];
if (RadioButton2->Checked==true)  Rintd[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rintd[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
  delete Mud, pMutd, Jh1d, JhSW1d, JSW1d, Jd, Mu00d;
};

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//void __fastcall TForm1::Button3Click(TObject *Sender)    // Розраховує  когер. складову від підкл. та ПШ
//{   RozrachKoger();  }

void TForm1::RozrachKoger()
{
  double *R_cogerTT;
  R_cogerTT  = new double[m1_teor+1];

if (RadioButton11->Checked==true) RozrachKogerTT(R_cogerTT);
if (RadioButton13->Checked==true) RozrachKogerUD(R_cogerTT);
if (RadioButton12->Checked==true) RozrachKogerTT_kin(R_cogerTT);
if (RadioButton16->Checked==true) RozrachKogerTT_kin_rozvor(R_cogerTT);
if (RadioButton33->Checked==true) RozrachKoger_kin_Mol(R_cogerTT);
if (RadioButton35->Checked==true) RozrachKogerUD_kin(R_cogerTT);
if (RadioButton50->Checked==true) RozrachKogerUD_din_Mol(R_cogerTT);
if (RadioButton68->Checked==true) RozrachKogerTT_PynByshKato(R_cogerTT);

for (int i=0; i<=m1_teor; i++) R_cogerTT_[i][KDV_lich]=R_cogerTT[i];

if (fitting==0)
{
for (int i=0; i<=m1_teor; i++)
{
if (number_KDV==1)
	{
	if (CheckBox19->Checked==false)
	{
	Series4->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia);
	Series13->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia);
	}
	Series8->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia);
	}
if (number_KDV==2 || number_KDV==3)
	{
	if (CheckBox19->Checked==false)
	{
	if (KDV_lich==3) 	Series47->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia	);
	if (KDV_lich==2) 	Series4->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia	);
	if (KDV_lich==1) 	Series13->AddXY(DeltaTeta[i]/M_PI*(3600*180),R_cogerTT[i],"",clFuchsia	);
	}
	}

}
}
if (CheckBox53->Checked==true) Memo8-> Lines->Add("Когерентне пораховано!");

  delete R_cogerTT;
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerTT(double *R_cogerTT) // функція для розрах. когер. КДВ (за Такагі-Топеном)
{
double R[3];
double L,hpl0;   // DDpd[KM],
  double *DDpd;
  DDpd   = new double[KM];
complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);

double      *x0i_a,*eta0_a;
  x0i_a   = new double[KM];
  eta0_a  = new double[KM];
complex< double>  xhp_a[3][KM],xhn_a[3][KM],eta_a,sigmasp_a,sigmasn_a;


         //x0r0=0;
         //x0r=0;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);


for (int k=1; k<=km;k++)
{
//         x0i0=ChiI0_a[k];
      x0i_a[k]=ChiI0_a[k];
      xhp_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);   //для  центр.-сим. крист.
      xhn_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);
      xhp_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);   //для  центр.-сим. крист.
      xhn_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);
      eta0_a[k]=M_PI*x0i_a[k]*(1+b_as)/(Lambda*gamma0);
}
/*
if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);
 */
if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
        DD0=(apl-a)/a;
 for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1)*(DD0+1)-1 ;
      L=0;
 for (int k=1; k<=km;k++) L=L+Dl[k] ;
      hpl0=hpl-L;
}
if (CheckBox31->Checked==false)
{
for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
}

      eta00=M_PI*x0i0*(1+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=-(eta0*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));

//      Обчислення амплітуди підкладки
          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0*Esum0*Esum0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0*Esum0);
//Memo9-> Lines->Add(FloatToStr(real(xhp0[n]))+'\t'+FloatToStr(real(xhn0[n]))+'\t'+FloatToStr(abs(As)));

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
{
//           YYs0=2*M_PI/d*DD0;
if (CheckBox18->Checked==false)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2*sin(2*tb);
}
if (CheckBox18->Checked==true)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2*sin(2*tb);
}
           YYs0=eta+YYs0;
           sqs=sqrt((YYs0/2.)*(YYs0/2.)-sigmasp*sigmasn*Esum0pl*Esum0pl);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs0/2.+sqs)/(sigmasn*Esum0pl);
//            if (abs(x2s-As)<0.0000000001) goto m1001pl;
            x1s=-(YYs0/2.-sqs)/(sigmasn*Esum0pl);
            x3s=(x1s-As)/(x2s-As);
            expcs=exp(-2.*ssigma*hpl0);
//m1001pl:
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}
//Memo9->Lines->Add( "RozrachKogerTT id пройшло");

//      Обчислення амплітуди від заданого профілю:
  if (CheckBox67->Checked==false) goto m102ps;    // якщо пор. шару немає
        for (int k=1; k<=km;k++)
{
      eta_a=-(eta0_a[k]*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
      sigmasp_a=M_PI*xhp_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn_a=M_PI*xhn_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));

      //           YYs[k]=2.*M_PI/d*DD[k];
if (CheckBox18->Checked==false)
{
 YYs[k]=M_PI/Lambda/gamma0*DDpd[k]*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2.*sin(2.*tb);
}
if (CheckBox18->Checked==true)
{
 YYs[k]=M_PI/Lambda/gamma0*DDpd[k]*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2.*sin(2.*tb);
}
           YYs[k]=eta_a+YYs[k];
            sqs=sqrt((YYs[k]/2.)*(YYs[k]/2.)-sigmasp_a*sigmasn_a*Esum[k]*Esum[k]);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0_a[k]<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs[k]/2.+sqs)/(sigmasn_a*Esum[k]);
//        if (abs(x2s-As).lt.1E-10) goto 1001
            x1s=-(YYs[k]/2.-sqs)/(sigmasn_a*Esum[k]);
            x3s=(x1s-As)/(x2s-As);
           expcs=exp(-2.*ssigma*Dl[k]);
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
//1001         continue
//Memo9->Lines->Add( "RozrachKogerTT km end пройшло");
//Memo9-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(DDpd[k])+'\t'+FloatToStr(Dl[k]));

}
//Memo9->Lines->Add( "RozrachKogerTT  start пройшло");
m102ps:
//Memo9-> Lines->Add(FloatToStr(real(xhp0[n]))+'\t'+FloatToStr(real(xhn0[n]))+'\t'+FloatToStr(abs(As)));
        R[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);
//Memo9->Lines->Add( "RozrachKogerTT km end пройшло");
}
//if (RadioButton1->Checked==true) R_cogerTT[i]=R[1];
//if (RadioButton2->Checked==true) R_cogerTT[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  R_cogerTT[i]=R[1];
if (RadioButton55->Checked==true) R_cogerTT[i]=R[1];
if (RadioButton2->Checked==true)  R_cogerTT[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) R_cogerTT[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
//Memo9->Lines->Add( "RozrachKogerTT R_cogerTT[i] end пройшло");
}
  delete DDpd, x0i_a, eta0_a;
//Memo9->Lines->Add( "RozrachKogerTT 3 пройшло");
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerTT_PynByshKato(double *R_cogerTT) // функція для розрах. когер. КДВ (за Такагі-Топеном)
{
double R[3];
double L,hpl0;   // DDpd[KM],
  double *DDpd;
  DDpd   = new double[KM];
complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);
double      *x0i_a,*eta0_a;
  x0i_a   = new double[KM];
  eta0_a  = new double[KM];
complex< double>  xhp_a[3][KM],xhn_a[3][KM],eta_a,sigmasp_a,sigmasn_a; 
double  TAU[KM], kTau,expcsr,expcsi,expcsi1;  // d,
complex< double>  RO[KM], TS[3];
long double       expcsr1;
         //x0r0=0;
         //x0r=0;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

for (int k=1; k<=km;k++)
{
//         x0i0=ChiI0_a[k];
      x0i_a[k]=ChiI0_a[k];
      xhp_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);   //для  центр.-сим. крист.
      xhn_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);
      xhp_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);   //для  центр.-сим. крист.
      xhn_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);
      eta0_a[k]=M_PI*x0i_a[k]*(1+b_as)/(Lambda*gamma0);
}


if (CheckBox3->Checked==true)            // !!!!!! ТУТ Однакове для всіх рефлексів
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);
//  d=a/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) Memo3-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(TAU[k])+'\t'+FloatToStr(Esum[k]));

if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
        DD0=(apl-a)/a;
 for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1)*(DD0+1)-1 ;
      L=0;
 for (int k=1; k<=km;k++) L=L+Dl[k] ;
      hpl0=hpl-L;
//  d=apl/sqrt(h*h+k*k+l*l);
}
if (CheckBox31->Checked==false)
{
for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
}

kTau=1e-8*StrToFloat(Edit395->Text);      //см
if (CheckBox84->Checked==true) for (int k=1; k<=km;k++) TAU[k]=f[k]*kTau;
if (CheckBox84->Checked==false) for (int k=1; k<=km;k++) TAU[k]=kTau;

      eta00=M_PI*x0i0*(1+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

//Memo9-> Lines->Add("  Обчислення теор. когер. КДВ  ");

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=-(eta0*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
//Memo9-> Lines->Add(FloatToStr(real(sigmasp))+'\t'+FloatToStr(imag(sigmasp))+'\t'+FloatToStr(abs(5555)));

//      Обчислення амплітуди підкладки
          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0*Esum0*Esum0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0*Esum0);
//Memo9-> Lines->Add(FloatToStr(real(xhp0[n]))+'\t'+FloatToStr(real(xhn0[n]))+'\t'+FloatToStr(abs(As)));

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
{
//           YYs0=2*M_PI/d*DD0;
if (CheckBox18->Checked==false)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2*sin(2*tb);
}
if (CheckBox18->Checked==true)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2*sin(2*tb);
}
           YYs0=eta+YYs0;
           sqs=sqrt((YYs0/2.)*(YYs0/2.)-sigmasp*sigmasn*Esum0pl*Esum0pl);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs0/2.+sqs)/(sigmasn*Esum0pl);
//            if (abs(x2s-As)<0.0000000001) goto m1001pl;
            x1s=-(YYs0/2.-sqs)/(sigmasn*Esum0pl);
            x3s=(x1s-As)/(x2s-As);
            expcs=exp(-2.*ssigma*hpl0);
//m1001pl:
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}
//Memo9->Lines->Add( "RozrachKogerTT id пройшло");

//      Обчислення амплітуди від заданого профілю:
  if (CheckBox67->Checked==false) goto m102ps;    // якщо пор. шару немає
        for (int k=1; k<=km;k++)
{
      eta_a=-(eta0_a[k]*cmplxi+2*M_PI*b_as*sin(2*tb)*DeltaTeta[i]/(Lambda*gamma0));
      sigmasp_a=M_PI*xhp_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn_a=M_PI*xhn_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));

      //           YYs[k]=2.*M_PI/d*DD[k];
if (CheckBox18->Checked==false)
{
 YYs[k]=M_PI/Lambda/gamma0*DDpd[k]*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2.*sin(2.*tb);
}
if (CheckBox18->Checked==true)
{
 YYs[k]=M_PI/Lambda/gamma0*DDpd[k]*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2.*sin(2.*tb);
}
           RO[k]=sigmasp_a*sigmasn_a*(1.-Esum[k])*TAU[k];
//           YYs[k]=2*M_PI/d*DD[k]+cmplxi*2.*RO[k];
           YYs[k]=eta_a+YYs[k]+cmplxi*2.*RO[k];
            sqs=sqrt((YYs[k]/2.)*(YYs[k]/2.)-sigmasp_a*sigmasn_a*Esum[k]*Esum[k]);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0_a[k]<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs[k]/2.+sqs)/(sigmasn_a*Esum[k]);
//        if (abs(x2s-As).lt.1E-10) goto 1001
            x1s=-(YYs[k]/2.-sqs)/(sigmasn_a*Esum[k]);
            x3s=(x1s-As)/(x2s-As);
           expcs=exp(-2.*ssigma*Dl[k]);
//           Memo9-> Lines->Add(FloatToStr(expcs.real())+' i*' + FloatToStr(expcs.real()));

            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}
//Memo9->Lines->Add( "RozrachKogerTT  start пройшло");
m102ps:
        R[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);

//Memo9->Lines->Add( "RozrachKogerTT km end пройшло");
}
if (RadioButton1->Checked==true)  R_cogerTT[i]=R[1];
if (RadioButton55->Checked==true) R_cogerTT[i]=R[1];
if (RadioButton2->Checked==true)  R_cogerTT[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) R_cogerTT[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
//Memo9->Lines->Add( "RozrachKogerTT R_cogerTT[i] end пройшло");
}

//for (int k=1; k<=km;k++)
//           Memo3-> Lines->Add(FloatToStr(k)+'\t'+FloatToStr(Dl[k])+'\t'+FloatToStr(TAU[k]));

  delete DDpd, x0i_a, eta0_a;
//Memo9->Lines->Add( "RozrachKogerTT 3 пройшло");
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerTT_kin_rozvor(double *R_cogerTT) // функція для розрах. когер. КДВ (за Такагі-Топеном)
{
double R[3],Rd[3];  //,Rk[3][100],Rd_[MM],Rkin[MM];
double L,hpl0,L_mozaik,dfi;   // DDpd[KM],
//int    NN[KM], zsuv[200];             //km_rozv
double d,AAA,Afi;  //,fi[KM],DDpl[100],LL[100];
//double   fff[100],fff1[100],nn_m1[100];
//double DD_rozv[100]; //!!! Чому так не іде?????  (Раніше не йшло)
double fm, qm,Amsin, Amcos,Am;
//double FMS[KM];
  double *Rd_, *Rkin, *DDpd, *fi, *DDpl, *LL, *fff, *fff1, *nn_m1, *FMS;
  int *NN, *zsuv;
  Rd_   = new double[m1_teor+1];
  Rkin  = new double[m1_teor+1];
  DDpd  = new double[KM];
  fi    = new double[KM];
  DDpl  = new double[100];
  LL    = new double[100];
  fff   = new double[100];
  fff1  = new double[100];
  nn_m1 = new double[100];
  FMS   = new double[KM];
  NN    = new int[KM];
  zsuv  = new int[200];
double **Rk;
Rk = new double*[MM];
for(int i=0;i<MM; i++)
{
    Rk[i]  = new double[2*km_rozv+1];
}

if (RadioButton14->Checked==true) km_rozv=0; // 1 блок + профіль
if (RadioButton15->Checked==true || RadioButton26->Checked==true) km_rozv=StrToInt(Edit84->Text);// Блоки без ПШ чи з ПШ
double **Rk_, **Rk_z, **Rk__;  //Rk_[MM][100],Rk_z[MM][100],Rk__[MM][100]
Rk_ = new double*[MM];
Rk_z = new double*[MM];
Rk__ = new double*[MM];
for(int i=0;i<MM; i++)
{
    Rk_[i]  = new double[2*km_rozv+1];
    Rk_z[i] = new double[2*km_rozv+1];
    Rk__[i] = new double[2*km_rozv+1];
}

complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);


         //x0r0=0.;
         //x0r=0.;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);

if (KDV_lich==1) AAA=StrToFloat(Edit78->Text);
if (KDV_lich==2) AAA=StrToFloat(Edit85->Text);
if (KDV_lich==3) AAA=StrToFloat(Edit86->Text);

if (RadioButton14->Checked==true)  // 1 блок + профіль
{
  km_rozv=0;
  L_mozaik=1e-8*StrToFloat(Edit83->Text);
//  EW=StrToFloat(Edit86->Text);
  DD_rozv[0]=0;
  fi[0]=StrToFloat(Edit79->Text);
  zsuv[0]= fi[0]/ik_[KDV_lich];
  fff[0]=1;
}
if (RadioButton15->Checked==true || RadioButton26->Checked==true) // Блоки без ПШ чи з ПШ
{
  km_rozv=StrToInt(Edit84->Text);
  L_mozaik=1e-8*StrToFloat(Edit83->Text);
  Afi=StrToFloat(Edit80->Text); // Коеф. в  DD_rozv[kr] (fi[kr]);
//  EW=StrToFloat(Edit86->Text);
  DD_rozv[0]=0;
  zsuv[0]=0;
  fi[0]=0;
if (fitting==0)
{   //   nn_m[kr] - функція розподілу по кутах (зчитується з Memo7)
dfi=StrToFloat(Edit132->Text);
if (CheckBox27->Checked==false)
   {
   ReadMemo2stovp(Memo7,km_rozv+1,nn_m,DFi);
   for (int k=0; k<=km_rozv;k++) //  Перенумерація елементів масивів
   {
   nn_m[k]=nn_m[k+1];
   DFi[k]=DFi[k+1];
   }
   }
if (CheckBox27->Checked==true)
   for (int kr=0; kr<=km_rozv;kr++)
   {
   nn_m[kr]=1;
   DFi[kr]=dfi;
   }
if (CheckBox50->Checked==true)
{
//k_param=StrToInt(Edit127->Text);
if (fitting==0) method_lich=0;
PARAM[0][1]=StrToFloat(Edit98->Text);
PARAM[0][2]=StrToFloat(Edit101->Text);
PARAM[0][3]=StrToFloat(Edit102->Text)*1e-8;
PARAM[0][4]=StrToFloat(Edit103->Text)*1e-8;
PARAM[0][5]=StrToFloat(Edit104->Text);
PARAM[0][6]=StrToFloat(Edit105->Text)*1e-8;
PARAM[0][7]=StrToFloat(Edit106->Text)*1e-8;
PARAM[0][8]=StrToFloat(Edit128->Text);
PARAM[0][13]=StrToFloat(Edit95->Text);       //   DDamax
PARAM[0][14]=StrToFloat(Edit107->Text);       //   DDamin
dl=1e-8*StrToFloat(Edit97->Text);
kEW=StrToFloat(Edit128->Text);                     //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   Profil(km_rozv,nn_m1,dl);
km_rozv=km_rozv-1;
   for (int k=0; k<=km_rozv;k++) //  Перенумерація елементів масивів
   {
   nn_m[k]=nn_m1[km_rozv-k];
   DFi[k]=dl*1e8;
   }
}
Memo7->Clear();
for (int kr=0; kr<=km_rozv;kr++) Memo7->Lines->Add(FloatToStr(nn_m[kr])+'\t'+FloatToStr(DFi[kr]));
}

  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
//    DD_rozv[kr]=0.000183202;

    zsuv[kr]=fi[kr]/ik_[KDV_lich];
//Edit85->Text=FloatToStr(DD_rozv[kr]);
//Edit85->Text=FloatToStr(DD_rozv[kr]);
  }
double    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;
for (int kr=0; kr<=km_rozv;kr++)  fff[kr]=fff1[kr];
for (int kr=1; kr<=km_rozv;kr++)  fff[kr+km_rozv]=fff1[kr];
//for (int kr=1; kr<=km_rozv;kr++)  Memo1->Lines->Add(FloatToStr(nn_m[kr])+'\t'+FloatToStr(DFi[kr])+'\t'+FloatToStr(zsuv[kr]));
}

if (RadioButton26->Checked==true) // Блоки без профіля
{
Dl[0]=L_mozaik;
km=0;
LL[0]=0;
}

if (RadioButton14->Checked==true || RadioButton15->Checked==true) // Блоки + профіль
{
 L=0.;
 for (int k=1; k<=km; k++) L=L+Dl[k] ;
        LL[0]=L;
Dl[0]=L_mozaik-L;
 for (int k=1; k<=km; k++)
{
 LL[k]=0;
 for (int i=k+1; i<=km; i++)   LL[k]=LL[k]+Dl[i];
}
}

if (CheckBox31->Checked==true)
{
 DD0=(apl-a)/a;
  hpl0=hpl-L_mozaik;
  d=apl/sqrt(h*h+k*k+l*l);
  for (int k=0; k<=km;k++) NN[k]=Dl[k]/d;
}

if (CheckBox31->Checked==false)
{
  d=a/sqrt(h*h+k*k+l*l);
  for (int k=0; k<=km;k++) NN[k]=Dl[k]/d;
}
// Memo1->Lines->Add(FloatToStr(222)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(Dl[0]));
// Memo1->Lines->Add(FloatToStr(222)+'\t'+FloatToStr(DD[1])+'\t'+FloatToStr(Dl[1]));
//Memo1->Lines->Add(IntToStr(km_rozv)+'\t'+IntToStr(km));

//*******
      eta00=M_PI*x0i0*(1.+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1.+b_as)/(Lambda*gamma0);

// Обчислення теор. когер. КДВ
for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=-(eta0*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));

//      Обчислення амплітуди підкладки
          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0*Esum0*Esum0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0*Esum0);
//  if (i==0) Memo1->Lines->Add(FloatToStr(3)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(Dl[0]));
//  if (i==1) Memo1->Lines->Add(FloatToStr(33)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(Dl[0]));
//  if (i==100) Memo1->Lines->Add(FloatToStr(333)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(Dl[0]));

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
{
//           YYs0=2*M_PI/d*DD0;
if (CheckBox18->Checked==false)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2*sin(2*tb);
}
if (CheckBox18->Checked==true)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2*sin(2*tb);
}
           YYs0=eta+YYs0;
           sqs=sqrt((YYs0/2.)*(YYs0/2.)-sigmasp*sigmasn*Esum0pl*Esum0pl);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs0/2.+sqs)/(sigmasn*Esum0pl);
//            if (abs(x2s-As)<0.0000000001) goto m1001pl;
            x1s=-(YYs0/2.-sqs)/(sigmasn*Esum0pl);
            x3s=(x1s-As)/(x2s-As);
            expcs=exp(-2.*ssigma*hpl0);
//m1012pl:
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}
//  if (i==100) Memo1->Lines->Add(FloatToStr(444)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(Dl[0]));

//      Обчислення амплітуди від заданого профілю (kinemat):
for (int kr=0; kr<=km_rozv;kr++)    // розвороти блока
  {
//Memo7->Lines->Add(IntToStr(i)+'\t'+IntToStr(kr));
          fm=0.;
          FMS[0]=0.;
          Amsin=0.;
          Amcos=0.;
//if (RadioButton26->Checked==true) km=0;   // якщо пор. шару немає

//  DD[0]=DD_rozv[kr];
  DD[0]=0;

//  if (i==100) Memo1->Lines->Add(IntToStr(kr)+'\t'+IntToStr(km_rozv));
//  if (i==100) Memo1->Lines->Add(FloatToStr(555)+'\t'+FloatToStr(DD[0]));

     for (int k=0; k<=km;k++)   //         do k=2,km+1
{
	if (CheckBox31->Checked==true)  // Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
	{
 	DDpl[k]=(DD[k]+1.)*(DD_rozv[kr]+1.)-1. ;
 	DDpd[k]=(DDpl[k]+1.)*(DD0+1.)-1. ;
	}
	if (CheckBox31->Checked==false)
	{
 	DDpl[k]=(DD[k]+1)*(DD_rozv[kr]+1)-1 ;
 	DDpd[k]=DDpl[k] ;
	}

//if (i==100) Memo1->Lines->Add(FloatToStr(666)+'\t'+FloatToStr(DD[0])+'\t'+FloatToStr(DDpd[0])+'\t'+FloatToStr(Dl[0]));
//if (i==100) Memo1->Lines->Add(FloatToStr(k)+'\t'+FloatToStr(DD[1])+'\t'+FloatToStr(DDpd[1])+'\t'+FloatToStr(Dl[1]));

	            FMS[k]=FMS[k-1]+fm;
double DeltaTetaDD_=1./b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
            fm=2.*M_PI*(1./DeltaTetaDD_*DeltaTeta[i]-(1./DeltaTetaDD_*DeltaTeta[i]+1.)*DDpd[k]);
 if (fabs(fm)<1e-15) fm=1e-15;
            qm=((NN[k]+1)/2.)*fm+NN[k]*FMS[k];
if (RadioButton17->Checked==true) Am=sin(NN[k]*fm/2.)/sin(fm/2.);
//if (RadioButton18->Checked==true)Am=sin(NN*fm/2.)/sin(fm/2.)*exp(-Mu0 *dl/sin(tb-psi)*(km-k));
if (RadioButton18->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k]);
if (RadioButton19->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k])*Esum[k];
            Amcos=Amcos+Am*cos(qm);
            Amsin=Amsin+Am*sin(qm);
}
 complex< double> Aks (Amcos+1, Amsin);
      Rk[n][kr]=abs(Aks)*abs(Aks)/40000000.*AAA ;
}  //m102ps:
      Rd[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);
//      R[n]=Rd[n]+Rk[n];
}
//if (RadioButton1->Checked==true) Rd_[i]=Rd[1];
//if (RadioButton2->Checked==true) Rd_[i]=(Rd[1]+C[2]*Rd[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rd_[i]=Rd[1];
if (RadioButton55->Checked==true) Rd_[i]=Rd[1];
if (RadioButton2->Checked==true)  Rd_[i]=(Rd[1]+Monohr[1]*Rd[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rd_[i]=(Rd[1]+Monohr[2]*Rd[2])/(1+Monohr[2]);

if (RadioButton1->Checked==true)
  for (int kr=0; kr<=km_rozv;kr++)
    Rk_[i][kr]=Rk[1][kr];
if (RadioButton55->Checked==true)
  for (int kr=0; kr<=km_rozv;kr++)
    Rk_[i][kr]=Rk[1][kr];
if (RadioButton2->Checked==true)
  for (int kr=0; kr<=km_rozv;kr++)
    Rk_[i][kr]=(Rk[1][kr]+Monohr[1]*Rk[2][kr])/(1+Monohr[1]);
if (RadioButton56->Checked==true)
  for (int kr=0; kr<=km_rozv;kr++)
    Rk_[i][kr]=(Rk[1][kr]+Monohr[2]*Rk[2][kr])/(1+Monohr[2]);
}

//Memo1->Lines->Add(FloatToStr(Rk_[20][0])+'\t'+FloatToStr(Rk_[20][1]));

for (int i=0; i<=m1_teor; i++) for (int kr=0; kr<=km_rozv;kr++) Rk__[i][kr]=Rk_[i][kr];

for (int kr=0; kr<=km_rozv;kr++)
  {
  for (int i=0; i<=m1_teor; i++)
     {
     if (i-zsuv[kr]<0 || i+zsuv[kr]>m1_teor); //Rk_z[i][kr]=Rk__[i][kr];
     else   Rk_z[i][kr]=Rk__[i-zsuv[kr]][kr];
     }
  }

for (int kr=1; kr<=km_rozv;kr++)
  {
  for (int i=0; i<=m1_teor; i++)
     {
     if (i-zsuv[kr]<0 || i+zsuv[kr]>m1_teor);// Rk_z[i][kr]=Rk__[i][kr];
     else   Rk_z[i][kr+km_rozv]=Rk__[i+zsuv[kr]][kr];
  }
}
//Memo1->Lines->Add(IntToStr(0)+'\t'+IntToStr(zsuv[0])+'\t'+FloatToStr(Rk_[20][0]));

for (int i=0; i<=m1_teor; i++)   //додає інтенсивність від всіх підшарів
{
Rkin[i]=0;
   for (int kr=0; kr<=2*km_rozv;kr++)
   {
        Rkin[i]=Rkin[i]+Rk_z[i][kr]*fff[kr];
   }
}
//Memo1->Lines->Add(FloatToStr(fff[0])+'\t'+FloatToStr(fff[1])+'\t'+FloatToStr(Rk_[20][0]));

  for (int i=0; i<=m1_teor; i++)      R_cogerTT[i]=Rd_[i]+Rkin[i];

if (fitting==0) for (int k=0; k<=km_rozv;k++)
  {
Series43->AddXY(fi[k],DD_rozv[k]*10000,"",clFuchsia);
Series44->AddXY(fi[k],fff1[k]*100,"",clFuchsia);
Series32->AddXY(fi[k],0.000002,"",clWhite);

  }
/*double L_tmp,z,fi_vse;
L_tmp=0;
fi_vse=0;
for (int k=1; k<=km_rozv;k++) fi_vse=fi_vse+DFi[k];
for (int k=1; k<=km_rozv;k++)
{
L_tmp=L_tmp+DFi[k];
z=L_tmp-DFi[k]/2.;
Series43->AddXY(z,DD_rozv[k]*10000,"",clFuchsia);
Series27->AddXY(z,nn_m[k],"",clGreen);
Series44->AddXY(z,fff1[k]*100,"",clFuchsia);
}

double Z_shod [2*KM],D_shod [2*KM],L_shod;
L_shod=0;
for (int k=1; k<=km_rozv;k++) L_shod=L_shod+DFi[k];
Z_shod[0]=0;
DFi[km_rozv+1]=0;
for (int k=1; k<=km_rozv;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+DFi[k];
D_shod[2*k-1]=fff1[k];
D_shod[2*k  ]=fff1[k];
}
Z_shod[2*km_rozv+1]=L_shod;
D_shod[2*km_rozv+1]=0;

for (int k=0; k<=2*km_rozv+1;k++) Series22->AddXY(Z_shod[k],D_shod[k]*100,"",clRed);
*/


/*TetaMin=-(m10)*ik;
for (int i=0; i<=m1; i++)
{
DeltaTeta1=(TetaMin+i*ik);
	Series35->AddXY(DeltaTeta1,Rd_[i],"");
	Series36->AddXY(DeltaTeta1,Rkin[i],"");
}     */

  delete Rd_, Rkin, DDpd, fi, DDpl, LL, fff, fff1, nn_m1, FMS;
  delete NN, zsuv;
for(int i=0;i<MM; i++)
{
   delete[] Rk[i];
}
delete[] Rk;

for(int i=0; i<MM; i++)
{
  delete[] Rk_[i];
  delete[] Rk_z[i];
  delete[] Rk__[i];
}
delete[] Rk_;
delete[] Rk_z;
delete[] Rk__;       
   }

//---------------------------------------------------------------------------
void TForm1::RozrachKogerTT_kin(double *R_cogerTT) // функція для розрах. когер. КДВ (за Такагі-Топеном)
{
double R[3],Rd[3],Rk[3]; //,Rd_[MM],Rk_[MM],Rk__[MM];
double L,hpl0; //,DDpd[KM],LL[100];
int    zsuv;   //  NN[KM] ,
double d,AAA,fi;
double fm, qm,Amsin, Amcos,Am;
//double FMS[KM];
  double *Rd_, *Rk_, *Rk__, *DDpd, *LL, *FMS;
  int *NN;
  Rd_   = new double[m1_teor+1];
  Rk_   = new double[m1_teor+1];
  Rk__  = new double[m1_teor+1];
  DDpd  = new double[KM];
  LL    = new double[100];
  FMS   = new double[KM];
  NN    = new int[KM];

complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);

         //x0r0=0.;
         //x0r=0.;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);

if (KDV_lich==1) AAA=StrToFloat(Edit78->Text);
if (KDV_lich==2) AAA=StrToFloat(Edit85->Text);
if (KDV_lich==3) AAA=StrToFloat(Edit86->Text);

 for (int k=1; k<=km; k++)
{
 LL[k]=0;
 for (int i=k+1; i<=km; i++) LL[k]=LL[k]+Dl[i];
}
//  EW=StrToFloat(Edit86->Text);

if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
        DD0=(apl-a)/a;
 for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1.)*(DD0+1.)-1. ;
      L=0.;
 for (int k=1; k<=km;k++) L=L+Dl[k] ;
      hpl0=hpl-L;
d=apl/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) NN[k]=Dl[k]/d;
}
if (CheckBox31->Checked==false)
{
for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
d=a/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) NN[k]=Dl[k]/d;
}

 fi=StrToFloat(Edit79->Text);
  zsuv=fi/ik_[KDV_lich];

      eta00=M_PI*x0i0*(1.+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1.+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=-(eta0*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)  
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));

//      Обчислення амплітуди підкладки
          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0*Esum0*Esum0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0*Esum0);

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
{
//           YYs0=2*M_PI/d*DD0;
if (CheckBox18->Checked==false)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2*sin(2*tb);
}
if (CheckBox18->Checked==true)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2*sin(2*tb);
}
           YYs0=eta+YYs0;
           sqs=sqrt((YYs0/2.)*(YYs0/2.)-sigmasp*sigmasn*Esum0pl*Esum0pl);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs0/2.+sqs)/(sigmasn*Esum0pl);
//            if (abs(x2s-As)<0.0000000001) goto m1001pl;
            x1s=-(YYs0/2.-sqs)/(sigmasn*Esum0pl);
            x3s=(x1s-As)/(x2s-As);
            expcs=exp(-2.*ssigma*hpl0);
//m1012pl:
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}

//      Обчислення амплітуди від заданого профілю (kinemat):
///////  if (CheckBox67->Checked==false) goto m102ps;    // якщо пор. шару немає

          fm=0.;
          FMS[0]=0.;
          Amsin=0.;
          Amcos=0.;
     for (int k=1; k<=km;k++)   //         do k=2,km+1
{            FMS[k]=FMS[k-1]+fm;
double DeltaTetaDD_=1./b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
            fm=2.*M_PI*(1./DeltaTetaDD_*DeltaTeta[i]-(1./DeltaTetaDD_*DeltaTeta[i]+1.)*DDpd[k]);
 if (fabs(fm)<1e-15) fm=1e-15;
            qm=((NN[k]+1.)/2.)*fm+NN[k]*FMS[k];
if (RadioButton17->Checked==true) Am=sin(NN[k]*fm/2.)/sin(fm/2.);
//if (CheckBox5->Checked==true)Am=sin(NN*fm/2.)/sin(fm/2.)*exp(-Mu0*dl/sin(tb-psi)*(km-k));
if (RadioButton18->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k]);
if (RadioButton19->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k])*Esum[k];
            Amcos=Amcos+Am*cos(qm);
            Amsin=Amsin+Am*sin(qm);
}
 complex< double> Aks (Amcos+1, Amsin);
      Rk[n]=abs(Aks)*abs(Aks)/40000000.*AAA ;
//m102ps:
      Rd[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);
//      R[n]=Rd[n]+Rk[n];
}
//if (RadioButton1->Checked==true) Rd_[i]=Rd[1];
//if (RadioButton2->Checked==true) Rd_[i]=(Rd[1]+C[2]*Rd[2])/(1+C[2]);
//if (RadioButton1->Checked==true) Rk_[i]=Rk[1];
//if (RadioButton2->Checked==true) Rk_[i]=(Rk[1]+C[2]*Rk[2])/(1+C[2]);

if (RadioButton1->Checked==true)  Rd_[i]=Rd[1];
if (RadioButton55->Checked==true) Rd_[i]=Rd[1];
if (RadioButton2->Checked==true)  Rd_[i]=(Rd[1]+Monohr[1]*Rd[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rd_[i]=(Rd[1]+Monohr[2]*Rd[2])/(1+Monohr[2]);
if (RadioButton1->Checked==true)  Rk_[i]=Rk[1];
if (RadioButton55->Checked==true) Rk_[i]=Rk[1];
if (RadioButton2->Checked==true)  Rk_[i]=(Rk[1]+Monohr[1]*Rk[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rk_[i]=(Rk[1]+Monohr[2]*Rk[2])/(1+Monohr[2]);
}

for (int i=0; i<=m1_teor; i++) Rk__[i]=Rk_[i];
for (int i=0; i<=m1_teor; i++)
{
if (i-zsuv<0 || i+zsuv>m1_teor) Rk_[i]=Rk__[i];
    else  Rk_[i]=Rk__[i-zsuv];

      R_cogerTT[i]=Rd_[i]+Rk_[i];
}

  delete Rd_, Rk_, Rk__, DDpd, LL, FMS, NN;
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerUD(double *R_cogerTT) // функція для розрах. когер. КДВ (за УДТ)
{
 double R[3];
complex< double> YYs[KM];
complex< double> xhp0[3];
complex< double> xhn0[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0;
double   m00;
complex< double> As;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);

        //x0r0=0.;
//         x0r=0.;
         x0i0=ChiI0;
//         x0i=ChiI0;
       xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
       xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
       xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
       xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
        // complex< double> xo0 (x0r0, ChiI0);

if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);

      eta00=M_PI*x0i0*(1.+b_as)/(Lambda*gamma0);
//      eta0=M_PI*x0i*(1.+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));
//     eta=-(eta0*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)  
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
//      sigmasp=M_PI*xhp*C[n]/(Lambda*fabs(gammah));
//      sigmasn=M_PI*xhn*C[n]/(Lambda*fabs(gamma0));

//      Обчислення амплітуди підкладки
/*          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0);           */

 double aa,d1,hh,z1,K;
complex< double> L,L1,L2,L3,Eps_cog,L2_,L2__;

z1=sqrt(b_as)*sin(2.*tb)*DeltaTeta[i]/(C[n]*ModChiRH);
aa=0;                                                            //++++++++++++++++++++
K=2.*M_PI/Lambda;
m00=-MuDSsum[i]/(K*Esum0);                                      // і яке тут MuDS !!!!!!!!!?????????????????????
hh=m00*(1.+1./b_as)*sqrt(b_as)/(2.*C[n]*ModChiRH);
d1=0;                                                            //++++++++++++++++++++
//C=1;
//E=1;

//p=ModChiIH/ModChiRH;

L1=z1*z1+(g[n]+hh)*(g[n]+hh);
L2_=(z1*z1-(g[n]+hh)*(g[n]+hh)-Esum0*Esum0*(1.-Kapa[n]*Kapa[n]-aa*aa))*(z1*z1-(g[n]+hh)*(g[n]+hh)-Esum0*Esum0*(1.-Kapa[n]*Kapa[n]-aa*aa));
L2__=4.*(z1*(g[n]+hh)-Esum0*Esum0*(p[n]+d1))*(z1*(g[n]+hh)-Esum0*Esum0*(p[n]+d1));
L2=L2_+L2__;
L3=Esum0*Esum0*Esum0*Esum0*((1.-Kapa[n]*Kapa[n]-aa*aa)*(1.-Kapa[n]*Kapa[n]-aa*aa)+4.*(p[n]+d1)*(p[n]+d1));
L=(L1+sqrt(L2))/sqrt(L3);
Eps_cog=C[n]*Esum0*xhp0[n]/(C[n]*Esum0*xhn0[n]);

As=abs(Eps_cog)*(L-sqrt(L*L-1.));

//         R_[1]=abs(A0)*abs(A0);
//         R_cogerUD_0[i]=R_[1];




  if (CheckBox67->Checked==false) goto m102ps;    // якщо пор. шару немає
//      Обчислення амплітуди від заданого профілю:
        for (int k=1; k<=km;k++)
{
//           YYs[k]=2.*M_PI/d*DD[k];
if (CheckBox18->Checked==false)
{
 YYs[k]=M_PI/Lambda/gamma0*DD[k]*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2.*sin(2.*tb);
}
if (CheckBox18->Checked==true)
{
 YYs[k]=M_PI/Lambda/gamma0*DD[k]*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2.*sin(2.*tb);
}
           YYs[k]=eta0pd+YYs[k];
            sqs=sqrt((YYs[k]/2.)*(YYs[k]/2.)-sigmasp0*sigmasn0*Esum[k]*Esum[k]);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta00<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs[k]/2.+sqs)/(sigmasn0*Esum[k]);
//        if (abs(x2s-As).lt.1E-10) goto 1001
            x1s=-(YYs[k]/2.-sqs)/(sigmasn0*Esum[k]);
            x3s=(x1s-As)/(x2s-As);
           expcs=exp(-2.*ssigma*Dl[k]);
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
//1001         continue
}
m102ps:
        R[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);
}
//if (RadioButton1->Checked==true) R_cogerTT[i]=R[1];
//if (RadioButton2->Checked==true) R_cogerTT[i]=(R[1]+C[2]*R[2])/(1+C[2]);
if (RadioButton1->Checked==true)  R_cogerTT[i]=R[1];
if (RadioButton55->Checked==true) R_cogerTT[i]=R[1];
if (RadioButton2->Checked==true)  R_cogerTT[i]=(R[1]+Monohr[1]*R[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) R_cogerTT[i]=(R[1]+Monohr[2]*R[2])/(1+Monohr[2]);
}
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerUD_kin(double *R_cogerTT) // функція для розрах. когер. КДВ (за Такагі-Топеном)
{
double R[3],Rd[3],Rk[3],Rkpl[3]; //,Rd_[MM],Rk_[MM],Rk__[MM];
double Lps; //,hpl0,DDpd[KM],LL[100];
int    zsuv,  kmpl;   //  NN[KM] ,
double   m00, dpl;
double d,AAA,AAApl,fi;
double fm, qm,Amsin, Amcos,Am;
//double FMS[KM];
  double *Rd_, *Rk_, *Rkpl_, *Rk__, *DDpd, *LL,*LLpl, *FMS;
  int *NN, *NNpl;
  Rd_   = new double[m1_teor+1];
  Rk_   = new double[m1_teor+1];
  Rkpl_ = new double[m1_teor+1];
  Rk__  = new double[m1_teor+1];
  DDpd  = new double[KM];
  LL    = new double[KM];
  LLpl  = new double[KM];
  FMS   = new double[KM];
  NN    = new int[KM];
  NNpl  = new int[KM];

complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i,eta00,eta0,DD0;
complex< double> As,YYs0;
complex< double> eta,sigmasp,sigmasn;
complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,ssigma,x1s,x2s,x3s,expcs;
    complex <double> cmplxi (0.,1.);

         //x0r0=0.;
         //x0r=0.;
         x0i0=ChiI0;
         x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);

if (KDV_lich==1) AAA=StrToFloat(Edit78->Text);
if (KDV_lich==2) AAA=StrToFloat(Edit85->Text);
if (KDV_lich==3) AAA=StrToFloat(Edit86->Text);
AAApl=StrToFloat(Edit87->Text);


 for (int k=1; k<=km; k++)
{
 LL[k]=0;
 for (int i=k+1; i<=km; i++) LL[k]=LL[k]+Dl[i];
}
//  EW=StrToFloat(Edit86->Text);

if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
        DD0=(apl-a)/a;
 for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1.)*(DD0+1.)-1. ;
      Lps=0.;
 for (int k=1; k<=km;k++) Lps=Lps+Dl[k] ;
      hpl0=hpl-Lps;
d=apl/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) NN[k]=Dl[k]/d;

kmpl=100;
 for (int k=1; k<=kmpl; k++)
{
 LLpl[k]=0;
 for (int i=k+1; i<=kmpl; i++) LLpl[k]=LLpl[k]+hpl/kmpl;
}
       dpl=apl/sqrt(h*h+k*k+l*l);
 for (int k=1; k<=kmpl; k++) NNpl[k]=hpl/dpl;


}

if (CheckBox31->Checked==false)
{
for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
d=a/sqrt(h*h+k*k+l*l);
for (int k=1; k<=km;k++) NN[k]=Dl[k]/d;
}

 fi=StrToFloat(Edit79->Text);
  zsuv=fi/ik_[KDV_lich];

      eta00=M_PI*x0i0*(1.+b_as)/(Lambda*gamma0);
      eta0=M_PI*x0i*(1.+b_as)/(Lambda*gamma0);
//      dpl=Lambda/2./sin(tb);

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
     eta0pd=-(eta00*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));
     eta=-(eta0*cmplxi+2.*M_PI*b_as*sin(2.*tb)*DeltaTeta[i]/(Lambda*gamma0));

       for (int n=nC1; n<=nC; n++)  
{
      sigmasp0=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn0=M_PI*xhn0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
      sigmasn=M_PI*xhn[n]*C[n]/(Lambda*sqrt(gamma0*gammah));

//      Обчислення амплітуди підкладки
/*          sqs=sqrt(eta0pd*eta0pd-4.*sigmasp0*sigmasn0);
          if (imag(sqs)<=0) sqs=-sqs;
          if (eta00<=0) sqs=-sqs;
          As=-(eta0pd+sqs)/(2.*sigmasn0);

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
{
//           YYs0=2*M_PI/d*DD0;
if (CheckBox18->Checked==false)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)+sin(psi)*cos(psi))*2*sin(2*tb);
}
if (CheckBox18->Checked==true)
{
 YYs0=M_PI/Lambda/gamma0*DD0*b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi))*2*sin(2*tb);
}
           YYs0=eta+YYs0;
           sqs=sqrt((YYs0/2.)*(YYs0/2.)-sigmasp*sigmasn);
            if (imag(sqs)<=0) sqs=-sqs;
            if (eta0<=0) sqs=-sqs;
            ssigma=sqs/cmplxi;
            x2s=-(YYs0/2.+sqs)/(sigmasn);
//            if (abs(x2s-As)<0.0000000001) goto m1001pl;
            x1s=-(YYs0/2.-sqs)/(sigmasn);
            x3s=(x1s-As)/(x2s-As);
            expcs=exp(-2.*ssigma*hpl0);
//m1012pl:
            As=(x1s-x2s*x3s*expcs)/(1.-x3s*expcs);
}
   */

 double aa,d1,hh,z1,K;
complex< double> L,L1,L2,L3,Eps_cog,L2_,L2__;

z1=sqrt(b_as)*sin(2.*tb)*DeltaTeta[i]/(C[n]*ModChiRH);
aa=0;                                                            //++++++++++++++++++++
K=2.*M_PI/Lambda;
m00=-MuDSsum[i]/(K*Esum0);                                      // і яке тут MuDS !!!!!!!!!?????????????????????
hh=m00*(1.+1./b_as)*sqrt(b_as)/(2.*C[n]*ModChiRH);
d1=0;                                                            //++++++++++++++++++++
//C=1;
//E=1;

//p=ModChiIH/ModChiRH;

L1=z1*z1+(g[n]+hh)*(g[n]+hh);
L2_=(z1*z1-(g[n]+hh)*(g[n]+hh)-Esum0*Esum0*(1.-Kapa[n]*Kapa[n]-aa*aa))*(z1*z1-(g[n]+hh)*(g[n]+hh)-Esum0*Esum0*(1.-Kapa[n]*Kapa[n]-aa*aa));
L2__=4.*(z1*(g[n]+hh)-Esum0*Esum0*(p[n]+d1))*(z1*(g[n]+hh)-Esum0*Esum0*(p[n]+d1));
L2=L2_+L2__;
L3=Esum0*Esum0*Esum0*Esum0*((1.-Kapa[n]*Kapa[n]-aa*aa)*(1.-Kapa[n]*Kapa[n]-aa*aa)+4.*(p[n]+d1)*(p[n]+d1));
L=(L1+sqrt(L2))/sqrt(L3);
Eps_cog=C[n]*Esum0*xhp0[n]/(C[n]*Esum0*xhn0[n]);

As=abs(Eps_cog)*(L-sqrt(L*L-1.));



if (CheckBox31->Checked==true)    // Обчислення амплітуди плівки
{
          fm=0.;
          FMS[0]=0.;
          Amsin=0.;
          Amcos=0.;
          for (int k=1; k<=kmpl;k++)   //         do k=2,km+1
  {
           FMS[k]=FMS[k-1]+fm;
double DeltaTetaDD_=1./b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
            fm=2.*M_PI*(1./DeltaTetaDD_*DeltaTeta[i]-(1./DeltaTetaDD_*DeltaTeta[i]+1.)*DD0);
 if (fabs(fm)<1e-15) fm=1e-15;
            qm=((NNpl[k]+1.)/2.)*fm+NNpl[k]*FMS[k];
///if (RadioButton17->Checked==true) Am=sin(NN[k]*fm/2.)/sin(fm/2.);
//if (CheckBox5->Checked==true)Am=sin(NN*fm/2.)/sin(fm/2.)*exp(-Mu0*dl/sin(tb-psi)*(km-k));
//if (RadioButton18->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[km]);
//if (RadioButton19->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[km])*Esum[k];
           Am=sin(NNpl[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LLpl[k])*AAApl;
           Amcos=Amcos+Am*cos(qm);
            Amsin=Amsin+Am*sin(qm);
  }
 complex< double> Aks (Amcos+1, Amsin);
      Rkpl[n]=abs(Aks)*abs(Aks)/40000000.*AAA ;
}



//      Обчислення амплітуди від заданого профілю (kinemat):
if (CheckBox67->Checked==true)     // якщо пор. шар є
{
          fm=0.;
          FMS[0]=0.;
          Amsin=0.;
          Amcos=0.;
for (int k=1; k<=km;k++)   //         do k=2,km+1
  {
           FMS[k]=FMS[k-1]+fm;
double DeltaTetaDD_=1./b_as*(cos(psi)*cos(psi)*tan(tb)-sin(psi)*cos(psi));
            fm=2.*M_PI*(1./DeltaTetaDD_*DeltaTeta[i]-(1./DeltaTetaDD_*DeltaTeta[i]+1.)*DDpd[k]);
 if (fabs(fm)<1e-15) fm=1e-15;
            qm=((NN[k]+1.)/2.)*fm+NN[k]*FMS[k];
if (RadioButton17->Checked==true) Am=sin(NN[k]*fm/2.)/sin(fm/2.);
//if (CheckBox5->Checked==true)Am=sin(NN*fm/2.)/sin(fm/2.)*exp(-Mu0*dl/sin(tb-psi)*(km-k));
if (RadioButton18->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k]);
if (RadioButton19->Checked==true)Am=sin(NN[k]*fm/2.)/sin(fm/2.)*exp(-Mu0*LL[k])*Esum[k];
            Amcos=Amcos+Am*cos(qm);
            Amsin=Amsin+Am*sin(qm);
  }
 complex< double> Aks (Amcos+1, Amsin);
      Rk[n]=abs(Aks)*abs(Aks)/40000000.*AAA ;
}
      Rd[n]=abs(xhp0[n]/xhn0[n])*abs(As)*abs(As);
//      R[n]=Rd[n]+Rk[n];
}
if (CheckBox31->Checked==true)
  {
//  if (RadioButton1->Checked==true) Rkpl_[i]=Rkpl[1];
//  if (RadioButton2->Checked==true) Rkpl_[i]=(Rkpl[1]+C[2]*Rkpl[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rkpl_[i]=Rkpl[1];
if (RadioButton55->Checked==true) Rkpl_[i]=Rkpl[1];
if (RadioButton2->Checked==true)  Rkpl_[i]=(Rkpl[1]+Monohr[1]*Rkpl[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rkpl_[i]=(Rkpl[1]+Monohr[2]*Rkpl[2])/(1+Monohr[2]);
  }
if (CheckBox67->Checked==true)
  {
//  if (RadioButton1->Checked==true) Rk_[i]=Rk[1];
//  if (RadioButton2->Checked==true) Rk_[i]=(Rk[1]+C[2]*Rk[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rk_[i]=Rk[1];
if (RadioButton55->Checked==true) Rk_[i]=Rk[1];
if (RadioButton2->Checked==true)  Rk_[i]=(Rk[1]+Monohr[1]*Rk[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rk_[i]=(Rk[1]+Monohr[2]*Rk[2])/(1+Monohr[2]);
  }
//if (RadioButton1->Checked==true) Rd_[i]=Rd[1];
//if (RadioButton2->Checked==true) Rd_[i]=(Rd[1]+C[2]*Rd[2])/(1+C[2]);
if (RadioButton1->Checked==true)  Rd_[i]=Rd[1];
if (RadioButton55->Checked==true) Rd_[i]=Rd[1];
if (RadioButton2->Checked==true)  Rd_[i]=(Rd[1]+Monohr[1]*Rd[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) Rd_[i]=(Rd[1]+Monohr[2]*Rd[2])/(1+Monohr[2]);
}

if (CheckBox67->Checked==true)
  {
  for (int i=0; i<=m1_teor; i++) Rk__[i]=Rk_[i];
  for (int i=0; i<=m1_teor; i++)
    {
      if (i-zsuv<0 || i+zsuv>m1_teor) Rk_[i]=Rk__[i];
      else  Rk_[i]=Rk__[i-zsuv];
      R_cogerTT[i]=Rd_[i]+Rk_[i];
    }
  }
if (CheckBox67->Checked==false) for (int i=0; i<=m1_teor; i++) R_cogerTT[i]=Rd_[i];
if (CheckBox31->Checked==true) for (int i=0; i<=m1_teor; i++) R_cogerTT[i]=R_cogerTT[i]+Rkpl_[i];

  delete Rd_, Rk_,Rkpl_, Rk__, DDpd, LL, LLpl, FMS, NN, NNpl;
}

//---------------------------------------------------------------------------
void TForm1::RozrachKoger_kin_Mol(double *R_cogerTT) // функція для розрах. когер. КДВ (за УДТ)
//void __fastcall TForm1::Button22Click(TObject *Sender)   //Розрахунок когерентної складової (може колись буде)
{
complex< double> Xjkin ;
complex< double> XLkin ;
complex <double> cmplxi (0.,1.);
double zj[KM] ;
double RcohLS[3] ;
double RcohL[3]   ;
double Zj_1 ;
double RcohS[3] ;
double Rcoh[3] ;
double z;
double omegaj, fij, Muj, Nuj, Lambdaj_, LambdaBj, psij, betaj, MuLj, ksi ;
double *MuDSj, *Fabsj;
  MuDSj  = new double[KM];
  Fabsj  = new double[KM];

for (int i=0; i<=m1; i++)          //????? ????
{

for (int n=nC1; n<=nC; n++)               //????? ???????????
{
Rcoh[n]=0;
//z=DeltaTeta[i]*Sin2Teta/(C[n]*ModChiRH)*sqrt(b_as);

for (int k=0; k<=km;k++)   MuDSj[k]=0 ;
for (int k=0; k<=km;k++)                   //
{
Fabsj[k]=1;
zj[k]=0;
for (int jk=k+1; jk<=km;jk++)
{
MuLj=(Mu0+MuDSj[jk])*(b_as+1)/(2*gamma0);               //
Fabsj[k]=Fabsj[k]*exp(-MuLj*Dl[jk]);
//Fabsj[k]=1;
zj[k] += Dl[jk] ;                        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}
//Memo8->Lines->Add(FloatToStr(MuLj)+'\t'+FloatToStr(zj[k])+'\t'+FloatToStr(Fabsj[k]));
}
XLkin=0 ;
for (int k=1; k<=km;k++)                   //????? ????????, XLkin ;
{
Esum[k]=1 ;                                 // Ej=exp(-LHj)   !!!!!!!!!!!!!!!!!!!!!!!

omegaj=Lambda*fabs(gammah)/(Dl[k]*Sin2Teta) ;
fij=2*M_PI*(DeltaTeta[i]-DeltaTetaDD[k])/omegaj ;
psij=fij*zj[k]/Dl[k] ;
//Memo8->Lines->Add(FloatToStr(fij)+'\t'+FloatToStr(psij)+'\t'+FloatToStr(zj[k])+'\t'+FloatToStr(Fabsj[k]));

//MuDSj[k]=MuDSpr[i] ;

Muj=(Mu0+MuDSj[k])*(b_as+1)/(2*gamma0);
Nuj=-Kapa[1]*fij+(1-pow(Kapa[1],2.))*Muj*Dl[k] ;

Lambdaj_=Lambda*sqrt(gamma0*fabs(gammah))/(C[1]*ModChiRH*(1.+pow(Kapa[1],2.))) ;
LambdaBj=Lambdaj_/(2*M_PI) ;

betaj=exp(-Muj*Dl[k]) ;
//Memo8->Lines->Add(FloatToStr(Dl[k])+'\t'+FloatToStr(Lambdaj_)+'\t'+FloatToStr(psij)+'\t'+FloatToStr(Nuj)+'\t'+FloatToStr(Muj));
//Memo8->Lines->Add(FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k])+'\t'+FloatToStr(fij)+'\t'+FloatToStr(psij)+'\t'+FloatToStr(Nuj)+'\t'+FloatToStr(Muj));
//Xjkin = ksi*sqrt(b_as)*exp(cmplxi*psij)*(Dl[k]*Esum[k]*fij-cmplxi*Nuj)*(exp(cmplxi*fij)-betaj)/(2.*LambdaBj*(pow(fij,2.)+pow(Nuj,2))) ;      //(32) i-???
  ksi=1 ;    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Xjkin = 1e-5*ksi*sqrt(b_as)*exp(cmplxi*psij)*(Dl[k]*Esum[k]*fij-cmplxi*Nuj)*(exp(cmplxi*fij)-betaj)/(2.*LambdaBj*(fij*fij+Nuj*Nuj)) ;
  XLkin += Fabsj[k]*Xjkin ;                   //(28)
//Memo8->Lines->Add(FloatToStr(real(Xjkin))+'\t'+FloatToStr(imag(Xjkin))+'\t'+FloatToStr(real(XLkin))+'\t'+FloatToStr(imag(XLkin))+'\t'+FloatToStr(Fabsj[k]));

}


//RcohLS[n]=2.*Fabsj[0]*real(XLkin*X0z) ;     // (27)   ???
RcohLS[n]=0 ;
//RcohL[n]=pow(fabs(XLkin),2.)/b_as ;          // (26)
RcohL[n]=abs(XLkin)*abs(XLkin)/b_as ;
//Rcohs[n]=Fabsj[0]*pow(fabs(X0),2.) ;         // (20)
RcohS[n]=0 ;
Rcoh[n]=RcohS[n]+RcohL[n]+RcohLS[n] ;               // (19)

}
//        R_cogerTT[i]=(Rcoh[1]+C[2]*Rcoh[2])/(1+C[2]);
        R_cogerTT[i]=Rcoh[1];
}
 for (int k=1; k<=m1;k++) Memo1->Lines->Add(FloatToStr(R_cogerTT[k]));
//for (int k=1; k<=km;k++) Memo2->Lines->Add(FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k]));

/*for (int i=0; i<=m1; i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series35->AddXY(DeltaTeta1,R_cogerTT[i],"",clBlue);
//Series36->AddXY(DeltaTeta1,Rcoh[2],"",clBlack);
}  */

delete MuDSj, Fabsj;
}

//---------------------------------------------------------------------------
void TForm1::RozrachKogerUD_din_Mol(double *R_cogerTT) // функція для розрах. когер. КДВ (за УДТ)
{
//double R[3];
double L,hpl0;   // DDpd[KM],
//  double *DDpd;
 // DDpd   = new double[KM];
//complex< double> YYs[KM];
complex< double> xhp0[3],xhp[3];
complex< double> xhn0[3],xhn[3];
double /*x0r0,x0r,*/ x0i0,x0i; //,eta00,eta0,DD0;
//complex< double> As,YYs0;
//complex< double> eta,sigmasn;
//complex< double> eta0pd,sigmasp0,sigmasn0;
complex< double> sqs,sigmasp,R; //,ssigma,x1s,x2s,x3s,expcs;
double pmut, MuHH, Mu00;
double  DeltaChi0H,DeltaChiH0;
complex< double> DeltaChiHH,DeltaChi00,alfa0;
if (CheckBox76->Checked==false)
  {
  DeltaChiHH=0;
  DeltaChi00=0;
  DeltaChi0H=0;
  DeltaChiH0=0;
  }
double alfa, A;
complex< double>  A1,dzeta,sigma,y,iasq,r[KM],r0[KM],t[KM],t0[KM],c1[KM],c2[KM],L_ext1[3];
double Rcoh[3] ;
double      *x0i_a;
  x0i_a  = new double[KM];
    complex <double> cmplxi (0.,1.);
complex< double>  xhp_a[3][KM],xhn_a[3][KM],PHH,P00;

         //x0r0=0;
         //x0r=0;
         //x0i0=ChiI0;
         //x0i=ChiI0pl;
      xhp0[1]=(ReChiRH+cmplxi* ReChiIH[1]);   //для  центр.-сим. крист.
      xhn0[1]=(ReChiRH+cmplxi* ReChiIH[1]);
      xhp0[2]=(ReChiRH+cmplxi* ReChiIH[2]);   //для  центр.-сим. крист.
      xhn0[2]=(ReChiRH+cmplxi* ReChiIH[2]);
      //   complex< double> xo0 (x0r0, ChiI0);
      xhp[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);   //для  центр.-сим. крист.
      xhn[1]=(ReChiRHpl+cmplxi* ReChiIHpl[1]);
      xhp[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);   //для  центр.-сим. крист.
      xhn[2]=(ReChiRHpl+cmplxi* ReChiIHpl[2]);
      //   complex< double> xo (x0r, ChiI0pl);

for (int k=1; k<=km;k++)
{
//         x0i0=ChiI0_a[k];
//      x0i_a[k]=ChiI0_a[k];
      xhp_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);   //для  центр.-сим. крист.
      xhn_a[1][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[1][k]);
      xhp_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);   //для  центр.-сим. крист.
      xhn_a[2][k]=(ReChiRH_a[k]+cmplxi* ReChiIH_a[2][k]);
//      eta0_a[k]=M_PI*x0i_a[k]*(1+b_as)/(Lambda*gamma0);
}

if (CheckBox3->Checked==true)
  for (int k=1; k<=km;k++) Esum[k]=StrToFloat(Edit131->Text);
if (CheckBox73->Checked==true) Esum0pl=StrToFloat(Edit301->Text);
if (CheckBox74->Checked==true) Esum0=StrToFloat(Edit326->Text);

double K=2*M_PI/Lambda;

if (CheckBox31->Checked==true)
{
// Перерахунок профiлю з вiдносних одиниць вiдносно плiвки у вiдноснi одиницi вiдносно пiдкладки
 //       DD0=(apl-a)/a;
// for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1)*(DD0+1)-1 ;
      L=0;
 for (int k=1; k<=km;k++) L=L+Dl[k];
      hpl0=hpl-L;
}
if (CheckBox31->Checked==false)
{
//for (int k=1; k<=km;k++) DDpd[k]=DD[k] ;
}

// Обчислення теор. когер. КДВ

for (int i=0; i<=m1_teor; i++)
{
       for (int n=nC1; n<=nC; n++)
{

//      Обчислення амплітуди підкладки
dzeta=sqrt(C[n]*Esum0*xhp0[n]+DeltaChiH0)/sqrt(C[n]*Esum0*xhn0[n]+DeltaChi0H);
alfa=DeltaTeta[i]*sin(2*tb);
alfa0=(ChiI0+DeltaChiHH+(ChiI0+DeltaChiHH)/b_as)/2.;
   //sigma=sqrt((C[n]*Esum0*xhp0[n]+DeltaChiH0)*(C[n]*Esum0*xhn0[n]+DeltaChi0H));
sigma=C[n]*Esum0*xhp0[n];  // !!! DeltaChi0H,Р0=0. Хоча sqrt((-3-I*2)*(-3-I*2))=3 + 2 I а не -3-I*2
y=-(alfa0*cmplxi+alfa)*sqrt(b_as)/sigma  ;       //  y=eta0pd/(sigmasp0*2.) ;
sqs=sqrt(y*y-1.);
sigmasp=M_PI*xhp0[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
if (imag(sqs*2.*sigmasp)<=0) sqs=-sqs;
R=-(y-sqs);      // !!!! Напевне ще спереду (sqrt b_as)-1
//R=sqrt(dzeta*b_as)*R;

Memo9->Lines->Add(FloatToStr(real(r[k]))+'\t'+FloatToStr(imag(r[k]))+'\t'+FloatToStr(real(t[k]))+'\t'+FloatToStr(imag(t[k])));

//      Обчислення амплітуди плівки
if (CheckBox31->Checked==true)
  {
  for (int k=1; k<=1;k++)      // бо плівка одна
    {
    dzeta=sqrt(C[n]*Esum0pl*xhp[n]+DeltaChiH0)/sqrt(C[n]*Esum0pl*xhn[n]+DeltaChi0H);
    alfa=(DeltaTeta[i]-DeltaTetaDDpl)*sin(2*tb);
    alfa0=(ChiI0pl+DeltaChiHH+(ChiI0pl+DeltaChiHH)/b_as)/2.;
    //sigma=sqrt((C[n]*Esum[k]*xhp0[n]+DeltaChiH0)*(C[n]*Esum[k]*xhn0[n]+DeltaChi0H));
    sigma=C[n]*Esum0pl*xhp[n];
    y=-(alfa0*cmplxi+alfa)*sqrt(b_as)/sigma;
    L_ext1[n]=Lambda*sqrt(gamma0*fabs(gammah))/sigma;
    A=0.;                       //!!!!!!!!!!!
    A1=M_PI*hpl0/L_ext1[n];
    sqs=sqrt(y*y-1.);
    sigmasp=M_PI*xhp[n]*C[n]/(Lambda*sqrt(gamma0*gammah));
    if (imag(sqs*2.*sigmasp)<=0) sqs=-sqs;
    iasq=cmplxi*A1*sqs;

    c1[k]=sqrt(b_as*dzeta)*(y-sqs);   //c1[k]=sqrt(b_as*dzeta)*(y-sqrt(y*y-1.));
    c2[k]=sqrt(b_as*dzeta)*(y+sqs);   //c2[k]=sqrt(b_as*dzeta)*(y+sqrt(y*y-1.));
    r0[k]=(exp(iasq)-exp(-iasq))/(c1[k]*exp(iasq)-c2[k]*exp(-iasq));
    r[k]=r0[k]*sqrt(dzeta);
    t0[k]=(c1[k]-c2[k])/(c1[k]*exp(iasq)-c2[k]*exp(-iasq));
    t[k]=t0[k]*exp((-cmplxi*K*(ChiI0pl+DeltaChi00)/(2.*gamma0)*hpl0-cmplxi*A*y)*1.); //може в показнику ще 2 має бути, однак на КДВ це не впливає.
    R=(r[k]+R*(t[k]*t0[k]-r[k]*r0[k]))/(1.-r0[k]*R);
    //R=sqrt(dzeta*b_as)*R;
    }
  }


//      Обчислення амплітуди від заданого профілю:
if (CheckBox67->Checked==false) goto m102ps;    // якщо пор. шару немає


for (int k=1; k<=km;k++)
{
  if (CheckBox76->Checked==true)
    {
    pmut=1;
    MuHH=MuDSsum_dl[i][k][n]*pmut;
    Mu00=b_as*MuHH;
    PHH=-MuHH/K;
    P00=-Mu00/K;
    DeltaChiHH=PHH-cmplxi*MuHH/K;
    DeltaChi00=P00-cmplxi*Mu00/K;
//Memo8->Lines->Add(FloatToStr(real(DeltaChi00))+'\t'+FloatToStr(imag(DeltaChi00))+'\t'+FloatToStr(MuHH)+'\t'+FloatToStr(Mu00)) ;
//Memo8->Lines->Add(FloatToStr(real(DeltaChi00))+'\t'+FloatToStr(imag(DeltaChi00))+'\t'+FloatToStr(real(DeltaChiHH))+'\t'+FloatToStr(imag(DeltaChiHH))) ;
    }

dzeta=sqrt(C[n]*Esum[k]*xhp_a[n][k]+DeltaChiH0)/sqrt(C[n]*Esum[k]*xhn_a[n][k]+DeltaChi0H);
alfa=(DeltaTeta[i]-DeltaTetaDD[k])*sin(2*tb);
alfa0=(ChiI0_a[k]+DeltaChiHH+(ChiI0_a[k]+DeltaChiHH)/b_as)/2.;
//sigma=sqrt((C[n]*Esum[k]*xhp0[n]+DeltaChiH0)*(C[n]*Esum[k]*xhn0[n]+DeltaChi0H));
sigma=C[n]*Esum[k]*xhp_a[n][k];
y=-(alfa0*cmplxi+alfa)*sqrt(b_as)/sigma;
L_ext1[n]=Lambda*sqrt(gamma0*fabs(gammah))/sigma;
A=0.;                        //!!!!!!!!!!!
A1=M_PI*Dl[k]/L_ext1[n];
sqs=sqrt(y*y-1.);
sigmasp=M_PI*xhp_a[n][k]*C[n]/(Lambda*sqrt(gamma0*gammah));
if (imag(sqs*2.*sigmasp)<=0) sqs=-sqs;
iasq=cmplxi*A1*sqs;
//iasq=cmplxi*A1*sqs-cmplxi*A1*(ChiI0+DeltaChiHH)/sigma*(1-b_as)/sqrt(b_as)-cmplxi*A1*DeltaTeta[i]*sin(2*tb);

c1[k]=sqrt(b_as*dzeta)*(y-sqs);   //c1[k]=sqrt(b_as*dzeta)*(y-sqrt(y*y-1.));
c2[k]=sqrt(b_as*dzeta)*(y+sqs);   //c2[k]=sqrt(b_as*dzeta)*(y+sqrt(y*y-1.));
r0[k]=(exp(iasq)-exp(-iasq))/(c1[k]*exp(iasq)-c2[k]*exp(-iasq));
r[k]=r0[k]*sqrt(dzeta);
t0[k]=(c1[k]-c2[k])/(c1[k]*exp(iasq)-c2[k]*exp(-iasq));
t[k]=t0[k]*exp((-cmplxi*K*(ChiI0_a[k]+DeltaChi00)/(2.*gamma0)*Dl[k]-cmplxi*A*y)*1.); //може в показнику ще 2 має бути, однак на КДВ це не впливає.
R=(r[k]+R*(t[k]*t0[k]-r[k]*r0[k]))/(1.-r0[k]*R);
//R=sqrt(dzeta*b_as)*R;

//Memo4->Lines->Add(FloatToStr(real(sqs*2.*sigmasp))+'\t'+FloatToStr(imag(sqs*2.*sigmasp))+'\t'+FloatToStr(real(iasq))+'\t'+FloatToStr(imag(iasq))+'\t'+FloatToStr(real(exp(iasq)))+'\t'+FloatToStr(imag(exp(iasq)))+'\t'+FloatToStr(real(exp(-iasq)))+'\t'+FloatToStr(imag(exp(-iasq)))) ;
//Memo8->Lines->Add(FloatToStr(real(c1[k]))+'\t'+FloatToStr(imag(c1[k]))+'\t'+FloatToStr(real(c2[k]))+'\t'+FloatToStr(imag(c2[k]))) ;
//Memo9->Lines->Add(FloatToStr(real(r[k]))+'\t'+FloatToStr(imag(r[k]))+'\t'+FloatToStr(real(t[k]))+'\t'+FloatToStr(imag(t[k]))) ;
//Memo3->Lines->Add(FloatToStr(real(r0[k]))+'\t'+FloatToStr(imag(r0[k]))+'\t'+FloatToStr(real(t0[k]))+'\t'+FloatToStr(imag(t0[k]))) ;
//Memo8->Lines->Add(FloatToStr(real(iasq))+'\t'+FloatToStr(imag(iasq))+'\t'+FloatToStr(555)) ;
}

m102ps:
//Rcoh[n]=abs(xhp0[n]/xhn0[n])*abs(R)*abs(R) ;
Rcoh[n]=abs(R)*abs(R) ;
}
//if (RadioButton1->Checked==true) R_cogerTT[i]=Rcoh[1];
//if (RadioButton2->Checked==true) R_cogerTT[i]=(Rcoh[1]+C[2]*Rcoh[2])/(1+C[2]);
if (RadioButton1->Checked==true)  R_cogerTT[i]=Rcoh[1];
if (RadioButton55->Checked==true) R_cogerTT[i]=Rcoh[1];
if (RadioButton2->Checked==true)  R_cogerTT[i]=(Rcoh[1]+Monohr[1]*Rcoh[2])/(1+Monohr[1]);
if (RadioButton56->Checked==true) R_cogerTT[i]=(Rcoh[1]+Monohr[2]*Rcoh[2])/(1+Monohr[2]);
}
//  delete DDpd;
//Memo9->Lines->Add( "RozrachKogerTT 3 пройшло");
delete  x0i_a;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//void __fastcall TForm1::Button14Click(TObject *Sender) // згортка з апаратною функцією
//{     Zgortka();     }

void TForm1::Zgortka()
{
double ik, II;
//double  R_vse_utochn[MM], R_vse_dTeta[MM],R_vseZ_dTeta[MM],R_vseZ_utochn[MM];
//double  POO_dTeta1[MM],POO_dTeta[MM],POO1[MM],DeltaTeta_utochn[MM];
//double POO [MM], R_vse[MM], R_vseZ[MM];
double  koefCKV1, koefCKV2, koefCKV3;
double  *R_vse_utochn, *R_vse_dTeta,*R_vseZ_dTeta,*R_vseZ_utochn;
double  *POO_dTeta1,*POO_dTeta,*POO1,*DeltaTeta_utochn;
double *POO , *R_vse, *R_vseZ;
  int nom10,nom11,nom12,nom20,nom21,nom22,nom30,nom31,nom32;
int  nsd,koef_dTeta ,ep, ek,  op,  ok,jp,jk ;
int m1_bez_utochn, m10_bez_utochn ;
int m1_teor_ut=StrToInt(Edit382->Text);

  DeltaTeta_utochn  = new double[m1_teor+1+m1_teor_ut];
  R_vse_utochn  = new double[m1_teor+1+m1_teor_ut];
  R_vse_dTeta   = new double[m1_teor+1+m1_teor_ut];
  R_vseZ_dTeta  = new double[m1_teor+1+m1_teor_ut];
  R_vseZ_utochn = new double[m1_teor+1+m1_teor_ut];
  POO_dTeta1    = new double[m1_teor+1+m1_teor_ut];
  POO_dTeta     = new double[m1_teor+1+m1_teor_ut];
  POO1          = new double[m1_teor+1+m1_teor_ut];
  POO           = new double[m1_teor+1+m1_teor_ut];
  R_vse         = new double[m1_teor+1+m1_teor_ut];
  R_vseZ        = new double[m1_teor+1+m1_teor_ut];

// Сума когер. і дифузної:
for (int i=0; i<=m1_teor; i++)
{
if (RadioButton5->Checked==true) R_vse[i]=R_dif_[i][KDV_lich]+R_cogerTT_[i][KDV_lich];
//if (RadioButton5->Checked==true) R_vse[i]=R_dif[i]+R_cogerTT[i];
if (RadioButton6->Checked==true) R_vse[i]=R_cogerTT_[i][KDV_lich];
}

if (fitting==0 || (fitting==1 && vse==2) || (fitting==10 && vse==2))
{
for (int i=0; i<=m1_teor; i++) R_vse_[i][KDV_lich]=R_vse[i];
}

if (KDV_lich==1) ik=ik_[1];
if (KDV_lich==2) ik=ik_[2];
if (KDV_lich==3) ik=ik_[3];

/*  if (CheckBox41->Checked==true) // розрах. тільки в межах СКВ (при набл. теж)
    {
    if (KDV_lich==1) {m1=kskvi1-nskvi1; m10=-nskvi1;}
    if (KDV_lich==2) {m1=kskvi2-nskvi2; m10=-nskvi2;}
    if (KDV_lich==3) {m1=kskvi3-nskvi3; m10=-nskvi3;}
Memo9->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(555));
Memo9->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(666));
    }     */

if (m10<m10z && fitting==0)
{
if (KDV_lich==1) Label179->Caption="m10<m10z";
if (KDV_lich==2) Label180->Caption="m10<m10z";
if (KDV_lich==3) Label181->Caption="m10<m10z";
}
if (KDV_lich==1) for (int i=0; i<=m1z; i++) POO[i]=POk2d[i][1];
if (KDV_lich==2) for (int i=0; i<=m1z; i++) POO[i]=POk2d[i][2];
if (KDV_lich==3) for (int i=0; i<=m1z; i++) POO[i]=POk2d[i][3];

 koef_dTeta=1;
 ep=-m10;
 ek=m1-m10;
 op=-m10z;      // AO any ooi?iaia i?e ?ic?aooieo
 ok=m1z-m10z;   // AO any ooi?iaia i?e ?ic?aooieo
 jp=ep-ok;
 jk=ek-op;
m10_teor=-jp;        // =m10+(m1z-m10z)
m1_teor=-jp+jk;      // =m10+(m1z-m10z)+m1-m10-(-m10z)=m1+m1z      =260
/* if (CheckBox41->Checked==true)
 {
 double gggg [2000];
	 for (int i=0; i<=m1; i++) gggg[i]=intIk2d[i][1];
         for (int i=0; i<=m1; i++)  intIk2d[i][1]=gggg[i+jp];
  }      */


/*  jp=-m10_teor;             // або таке задання задом наперед // 260/275
 jk=m1_teor-m10_teor;
 op=-m10z;
 ok=m1z-m10z;
 ep=jp+ok;     //бо АФ вся уточнена
 ek=jk+op;     */
 nsd=-jp;
for (int i=m1z; i>=0; i--) POO1[i+nsd-m10z]=POO[i];
//for (int i=0; i<nsd-m10z; i++) POO1[i]=0;
//for (int i=nsd-m10z+1; i<=m1z+nsd-m10z; i++) POO1[i]=0;

//Memo2->Lines->Add("Zgortka");
/*Memo2->Lines->Add(IntToStr(m1_teor)+'\t'+IntToStr(m10_teor)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(ep)+'\t'+IntToStr(ek)+'\t'+IntToStr(op)+'\t'+IntToStr(ok));
Memo2->Lines->Add(IntToStr(jp)+'\t'+IntToStr(jk)+'\t'+IntToStr(nsd)+'\t'+IntToStr(555));
*/
for (int j=ep; j<=ek; j++)                     //   do 13 j=ep,ek+jkd
{
     R_vseZ[j+nsd]=0;                            //   PZ(j)=0.
for (int i=op; i<=ok; i++)          //   do i=op,ok
{
  R_vseZ[j+nsd]= R_vseZ[j+nsd]+ R_vse[j-i+nsd]*POO1[i+nsd];    //   PZ(j)=PZ(j)+PR(j-i)*PO(i);
}
}

if (CheckBox53->Checked==true) Memo8-> Lines->Add("Згортка порахована!");

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
for (int i=0; i<=m1_teor; i++) DeltaTeta_utochn[i]=DeltaTeta[i];
for (int i=0; i<=m1_teor; i++) R_vse_utochn[i]= R_vse[i];  // ця змінна не обовязкова, однак без неї стрибки на згорнутій КДВ
if (KDV_lich==1) koef_dTeta=StrToInt(Edit152->Text);
if (KDV_lich==2) koef_dTeta=StrToInt(Edit385->Text);
if (KDV_lich==3) koef_dTeta=StrToInt(Edit391->Text);

 ep=-m10;
 ek=m1-m10;
 op=-m10z;      // АФ вся уточнена при розрахунку
 ok=m1z-m10z;   // АФ вся уточнена при розрахунку
 jp=ep-ok;
 jk=ek-op;
 nsd=-jp;
m10_bez_utochn= -jp;
m1_bez_utochn =-jp+jk;

/*Memo2->Lines->Add("Zgortka 0");
Memo2->Lines->Add(IntToStr(m1_teor)+'\t'+IntToStr(m10_teor)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(ep)+'\t'+IntToStr(ek)+'\t'+IntToStr(op)+'\t'+IntToStr(ok));
Memo2->Lines->Add(IntToStr(jp)+'\t'+IntToStr(jk)+'\t'+IntToStr(nsd)+'\t'+IntToStr(666));
Memo2->Lines->Add(IntToStr(m1_bez_utochn)+'\t'+IntToStr(m10_bez_utochn)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
*/
for (int i=0; i<=m1_bez_utochn; i++) DeltaTeta[i]=(-m10_bez_utochn+i)*ik*M_PI/(3600*180);

for (int i=0; i<=m1_bez_utochn; i++)   // 330        //Створює  теор. КДВ без уточнення
{
   if (i<=nkoef_dTetai)  // 150
   {
   R_vse_dTeta[i]=R_vse[i];
   }
  if (i>nkoef_dTetai && i<kkoef_dTetai)   // 150..165
   {
   R_vse_dTeta[i]=R_vse[nkoef_dTetai+koef_dTeta*(i-nkoef_dTetai)];
   }
    if (i>=kkoef_dTetai) // 165
   {
   R_vse_dTeta[i]=R_vse[(i-kkoef_dTetai)+nkoef_dTetai+koef_dTeta*(kkoef_dTetai-nkoef_dTetai)];
   }
}

 op=-m10z/koef_dTeta;      //-35
 ok=(m1z-m10z)/koef_dTeta; // 35
/*Memo2->Lines->Add("Zgortka без уточн");
Memo2->Lines->Add(IntToStr(m1_teor)+'\t'+IntToStr(m10_teor)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(m1)+'\t'+IntToStr(m10)+'\t'+IntToStr(m1z)+'\t'+IntToStr(m10z));
Memo2->Lines->Add(IntToStr(ep)+'\t'+IntToStr(ek)+'\t'+IntToStr(op)+'\t'+IntToStr(ok));
Memo2->Lines->Add(IntToStr(jp)+'\t'+IntToStr(jk)+'\t'+IntToStr(nsd)+'\t'+IntToStr(666));
*/
double Ap_sum=0.;   //Створюємо POO_dTeta[i], тобто АФ без уточнення
for (int i=0; i<=m1z/koef_dTeta; i++) POO_dTeta1[i]=POO[koef_dTeta*i];
for (int i=0; i<=m1z/koef_dTeta; i++)  Ap_sum=Ap_sum+POO_dTeta1[i];
for (int i=0; i<=m1z/koef_dTeta; i++)  POO_dTeta1[i]=POO_dTeta1[i]/Ap_sum;
for (int i=m1z/koef_dTeta; i>=0; i--) POO_dTeta[i+nsd-m10z/koef_dTeta]=POO_dTeta1[i];

// Згортка без уточнення
for (int j=ep; j<=ek; j++)                     //   do 13 j=ep,ek+jkd
{
     R_vseZ_dTeta[j+nsd]=0;                            //   PZ(j)=0.
for (int i=op; i<=ok; i++)          //   do i=op,ok
{
  R_vseZ_dTeta[j+nsd]= R_vseZ_dTeta[j+nsd]+ R_vse_dTeta[j-i+nsd]*POO_dTeta[i+nsd];    //   PZ(j)=PZ(j)+PR(j-i)*PO(i);
}
}

// Зєднує згортки для уточненої КДВ R_vseZ[i]
for (int i=0; i<=m1_teor; i++)  // 345 цикл по R_vse[i] (тобто з уточненням)
{
   if (i<=nkoef_dTetai)         // 150
   {
   R_vseZ_utochn[i]=R_vseZ_dTeta[i];
   }
   if (i>nkoef_dTetai && i<nkoef_dTetai+(kkoef_dTetai-nkoef_dTetai)*koef_dTeta)
   {
   R_vseZ_utochn[i]=R_vseZ[i];
   }
   if (i>=nkoef_dTetai+(kkoef_dTetai-nkoef_dTetai)*koef_dTeta)     // 180
   {
   R_vseZ_utochn[i]=R_vseZ_dTeta[i-(kkoef_dTetai-nkoef_dTetai)*(koef_dTeta-1)];
   }
}
}



if (CheckBox20->Checked==true)  // нормування на 1 поч. (стосується тільки без уточнення)
{
double  PRmax=0;
double  PZmax=0;
for (int j=0; j<=m1; j++)
{
  if (R_vse[j]>PRmax) PRmax=R_vse[j] ;
  if (R_vseZ[j]>PZmax) PZmax=R_vseZ[j];
}
for (int j=0; j<=m1; j++)
{
R_vse[j]=R_vse[j]/PRmax;
R_vseZ[j]=R_vseZ[j]/PZmax;
}
}                      // нормування на 1 кін.

if (CheckBox53->Checked==true) Memo8-> Lines->Add("Уточнення пораховане!");

if (fitting==0)
{

if (number_KDV==1)
	{
if (CheckBox19->Checked==false)
{
if (CheckBox51->Checked==true)  for (int i=0; i<=m1_teor; i++) Series14->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vse_utochn[i],"",clRed);
if (CheckBox51->Checked==false) for (int i=0; i<=m1_teor; i++) Series14->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed);
//for (int i=0; i<=m1_bez_utochn; i++)Series6->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed);   //низ
if (CheckBox51->Checked==true)  for (int i=0; i<=m1_bez_utochn; i++) Series6->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse_dTeta[i],"",clRed);  //низ
if (CheckBox51->Checked==false) for (int i=0; i<=m1_teor; i++) Series6->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed);  //низ
}
if (CheckBox51->Checked==true)  for (int i=0; i<=m1_teor; i++) Series12->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vseZ_utochn[i],"",clBlue);
if (CheckBox51->Checked==false) for (int i=0; i<=m1_teor; i++) Series12->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ[i],"",clBlue);
if (CheckBox51->Checked==true)  for (int i=0; i<=m1_bez_utochn; i++) Series10->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ_dTeta[i],"",clBlue);  //низ
if (CheckBox51->Checked==false) for (int i=0; i<=m1_teor; i++) Series10->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ[i],"",clBlue);  //низ
	}

if (CheckBox51->Checked==false)   // без уточнення
  for (int i=0; i<=m1_teor; i++)
    {
    if (number_KDV==2 || number_KDV==3 )
      {
      if (CheckBox19->Checked==false)
	{
	if (KDV_lich==3) Series48->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed );
	if (KDV_lich==2) Series6 ->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed );
	if (KDV_lich==1) Series14->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vse[i],"",clRed );
	}
	if (KDV_lich==3) Series49->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ[i],"",clBlue );
	if (KDV_lich==2) Series10->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ[i],"",clBlue );
	if (KDV_lich==1) Series12->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZ[i],"",clBlue );
      }
    }
if (CheckBox51->Checked==true)    // уточнення
  for (int i=0; i<=m1_teor; i++)
    {
    if (number_KDV==2 || number_KDV==3 )
      {
      if (CheckBox19->Checked==false)
	{
	if (KDV_lich==3) Series48->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vse_utochn[i],"",clRed	);
	if (KDV_lich==2) Series6 ->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vse_utochn[i],"",clRed	);
	if (KDV_lich==1) Series14->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vse_utochn[i],"",clRed	);
	}
	if (KDV_lich==3) Series49->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vseZ_utochn[i],"",clBlue );
	if (KDV_lich==2) Series10->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vseZ_utochn[i],"",clBlue );
	if (KDV_lich==1) Series12->AddXY(DeltaTeta_utochn[i]/M_PI*(3600.*180.),R_vseZ_utochn[i],"",clBlue );
      }
    }
}


if (fitting==0 || (fitting==1 && vse==2) || (fitting==10 && vse==2))
  {
  //int op=-m10z;
  //int ok=m1z-m10z;
  for (int i=0; i<=m1_teor-(op+ok); i++)  //   Зсув КДВ до початку інформативної області
    {
    R_dif_[i][KDV_lich]=R_dif_[i+ok][KDV_lich];
    R_cogerTT_[i][KDV_lich]=R_cogerTT_[i+ok][KDV_lich];
    R_vse_[i][KDV_lich]=R_vse_[i+ok][KDV_lich];
    }
  }


for (int i=0; i<=m1_teor-(op+ok); i++) R_vseZ[i]= R_vseZ[i+ok];    //   Зсув КДВ до початку інформативної області

nsd=m10;

if (CheckBox53->Checked==true) Memo8-> Lines->Add("Виведення КДВ!");

if (vved_exper==1 || vved_exper==2)
  {
  if ((KDV_lich==1) || (KDV_lich==2 && CheckBox42->Checked==false) ||
    (KDV_lich==3 && CheckBox42->Checked==false && CheckBox43->Checked==false))
    {CKV_[1]=0; CKV_[2]=0; CKV_[3]=0;}

  if (KDV_lich==1)
    {
    for (int i=0; i<=m1; i++) R_vseZg[i][1]=R_vseZ[i];
    CKV=0;
    if (CheckBox29->Checked==false)  //  vskv/askv   false->askv
      {
      II=0;        // CheckBox22->Checked==true -> Розрив при обч. СКВ
      if (CheckBox22->Checked==false) for (int i=m10+nskvi1;   i<=m10+kskvi1; i++) {CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1]); II=II+intIk2d[i][1]*R_vseZg[i][1];}
      if (CheckBox22->Checked==true)  for (int i=m10+nskvi1; i<=m10+nskvi1_r; i++) {CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1]); II=II+intIk2d[i][1]*R_vseZg[i][1];}
      if (CheckBox22->Checked==true)  for (int i=m10+kskvi1_r; i<=m10+kskvi1; i++) {CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1]); II=II+intIk2d[i][1]*R_vseZg[i][1];}
      CKV=CKV/II;
      }
    if (CheckBox29->Checked==true)   //  vskv/askv   true->vskv
      {            // CheckBox22->Checked==true -> Розрив при обч. СКВ
      if (CheckBox22->Checked==false) for (int i=m10+nskvi1;   i<=m10+kskvi1; i++) CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1])/intIk2d[i][1]/intIk2d[i][1];
      if (CheckBox22->Checked==true)  for (int i=m10+nskvi1; i<=m10+nskvi1_r; i++) CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1])/intIk2d[i][1]/intIk2d[i][1];
      if (CheckBox22->Checked==true)  for (int i=m10+kskvi1_r; i<=m10+kskvi1; i++) CKV=CKV+(intIk2d[i][1]-R_vseZg[i][1])*(intIk2d[i][1]-R_vseZg[i][1])/intIk2d[i][1]/intIk2d[i][1];
      }
    if (CheckBox22->Checked==false) CKV=sqrt(CKV/(kskvi1-nskvi1+1)/(kskvi1-nskvi1+1));
    if (CheckBox22->Checked==true)  CKV=sqrt(CKV/((kskvi1-kskvi1_r+1)+(nskvi1_r-nskvi1+1))/((kskvi1-kskvi1_r+1)+(nskvi1_r-nskvi1+1)));
    if (CKV<=1e-9)  CKV=1e-9;
    CKV_[1]=CKV;
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka     CKV_[1]");
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(FloatToStr(CKV_[1]));
  }
  if (KDV_lich==2)
    {
    for (int i=0; i<=m1; i++) R_vseZg[i][2]=R_vseZ[i];
    CKV=0;
    if (CheckBox29->Checked==false)  //  vskv/askv   false->askv
      {
      II=0;         // CheckBox45->Checked==true -> Розрив при обч. СКВ
      if (CheckBox45->Checked==false) for (int i=m10+nskvi2;   i<=m10+kskvi2; i++) {CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2]); II=II+intIk2d[i][2]*R_vseZg[i][2];}
      if (CheckBox45->Checked==true)  for (int i=m10+nskvi2; i<=m10+nskvi2_r; i++) {CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2]); II=II+intIk2d[i][2]*R_vseZg[i][2];}
      if (CheckBox45->Checked==true)  for (int i=m10+kskvi2_r; i<=m10+kskvi2; i++) {CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2]); II=II+intIk2d[i][2]*R_vseZg[i][2];}
      CKV=CKV/II;
      }
    if (CheckBox29->Checked==true)   //  vskv/askv   true->vskv
      {             // CheckBox45->Checked==true -> Розрив при обч. СКВ
      if (CheckBox45->Checked==false) for (int i=m10+nskvi2;   i<=m10+kskvi2; i++) CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2])/intIk2d[i][2]/intIk2d[i][2];
      if (CheckBox45->Checked==true)  for (int i=m10+nskvi2; i<=m10+nskvi2_r; i++) CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2])/intIk2d[i][2]/intIk2d[i][2];
      if (CheckBox45->Checked==true)  for (int i=m10+kskvi2_r; i<=m10+kskvi2; i++) CKV=CKV+(intIk2d[i][2]-R_vseZg[i][2])*(intIk2d[i][2]-R_vseZg[i][2])/intIk2d[i][2]/intIk2d[i][2];
      if (CheckBox45->Checked==false) CKV=sqrt(CKV/(kskvi2-nskvi2+1)/(kskvi2-nskvi2+1));
      if (CheckBox45->Checked==true)  CKV=sqrt(CKV/((kskvi2-kskvi2_r+1)+(nskvi2_r-nskvi2+1))/((kskvi2-kskvi2_r+1)+(nskvi2_r-nskvi2+1)));
      }
    if (CKV<=1e-9)  CKV=1e-9;
    CKV_[2]=CKV;
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka     CKV_[2]");
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(FloatToStr(CKV_[2]));
    }
  if (KDV_lich==3)
    {
    for (int i=0; i<=m1; i++) R_vseZg[i][3]=R_vseZ[i];
    CKV=0;
    if (CheckBox29->Checked==false)  //  vskv/askv   false->askv
      {
      II=0;          // CheckBox46->Checked==true -> Розрив при обч. СКВ
      if (CheckBox46->Checked==false) for (int i=m10+nskvi3;   i<=m10+kskvi3; i++) {CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3]); II=II+intIk2d[i][3]*R_vseZg[i][3];}
      if (CheckBox46->Checked==true)  for (int i=m10+nskvi3; i<=m10+nskvi3_r; i++) {CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3]); II=II+intIk2d[i][3]*R_vseZg[i][3];}
      if (CheckBox46->Checked==true)  for (int i=m10+kskvi3_r; i<=m10+kskvi3; i++) {CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3]); II=II+intIk2d[i][3]*R_vseZg[i][3];}
      CKV=CKV/II;
      }
    if (CheckBox29->Checked==true)   //  vskv/askv   true->vskv
      {              // CheckBox46->Checked==true -> Розрив при обч. СКВ
      if (CheckBox46->Checked==false) for (int i=m10+nskvi3;   i<=m10+kskvi3; i++) CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3])/intIk2d[i][3]/intIk2d[i][3];
      if (CheckBox46->Checked==true)  for (int i=m10+nskvi3; i<=m10+nskvi3_r; i++) CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3])/intIk2d[i][3]/intIk2d[i][3];
      if (CheckBox46->Checked==true)  for (int i=m10+kskvi3_r; i<=m10+kskvi3; i++) CKV=CKV+(intIk2d[i][3]-R_vseZg[i][3])*(intIk2d[i][3]-R_vseZg[i][3])/intIk2d[i][3]/intIk2d[i][3];
      if (CheckBox46->Checked==false) CKV=sqrt(CKV/(kskvi3-nskvi3+1)/(kskvi3-nskvi3+1));
      if (CheckBox46->Checked==true)  CKV=sqrt(CKV/((kskvi3-kskvi3_r+1)+(nskvi3_r-nskvi3+1))/((kskvi3-kskvi3_r+1)+(nskvi3_r-nskvi3+1)));
      }
    if (CKV<=1e-9)  CKV=1e-9;
    CKV_[3]=CKV;
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka     CKV_[3]");
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(FloatToStr(CKV_[3]));
    }

  koefCKV1=StrToFloat(Edit400->Text);
  koefCKV2=StrToFloat(Edit401->Text);
  koefCKV3=StrToFloat(Edit402->Text);
  if (KDV_lich==number_KDV)CKV=CKV_[1]*koefCKV1+CKV_[2]*koefCKV2+CKV_[3]*koefCKV3;
  if (fitting==0 && KDV_lich==number_KDV)
    {
    Edit147->Text=FloatToStr(CKV);
    Edit388->Text=FloatToStr(CKV_[1]);
    Edit392->Text=FloatToStr(CKV_[2]);
    Edit393->Text=FloatToStr(CKV_[3]);
    Memo11->Lines->Add(FloatToStr(CKV));
    }
  if (CheckBox48->Checked==true) if (KDV_lich==number_KDV) Form2->Memo1->Lines->Add(FloatToStr(CKV));
  }

if (CheckBox53->Checked==true) Memo8-> Lines->Add("СКВ пораховано!");

if (fitting==1 || fitting==10)
  {
  //int nsd=m10;
  if (CheckBox42->Checked==true && KDV_lich==1) nom=0;
  if (CheckBox42->Checked==false && CheckBox43->Checked==true && KDV_lich==2) nom=0;
  if (CheckBox42->Checked==false && CheckBox43->Checked==false && CheckBox44->Checked==true && KDV_lich==3) nom=0;

  if (KDV_lich==1)
    {
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka R_vseZ PE KDV_lich==1:");
    if (CheckBox22->Checked==false)  //Без розрив. при обч. СКВ
      {
      nom10=kskvi1-nskvi1+1;
      for (int i=1; i<=nom10; i++)
 	    {
 	    R_vseZa[i]= R_vseZ[i+nskvi1-1+nsd];
 	    if (vse==1) PE[i]=intIk2d[i+nskvi1-1+nsd][1];
 	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom10; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i])+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(10));
      nom=nom+nom10;
      }
    if (CheckBox22->Checked==true)  // Розрив при обч. СКВ
      {
      nom11=nskvi1_r-nskvi1+1;
      for (int i=1; i<=nom11; i++)
	    { 
	    R_vseZa[i]= R_vseZ[i+nskvi1-1+nsd];
	    if (vse==1) PE[i]= intIk2d[i+nskvi1-1+nsd][1];
	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom11; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i])+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(11));
      nom=nom+nom11;
      nom12=kskvi1-kskvi1_r+1;
      for (int i=1; i<=nom12; i++)
        {
        R_vseZa[i+nom]= R_vseZ[i+kskvi1_r-1+nsd];
        if (vse==1) PE[i+nom]= intIk2d[i+kskvi1_r-1+nsd][1];
        }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom12; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(12));
      nom=nom+nom12;
      }
//if (CheckBox48->Checked==true) for (int i=0; i<=30; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(R_vseZa[i])+'\t'+FloatToStr(R_vseZ[i]));
//Memo1->Lines->Add("R_vseZR_vseZR_vseZR_vseZR_vseZ 11111");
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
    }

  if (KDV_lich==2)
    {
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka R_vseZ PE KDV_");
    if (CheckBox45->Checked==false)  //Без розрив. при обч. СКВ
      {
      nom20=kskvi2-nskvi2+1;
      for (int i=1; i<=nom20; i++)
    	{
 	    R_vseZa[i+nom]= R_vseZ[i+nskvi2-1+nsd];
 	    if (vse==1)	 PE[i+nom]= intIk2d[i+nskvi2-1+nsd][2];
        }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom20; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(20));
      nom=nom+nom20;
      }
    if (CheckBox45->Checked==true)  // Розрив при обч. СКВ
      {
      nom21=nskvi2_r-nskvi2+1;
      for (int i=1; i<=nom21; i++)
	    {
	    R_vseZa[i+nom]= R_vseZ[i+nskvi2-1+nsd];
 	    if (vse==1) PE[i+nom]= intIk2d[i+nskvi2-1+nsd][2];
 	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom21; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(21));
      nom=nom+nom21;
      nom22=kskvi2-kskvi2_r+1;
      for (int i=1; i<=nom22; i++)
	    {
	    R_vseZa[i+nom]= R_vseZ[i+kskvi2_r-1+nsd];
 	    if (vse==1) PE[i+nom]= intIk2d[i+kskvi2_r-1+nsd][2];
 	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom22; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(22));
      nom=nom+nom22;
      }
//if (CheckBox48->Checked==true) for (int i=0; i<=30; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(R_vseZa[i])+'\t'+FloatToStr(R_vseZ[i]));
//Memo1->Lines->Add("R_vseZaR_vseZaR_vseZaR_vseZa 222222");
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
    }

  if (KDV_lich==3)
    {
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("Zgortka R_vseZ PE KDV_");
    if (CheckBox46->Checked==false)  //Без розрив. при обч. СКВ
      {
      nom30=kskvi3-nskvi3+1;
      for (int i=1; i<=nom30; i++)
 	    {
 	    R_vseZa[i+nom]= R_vseZ[i+nskvi3-1+nsd];
 	    if (vse==1) PE[i+nom]= intIk2d[i+nskvi3-1+nsd][3];
 	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom30; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(30));
      nom=nom+nom30;
      }
    if (CheckBox46->Checked==true)  // Розрив при обч. СКВ
      {
      nom31=nskvi3_r-nskvi3+1;
      for (int i=1; i<=nom31; i++)
	    {
	    R_vseZa[i+nom]= R_vseZ[i+nskvi3-1+nsd];
	    if (vse==1) PE[i+nom]= intIk2d[i+nskvi3-1+nsd][3];
        }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom31; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(31));
      nom=nom+nom31;
      nom32=kskvi3-kskvi3_r+1;
      for (int i=1; i<=nom32; i++)
	    {
	    R_vseZa[i+nom]= R_vseZ[i+kskvi3_r-1+nsd];
	    if (vse==1) PE[i+nom]= intIk2d[i+kskvi3_r-1+nsd][3];
	    }
      //if (CheckBox48->Checked==true) for (int i=1; i<=nom32; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(R_vseZa[i+nom])+'\t'+FloatToStr(PE[i+nom])+'\t'+FloatToStr(32));
      nom=nom+nom32;
      }
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi3)+'\t'+IntToStr(kskvi3)+'\t'+IntToStr(nskvi3_r)+'\t'+IntToStr(kskvi3_r)+'\t'+IntToStr(nsd));
    }
  if (CheckBox48->Checked==true)  // Запис в Form2->Memo1 з Zgortka
    {
    Form2->Memo1->Lines->Add(IntToStr(nom));
    Form2->Memo1->Lines->Add(IntToStr(nom10)+'\t'+IntToStr(nom11)+'\t'+IntToStr(nom12));
    Form2->Memo1->Lines->Add(IntToStr(nom20)+'\t'+IntToStr(nom21)+'\t'+IntToStr(nom22));
    Form2->Memo1->Lines->Add(IntToStr(nom30)+'\t'+IntToStr(nom31)+'\t'+IntToStr(nom32));
    Form2->Memo1->Lines->Add(" Всі      R_vseZa[i]    PE[i]");
//    for (int i=0; i<=nom; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(R_vseZa[i])+'\t'+FloatToStr(PE[i]));
    }

  if (vse==1)
    {
/*
n=0;
if (KDV_lich==1)
{
if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("   vse1  KDV_lich==1:");
	if (CheckBox22->Checked==false)
	{
 	n=kskvi1-nskvi1+1;
 	for (int i=1; i<=kskvi1-nskvi1+1; i++) PE[i]=intIk2d[i+nskvi1-1+nsd][1];
if (CheckBox48->Checked==true) for (int i=1; i<=nom10; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(110));
	}
	if (CheckBox22->Checked==true)
	{
	 n=n+(kskvi1-kskvi1_r+1)+(nskvi1_r-nskvi1+1);
	for (int i=1; i<=nskvi1_r-nskvi1+1; i++) PE[i]= intIk2d[i+nskvi1-1+nsd][1];
if (CheckBox48->Checked==true) for (int i=1; i<=nom10; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(110));
	for (int i=1; i<=kskvi1-kskvi1_r+1; i++) PE[i+nskvi1_r-nskvi1+1]= intIk2d[i+kskvi1_r-1+nsd][1];
if (CheckBox48->Checked==true) for (int i=1; i<=nom10; i++)  Form2->Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(110));
	}
if (CheckBox48->Checked==true) for (int i=0; i<=30; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(intIk2d[i][1]));
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
}

if (KDV_lich==2)
{
if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("   vse1  KDV_lich==2:");
	if (CheckBox45->Checked==false)
	{
	 n=n+kskvi2-nskvi2+1;
 	for (int i=1; i<=kskvi2-nskvi2+1; i++) PE[i+kskvi1-nskvi1+1]= intIk2d[i+nskvi2-1+nsd][2];
	}
	if (CheckBox45->Checked==true)
	{
	 n=n+(kskvi2-kskvi2_r+1)+(nskvi2_r-nskvi2+1);
	for (int i=1; i<=nskvi2_r-nskvi2+1; i++) PE[i+nskvi1_r-nskvi1+1+kskvi1-kskvi1_r+1]= intIk2d[i+nskvi2-1+nsd][2];
	for (int i=1; i<=kskvi2-kskvi2_r+1; i++) PE[i+nskvi1_r-nskvi1+1+kskvi1-kskvi1_r+1+nskvi2_r-nskvi2+1]= intIk2d[i+kskvi2_r-1+nsd][2];
	}
if (CheckBox48->Checked==true) for (int i=0; i<=30; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(intIk2d[i][2]));
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
}

if (KDV_lich==3)
{
if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add("   vse1  KDV_lich==3:");
	if (CheckBox46->Checked==false)
	{
	 n=n+kskvi3-nskvi3+1;
 	for (int i=1; i<=kskvi3-nskvi3+1; i++) PE[i+kskvi1-nskvi1+1+kskvi2-nskvi2+1]= intIk2d[i+nskvi3-1+nsd][3];
   }
	if (CheckBox46->Checked==true)
	{
	 n=n+(kskvi3-kskvi3_r+1)+(nskvi3_r-nskvi3+1);
	for (int i=1; i<=nskvi3_r-nskvi3+1; i++) PE[i+nskvi1_r-nskvi1+1+kskvi1-kskvi1_r+1+nskvi2_r-nskvi2+1+kskvi2-kskvi2_r+1]= intIk2d[i+nskvi3-1+nsd][3];
	for (int i=1; i<=kskvi3-kskvi3_r+1; i++) PE[i+nskvi1_r-nskvi1+1+kskvi1-kskvi1_r+1+nskvi2_r-nskvi2+1+kskvi2-kskvi2_r+1+nskvi3_r-nskvi3+1]= intIk2d[i+kskvi3_r-1+nsd][3];
	}
if (CheckBox48->Checked==true) for (int i=0; i<=30; i++)  Form2->Memo1->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(PE[i])+'\t'+FloatToStr(intIk2d[i][3]));
//Memo1->Lines->Add(IntToStr(nskvi1)+'\t'+IntToStr(kskvi1)+'\t'+IntToStr(nskvi1_r)+'\t'+IntToStr(kskvi1_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi2)+'\t'+IntToStr(kskvi2)+'\t'+IntToStr(nskvi2_r)+'\t'+IntToStr(kskvi2_r)+'\t'+IntToStr(nsd));
//Memo1->Lines->Add(IntToStr(nskvi3)+'\t'+IntToStr(kskvi3)+'\t'+IntToStr(nskvi3_r)+'\t'+IntToStr(kskvi3_r)+'\t'+IntToStr(nsd));
}
*/

    if (KDV_lich==number_KDV) Edit147->Text=FloatToStr(CKV); // останній цикл завершився і вивело стартове СКВ

    Form2->Visible = true;
    if (KDV_lich==1) Form2->Edit6->Text=FloatToStr(CKV_[1]);
    if (KDV_lich==2) Form2->Edit7->Text=FloatToStr(CKV_[2]);
    if (KDV_lich==3) Form2->Edit8->Text=FloatToStr(CKV_[3]);
    if (KDV_lich==number_KDV)Form2->Edit4->Text=FloatToStr(CKV);

    if (KDV_lich==1) for (int i=0; i<=m1; i++) R_vseZg0[i][1]=R_vseZg[i][1];
    if (KDV_lich==2) for (int i=0; i<=m1; i++) R_vseZg0[i][2]=R_vseZg[i][2];
    if (KDV_lich==3) for (int i=0; i<=m1; i++) R_vseZg0[i][3]=R_vseZg[i][3];
    for (int i=0; i<=m1; i++)
      {
      if (KDV_lich==1)  Form2->Series3->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg0[i][1],"",clBlue);
      if (KDV_lich==1)  Form2->Series2->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),intIk2d[i][1],"",clGreen);
      if (KDV_lich==2)  Form2->Series6->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg0[i][2],"",clBlue);
      if (KDV_lich==2)  Form2->Series5->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),intIk2d[i][2],"",clGreen);
      if (KDV_lich==3)  Form2->Series9->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg0[i][3],"",clBlue);
      if (KDV_lich==3)  Form2->Series8->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),intIk2d[i][3],"",clGreen);
      }
    }
  if (vse==2)    // запис результату наближення
    {
    if (KDV_lich==number_KDV) Edit71->Text=FloatToStr(CKV); // останній цикл завершився і вивело кінцеве СКВ

    if (KDV_lich==1) Form2->Edit1->Text=FloatToStr(CKV_[1]);
    if (KDV_lich==2) Form2->Edit2->Text=FloatToStr(CKV_[2]);
    if (KDV_lich==3) Form2->Edit5->Text=FloatToStr(CKV_[3]);
    if (KDV_lich==number_KDV)Form2->Edit3->Text=FloatToStr(CKV);
    for (int i=0; i<=m1; i++)
      {
      if (KDV_lich==1)  Form2->Series1->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg[i][1],"",clRed);
      if (KDV_lich==2)  Form2->Series4->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg[i][2],"",clRed);
      if (KDV_lich==3)  Form2->Series7->AddXY(DeltaTeta[i]/M_PI*(3600.*180.),R_vseZg[i][3],"",clRed);
      }
    }
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(" Кінець згортки 1  10");
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(IntToStr(nom));
  }

    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(" Кінець згортки");
    if (CheckBox48->Checked==true) Form2->Memo1->Lines->Add(IntToStr(nom));

delete  R_vse_utochn, R_vse_dTeta, R_vseZ_dTeta, R_vseZ_utochn;
delete  POO_dTeta1, POO_dTeta, POO1, DeltaTeta_utochn;
delete  POO, R_vse, R_vseZ;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//             ПІДПРОГРАМИ ДЛЯ   !Auto7c.for!
//---------------------------------------------------------------------------

void TForm1::CALCULATION()
{
double   Dmax, DD0;
double Z_shod [2*KM],D_shod [2*KM],L_shod;

//TStringList *List3 = new TStringList;

if (RadioButton22->Checked==true)
{     //Набл. диф.розс. в ід. част. монокр. ( ідеальний монокристал )
if (CheckBox79->Checked==true)  //   CheckBox13
{
Defekts_mon[1]=DLa[method_lich][1];   // R001=
Defekts_mon[2]=DDa[method_lich][1];  // nL01=
}
if (CheckBox80->Checked==true)   //   CheckBox17
{
Defekts_mon[4]=DLa[method_lich][2];   // R002=
Defekts_mon[5]=DDa[method_lich][2];  // nL02=
}
	if (vse==2)    // запис результату наближення
	{
if (CheckBox79->Checked==true)
{
	Edit158->Text=FloatToStr(Defekts_mon[1]*1e8);  // R001
	Edit155->Text=FloatToStr(Defekts_mon[2]);      // nL01
}
if (CheckBox80->Checked==true)
{
	Edit157->Text=FloatToStr(Defekts_mon[4]*1e8);  // R002
	Edit156->Text=FloatToStr(Defekts_mon[5]);      // nL02
}
	}
}

if (RadioButton27->Checked==true)
{     //Набл. диф.розс. в ід. част. плівки
if (CheckBox34->Checked==true)
{
Defekts_film[1]=DLa[method_lich][1];   // R001=
Defekts_film[2]=DDa[method_lich][1];  // nL01=
}
if (CheckBox35->Checked==true)
{
Defekts_film[4]=DLa[method_lich][2];   // R002=
Defekts_film[5]=DDa[method_lich][2];  // nL02=
}
	if (vse==2)    // запис результату наближення
	{
if (CheckBox34->Checked==true)
{
	Edit307->Text=FloatToStr(Defekts_film[1]*1e8);  // R001
	Edit74->Text=FloatToStr(Defekts_film[2]);      // nL01
}
if (CheckBox35->Checked==true)
{
	Edit306->Text=FloatToStr(Defekts_film[4]*1e8);  // R002
	Edit305->Text=FloatToStr(Defekts_film[5]);      // nL02
}
        }
}

if (RadioButton23->Checked==true)
{     //Набл. диф.розс. в ППШ (з врах. диф.розс. в  ід. част. монокр та ППШ.)
if (CheckBox1->Checked==true)
{
Defekts_SL[1]=DLa[method_lich][1];      // R0_max=
Defekts_SL[2]=DDa[method_lich][1];     // nL_max=
}
if (CheckBox2->Checked==true)
{
Defekts_SL[4]=DLa[method_lich][1];      // R0p_max=
Defekts_SL[5]=DDa[method_lich][1];     // np_max=
}

	if (vse==2)    // запис результату наближення
	{
	if (CheckBox1->Checked==true)
	{
	Edit154->Text=FloatToStr(Defekts_SL[1]*1e8);  // R0_max
	Edit153->Text=FloatToStr(Defekts_SL[2]);      // nL_max
	}
	if (CheckBox2->Checked==true)
	{
	Edit154->Text=FloatToStr(Defekts_SL[4]*1e8);  // R0p_max    !!!!!!!!!!!!!!
	Edit153->Text=FloatToStr(Defekts_SL[5]);      // np_max     !!!!!!!!!!!!!!!!
	}
	}
}

if (RadioButton24->Checked==true)
{      //Набл. профіль сход.  в ППШ (з врах. диф.розс. в  ід. част. монокр. та ППШ)
Dmax=-100;
for (int k=1; k<=km; k++)
{
DD[k]=DDa[method_lich][k];
Dl[k]=DLa[method_lich][k];
if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
}
for (int k=1; k<=km; k++) f[k]=fabs(DD[k]/Dmax);

	if (vse==2)    // запис результату наближення
	{
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;

for (int k=1; k<=2*km+1;k++) Series3->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clRed);
//Запис вихідного профілю:
for (int k=1; k<=km; k++)
{
DD[k]=DDa[5][k];
Dl[k]=DLa[5][k];
}
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
for (int k=1; k<=2*km+1;k++) Series34->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlue);

Memo6->Clear();
Memo5->Clear();
for (int k=1; k<=km;k++)
{
DD[k]=DDa[method_lich][k];
Dl[k]=DLa[method_lich][k];
Memo6-> Lines->Add(IntToStr(k));
Memo5-> Lines->Add(FloatToStr(DDa[5][k])+'\t'+FloatToStr(DLa[5][k])*1e8);
Memo2-> Lines->Add(FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k])*1e8);
}
	}
}

if (RadioButton24->Checked==true && CheckBox24->Checked==true)
{      //Набл. профіль. та диф. розс. в ППШ (з врах. диф.розс. в  ід. част. монокр. та ППШ)
Dmax=0;
for (int k=1; k<=km; k++)
{
DD[k]=DDa[method_lich][k];
Dl[k]=DLa[method_lich][k];
if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
}
for (int k=1; k<=km; k++) f[k]=fabs(DD[k]/Dmax);

if (CheckBox1->Checked==true)
{
Defekts_SL[1]=DLa[method_lich][km+km_dp];    // R0_max=
Defekts_SL[2]=DDa[method_lich][km+km_dp];   // nL_max=
}
if (CheckBox2->Checked==true)
{
Defekts_SL[4]=DLa[method_lich][km+km_dp];    // R0p_max=
Defekts_SL[5]=DDa[method_lich][km+km_dp];   // np_max=
}

	if (vse==2)    // запис результату наближення
	{
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;

for (int k=1; k<=2*km+1;k++) Series3->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clRed);

//Запис вихідного профілю:
for (int k=1; k<=km; k++)
{
DD[k]=DDa[5][k];
Dl[k]=DLa[5][k];
}
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
for (int k=1; k<=2*km+1;k++) Series34->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlue);

Dmax=-100;
for (int k=1; k<=km;k++)
{
DD[k]=DDa[method_lich][k];
Dl[k]=DLa[method_lich][k];
if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
}
        }
}

if (RadioButton25->Checked==true)
{      //Набл. ф-ю розподілу розворотів блоків від кута (без врах. диф.розс. в  ід. част. монокр. та ППШ)
double    Snn,Afi,fff1[100],fi[100];
for (int k=0; k<=km_rozv; k++)
{
nn_m[k]=DDa[method_lich][k+1];
DFi[k]=DLa[method_lich][k+1];
}

	if (vse==2)    // запис результату наближення
	{
    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;
  Afi=StrToFloat(Edit80->Text); // Коеф. в  DD_rozv[kr] (fi[kr]);
  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
for (int k=0; k<=km_rozv;k++)
{
Series43->AddXY(fi[k],DD_rozv[k]*10000,"",clRed);
Series22->AddXY(fi[k],fff1[k]*100,"",clRed);
}
//Запис вихідного профілю:
for (int k=0; k<=km_rozv; k++)
{
nn_m[k]=DDa[5][k+1];
DFi[k]=DLa[5][k+1];
}
    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;

  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
for (int k=0; k<=km_rozv;k++)
{
Series23->AddXY(fi[k],DD_rozv[k]*10000,"",clBlue);
Series44->AddXY(fi[k],fff1[k]*100,"",clBlue);
}
Memo6->Clear();
Memo5->Clear();
for (int k=0; k<=km_rozv;k++)
{
nn_m[k]=DDa[method_lich][k+1];
DFi[k]=DLa[method_lich][k+1];
Memo6-> Lines->Add(IntToStr(k));
Memo5-> Lines->Add(FloatToStr(DDa[5][k+1])+'\t'+FloatToStr(DLa[5][k+1]));
Memo2-> Lines->Add(FloatToStr(DDa[method_lich][k+1])+'\t'+FloatToStr(DLa[method_lich][k+1]));
}
	}
}



//RozrachDiduz();
//RozrachKoger();
//Zgortka();

QuickStart();

//List3->Add(FloatToStr(CKV));
//List3->Add("");
//List3->SaveToFile("dataCalc.dat");

}

//---------------------------------------------------------------------------
//              МІНІМІЗАЦІЯ СКВ   !Auto7c.for!
void __fastcall TForm1::Button15Click(TObject *Sender)
{
int  m,n, np, kkk,ll,ns,nr,nrai,nraj,nsai,npn, lich_sikleK, NN[4],kma;
double  dDDa_dp,dDL_dp,dDDa_p,dDL_p,DDa_max,DDa_min,S2,S3,Amax,DDa_max2,DDa_min2;
double   aAa, s ,koef_d_DD;
double   Dmax,L,DD0,riznSKV;
double    dfi,nenull,nenull_d_DL;
double   SKV[12];
double *d_DDa, *d_DL, *DDamax, *DDamin; // [kma]
double *PZ0, *YYY, *PZ;                 // [n]
double **DIDD, **DIDDTRAN;              // [n][m], [m][n]
double **DOB1, **X, **OBER;             // [m][m]
double **AA;                            // [m][2*m]
double *NSA, *NRA;                      // [m]
double **DOB2, **YYYY, **TTTT;          // [m][n], [n][1], [m][1]
double **CHANGE;                        // [3][m]
TStringList *List1 = new TStringList;

fitting=1;
riznSKV=StrToFloat(Edit139->Text);
if (RadioButton22->Checked==false && RadioButton23->Checked==false && RadioButton24->Checked==false && RadioButton25->Checked==false && RadioButton27->Checked==false)
{
MessageBox(0,"Так що ж наближати?","Спиш???!", MB_OK + MB_ICONEXCLAMATION);
goto m99999;
}

for (int j=0; j<=14; j++) SKV[j]=0;

if (RadioButton22->Checked==true) //Набл. диф.розс. в ід. част. монокр. шляхом зміни параметрів дисл.петель
{
lich_sikleK=StrToInt(Edit73->Text);
kma=0;
if (CheckBox79->Checked==true) kma=kma+1;    // CheckBox13
if (CheckBox80->Checked==true) kma=kma+1;    // CheckBox17
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DLa[j][k]=0;

koef_d_DD=StrToFloat(Edit344->Text);

if (CheckBox79->Checked==true)
{
DDa[0][1]=StrToFloat(Edit53->Text);       // =nL01;
DLa[0][1]=1e-8*StrToFloat(Edit54->Text);   // =R001;
}
if (CheckBox80->Checked==true)
{
DDa[0][2]=StrToFloat(Edit64->Text);       // =nL02;
DLa[0][2]=1e-8*StrToFloat(Edit65->Text);   // =R002;
}
  dDDa_dp=StrToFloat(Edit66->Text);
  dDL_dp =StrToFloat(Edit67->Text)*1e-8;
  DDa_max=StrToFloat(Edit68->Text);
  DDa_min=StrToFloat(Edit48->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
if (RadioButton60->Checked==true)
  for (int k=1; k<=kma; k++)
    {
    d_DDa[k]=dDDa_dp;
    d_DL[k]=dDL_dp;
    }
if (RadioButton61->Checked==true)
  for (int k=1; k<=kma; k++)
    {
    d_DDa[k]=DDa[0][k]*koef_d_DD;
    d_DL[k]=DLa[0][k]*koef_d_DD;
    }
for (int k=1; k<=kma; k++) DDamax[k]=DDa_max;
for (int k=1; k<=kma; k++) DDamin[k]=DDa_min;
}

if (RadioButton27->Checked==true) //Набл. диф.розс. в ід. част. плівки шляхом зміни параметрів дисл.петель
{
lich_sikleK=StrToInt(Edit73->Text);
kma=0;
if (CheckBox34->Checked==true)  kma=kma+1;
if (CheckBox35->Checked==true)  kma=kma+1;
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DLa[j][k]=0;

koef_d_DD=StrToFloat(Edit344->Text);

if (CheckBox81->Checked==true)
{
DDa[0][1]=StrToFloat(Edit174->Text);       // =nL01;
DLa[0][1]=1e-8*StrToFloat(Edit175->Text);   // =R001;
}
if (CheckBox82->Checked==true)
{
DDa[0][2]=StrToFloat(Edit176->Text);       // =nL02;
DLa[0][2]=1e-8*StrToFloat(Edit177->Text);   // =R002;
}
  dDDa_dp=StrToFloat(Edit66->Text);
  dDL_dp =StrToFloat(Edit67->Text)*1e-8;
  DDa_max=StrToFloat(Edit68->Text);
  DDa_min=StrToFloat(Edit48->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
if (RadioButton60->Checked==true)
  for (int k=1; k<=kma; k++)
    {
    	 d_DDa[k]=dDDa_dp;
     d_DL[k]=dDL_dp;
    }
if (RadioButton61->Checked==true)
  for (int k=1; k<=kma; k++) 
    {
    d_DDa[k]=DDa[0][k]*koef_d_DD;
    	d_DL[k]=DLa[0][k]*koef_d_DD;
    }
for (int k=1; k<=kma; k++) DDamax[k]=DDa_max;
for (int k=1; k<=kma; k++) DDamin[k]=DDa_min;
}

if (RadioButton23->Checked==true) //Набл. диф.розс. в ППШ шляхом зміни параметрів дисл.петель чи сф.кластерів
{
if (CheckBox23->Checked==true)  Profil(km,DD,dl);   // Стартовий профіль - гаусіана
if (CheckBox23->Checked==false)     // Стартовий профіль - сходинки
{
        km=StrToInt(Edit90->Text);
        ReadMemo2stovp(Memo5,km,DD,Dl);    //   Зчитуємо профіль з Memo5
        for (int k=1; k<=km;k++) Dl[k]=Dl[k]*1e-8;
        Dmax= -100;
        for (int k=1; k<=km; k++) if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
        for (int k=1; k<=km; k++) f[k]=fabs(DD[k]/Dmax);
}

lich_sikleK=StrToInt(Edit73->Text);
if (CheckBox1->Checked==true) kma=1;
if (CheckBox2->Checked==true) kma=1;
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=kma; k++) DLa[j][k]=0;

if (CheckBox1->Checked==true)
{
DDa[0][1]=StrToFloat(Edit2->Text);      // =nL_max;
DLa[0][1]=1e-8*StrToFloat(Edit3->Text);  // =R0_max;
}
if (CheckBox2->Checked==true)
{
DDa[0][1]=StrToFloat(Edit14->Text);      // =np_max;
DLa[0][1]=1e-8*StrToFloat(Edit15->Text);  // =R0p_max;
}

  dDDa_dp=StrToFloat(Edit66->Text);
  dDL_dp =StrToFloat(Edit67->Text)*1e-8;
  DDa_max=StrToFloat(Edit68->Text);
  DDa_min=StrToFloat(Edit48->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
for (int k=1; k<=kma; k++) d_DDa[k]=dDDa_dp;
for (int k=1; k<=kma; k++) d_DL[k]=dDL_dp;
for (int k=1; k<=kma; k++) DDamax[k]=DDa_max;
for (int k=1; k<=kma; k++) DDamin[k]=DDa_min;
}

if (RadioButton24->Checked==true) //Набл. профіль в ППШ (з врах. диф.розс. в  ід. част. монокр. та ППШ)
{
lich_sikleK=StrToInt(Edit73->Text);
if (CheckBox23->Checked==false)     // Стартовий профіль - сходинки
{
km=StrToInt(Edit90->Text);
for (int j=0; j<=5; j++) for (int k=1; k<=km; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=km; k++) DLa[j][k]=0;
ReadMemo2stovp(Memo5,km,DD,Dl);    //   Зчитуємо профіль з Memo5
for (int k=1; k<=km;k++) Dl[k]=Dl[k]*1e-8;
for (int k=1; k<=km; k++)  DDa[0][k]=DD[k];
for (int k=1; k<=km; k++)  DLa[0][k]=Dl[k];
}
if (CheckBox23->Checked==true)     // Стартовий профіль - гаусіана
{
Profil(km,DD,dl) ;
for (int j=0; j<=5; j++) for (int k=1; k<=km; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=km; k++) DLa[j][k]=0;
for (int k=1; k<=km; k++)  DDa[0][k]=DD[k];
for (int k=1; k<=km; k++)  DLa[0][k]=dl;
Edit90->Text=IntToStr(km);
}

kma=km;
  dDDa_p=StrToFloat(Edit93->Text);
  dDL_p =StrToFloat(Edit94->Text)*1e-8;
  DDa_max=StrToFloat(Edit95->Text);
  DDa_min=StrToFloat(Edit96->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
for (int k=1; k<=kma; k++) d_DDa[k]=dDDa_p;
for (int k=1; k<=kma; k++) d_DL[k]=dDL_p;
for (int k=1; k<=kma; k++) DDamax[k]=DDa_max;
for (int k=1; k<=kma; k++) DDamin[k]=DDa_min;


if (CheckBox24->Checked==true) //Набл. профіль. та диф. розс. в ППШ (шляхом зміни параметрів дисл.петель чи сф.кластерів)
{
if (CheckBox1->Checked==true && CheckBox2->Checked==false ) km_dp=1;    //В профілі врах. тільки петлі
if (CheckBox1->Checked==false && CheckBox2->Checked==true ) km_dp=1;    //В профілі врах. тільки кластери

if (CheckBox1->Checked==true)
{
DDa[0][km+1]=StrToFloat(Edit2->Text);       // =nL_max;
DLa[0][km+1]=1e-8*StrToFloat(Edit3->Text);   // =R0_max;
}
if (CheckBox2->Checked==true)
{
DDa[0][km+1]=StrToFloat(Edit14->Text);      // =np_max;
DLa[0][km+1]=1e-8*StrToFloat(Edit15->Text);  // =R0p_max;
}

kma=km+km_dp;
  dDDa_dp=StrToFloat(Edit66->Text);
  dDL_dp =StrToFloat(Edit67->Text)*1e-8;
  DDa_max2=StrToFloat(Edit68->Text);
  DDa_min2=StrToFloat(Edit48->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
for (int k=km+1; k<=km+km_dp; k++) d_DDa[k]=dDDa_dp;
for (int k=km+1; k<=km+km_dp; k++) d_DL[k]=dDL_dp;
for (int k=km+1; k<=km+km_dp; k++) DDamax[k]=DDa_max2;
for (int k=km+1; k<=km+km_dp; k++) DDamin[k]=DDa_min2;
}
}

if (RadioButton25->Checked==true) //Набл. ф-ю розподілу розворотів блоків від кута (без врах. диф.розс. в  ід. част. монокр. та ППШ)
{
lich_sikleK=StrToInt(Edit73->Text);
if (RadioButton15->Checked==true)
{
if (CheckBox23->Checked==true)  Profil(km,DD,dl);   // Стартовий профіль - гаусіана
if (CheckBox23->Checked==false)             // Стартовий профіль - сходинки
{
km=StrToInt(Edit90->Text);
ReadMemo2stovp(Memo5,km,DD,Dl);    //   Зчитуємо профіль з Memo5
for (int k=1; k<=km;k++) Dl[k]=Dl[k]*1e-8;
Dmax= -100;
for (int k=1; k<=km; k++) if (Dmax<fabs(DD[k])) Dmax=fabs(DD[k]);
for (int k=1; k<=km; k++) f[k]=fabs(DD[k]/Dmax);
}
}
if (RadioButton15->Checked==true || RadioButton26->Checked==true) // Блоки + профіль
{
km_rozv=StrToInt(Edit84->Text);
dfi=StrToFloat(Edit132->Text);
 //   nn_m[kr] - функція розподілу по кутах (зчитується з Memo7)
//if (CheckBox27->Checked==false) ReadMemo1stovp(Memo7,km_rozv,nn_m);
//if (CheckBox27->Checked==true) for (int kr=0; kr<=km_rozv;kr++) nn_m[kr]=1;
if (CheckBox27->Checked==false)
{
ReadMemo2stovp(Memo7,km_rozv,nn_m,DFi);
for (int k=0; k<=km_rozv;k++) //  Перенумерація елементів масивів
{
nn_m[k]=nn_m[k+1];
DFi[k]=DFi[k+1];
}
}
if (CheckBox27->Checked==true)for (int kr=0; kr<=km_rozv;kr++) {nn_m[kr]=1; DFi[kr]=dfi;}
}

for (int j=0; j<=5; j++) for (int k=1; k<=km_rozv+1; k++) DDa[j][k]=0;
for (int j=0; j<=5; j++) for (int k=1; k<=km_rozv+1; k++) DLa[j][k]=0;

for (int k=1; k<=km_rozv+1; k++)  DDa[0][k]=nn_m[k-1];
for (int k=1; k<=km_rozv+1; k++)  DLa[0][k]=DFi[k-1];

kma=km_rozv+1;
  dDDa_p=StrToFloat(Edit133->Text);
  dDL_p =StrToFloat(Edit136->Text);
  DDa_max=StrToFloat(Edit137->Text);
  DDa_min=StrToFloat(Edit138->Text);
d_DDa  = new double[kma+1];
d_DL   = new double[kma+1];
DDamax = new double[kma+1];
DDamin = new double[kma+1];
for (int k=1; k<=kma; k++) d_DDa[k]=dDDa_p;
for (int k=1; k<=kma; k++) d_DL[k]=dDL_p;
for (int k=1; k<=kma; k++) DDamax[k]=DDa_max;
for (int k=1; k<=kma; k++) DDamin[k]=DDa_min;
}

vse=0;   //Для збереження стартових даних vse=1  // Для збереження остаточного результату vse=2;

for (int k=1; k<=kma; k++) DDa[5][k]=DDa[0][k];     //Стартовий профіль (для графіка)
for (int k=1; k<=kma; k++) DLa[5][k]=DLa[0][k];     //Стартовий профіль (для графіка)
      L=0.;
      for (int k=1; k<=kma;k++) L=L+DLa[5][k];
      hpl0=hpl-L;

//  goto m99999;
lich_sikle=0;

m226:
   Lich_na_DD=0;
   method_lich=0;
if (lich_sikle==0) vse=1;

if (RadioButton62->Checked==true)
{
  for (int k=1; k<=kma; k++)
  {
     d_DDa[k]=DDa[0][k]*koef_d_DD;
     d_DL[k] =DLa[0][k]*koef_d_DD;
  //if (CheckBox16->Checked==true) STEP[4]=0;
  //if (CheckBox17->Checked==true) STEP[1]=0;
  }
}

List1->Add("         00  ");
List1->Add("    a D+-00  ");
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k]));
if (CheckBox77->Checked==true) Form2->Memo1-> Lines->Add("k  DDa[0][k]   DLa[0][k]");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
if (CheckBox77->Checked==true) Form2->Memo1-> Lines->Add("k  d_DDa[k]   d_DL[k]");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(d_DL[k])+'\t'+FloatToStr(d_DDa[k]));

        CALCULATION();
List1->Add(FloatToStr(CKV));
     SKV[0]=CKV;
if(vse==1)    // Задали розмір при першиму входженні в підпрограму
{
m=2*kma;
n=nom;
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("nom/n/m "+IntToStr(nom)+IntToStr(n)+'\t'+IntToStr(m));
if (CheckBox77->Checked==true) Form2->Memo1-> Lines->Add("nom/n/m "+IntToStr(nom)+IntToStr(n)+'\t'+IntToStr(m));

PZ0    = new double[n+1];        // [n]
YYY    = new double[n+1];        // [n]
PZ     = new double[n+1];        // [n]
for (int j=0; j<=n; j++)
  {
  PZ0[j]    = 1e-20;
  YYY[j]    = 1e-20;
  PZ[j]     = 1e-20;
  }

NSA = new double[m+1];         // [m]
NRA = new double[m+1];         // [m]
for (int j=0; j<=m; j++)
  {
  NSA[j]    = 1e-20;
  NRA[j]    = 1e-20;
  }

DIDD = new double*[n+1];           // [n][m]
for(int i=1;i<=n; i++)
{
    DIDD[i]  = new double[m+1];
}
for (int k=0; k<=m; k++)          // Ініціалізація DIDD[j][k]
  for (int j=1; j<=n; j++)
    DIDD[j][k]=1e-20;

DIDDTRAN = new double*[m+1];       // [m][m]   має бути  [m][n]
DOB2     = new double*[m+1];       // [m][m]   має бути  [m][n]
for(int i=1;i<=m; i++)
  {
  DIDDTRAN[i] = new double[n+1];
  DOB2[i]     = new double[n+1];
  }
for (int k=1; k<=m; k++)          // Ініціалізація
  for (int j=0; j<=n; j++)
    {
    DIDDTRAN[k][j] = 1e-20;
    DOB2[k][j]     = 1e-20;
    }

DOB1 = new  double*[m+1];           // [m][m]
X    = new  double*[m+1];           // [m][m]
OBER = new  double*[m+1];           // [m][m]
for(int i=1;i<=m; i++)
{
    DOB1[i]  = new  double[m+1];
    X[i]     = new  double[m+1];
    OBER[i]  = new  double[m+1];
}
for (int k=1; k<=m; k++)          // Ініціалізація
  for (int j=0; j<=m; j++)
    {
    DOB1[k][j]  = 1e-20;
    X[k][j]     = 1e-20;
    OBER[k][j]  = 1e-20;
    }

AA = new double*[m+1];             // [m][2*m]
for(int i=1;i<=m; i++)
{
    AA[i]  = new double[2*m+1];
}
for (int k=1; k<=m; k++)          // Ініціалізація
  for (int j=0; j<=2*m; j++)
    {
    DOB1[k][j]  = 1e-20;
    }

YYYY = new double*[n+1];       // [n][1]
for(int i=1;i<=n; i++)
{
    YYYY[i]  = new double[2];
}
for (int k=1; k<=n; k++)          // Ініціалізація
  for (int j=0; j<=1; j++)
    {
    YYYY[k][j]  = 1e-20;
    }

TTTT = new double*[m+1];       // [m][1]
for(int i=1;i<=m; i++)
{
    TTTT[i]  = new double[2];
}
for (int k=1; k<=m; k++)          // Ініціалізація
  for (int j=0; j<=1; j++)
    {
    TTTT[k][j]  = 1e-20;
    }

CHANGE = new double*[4];       // [3][m]
for(int i=1;i<=3; i++)
{
    CHANGE[i]  = new double[m+1];
}
for (int k=1; k<=3; k++)          // Ініціалізація
  for (int j=0; j<=m; j++)
    {
    CHANGE[k][j]  = 1e-20;
    }

}

for (int i=1; i<=n; i++) PZ0[i]=R_vseZa[i]; //for (int i=0; i<=m1; i++) PZ0[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ0");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(PZ0[j]));

if (lich_sikle==0)
{
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add(" n/m "+IntToStr(n)+'\t'+IntToStr(m));
Memo1-> Lines->Add(" Стартові дані: СКВ="+FloatToStr( SKV[0]));
//for (int kr=0; kr<=km_rozv;kr++)  Memo1->Lines->Add("Asd"+'\t'+FloatToStr(nn_m[kr])+'\t'+FloatToStr(DFi[kr])+'\t'+FloatToStr(kr));
vse=0;
}
   Lich_na_DD=1;

//    Обчислення масиву Y.      Іексп-Ітеор, стартовий профіль
for (int j=1; j<=n; j++) YYY[j]=PE[j]-PZ0[j] ;//YYY[j]=PE[j+jj]-PZ0[j+jj] ;
      np=0;
      SKV[1]=1000.;
      S2=0.;
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya YYY");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(YYY[j]));
//  goto m99999;

//      do 241 k=1,km
for (int k=1; k<=kma; k++)   //Стрибок у бік + або - по DD:
{
      DDa[0][k]=DDa[0][k]+d_DDa[k];

List1->Add("    a D+-1  ");
if (CheckBox47->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("    a D+-1  ");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k]));
         CALCULATION();
List1->Add(FloatToStr(CKV));
        SKV[10]=CKV;            //    SKV[10]  === skvp
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i]; //for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DD+");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+FloatToStr(PZ[j]));
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("DIDD[j][k]=PZ[j+jj]-PZ0[j+jj] DD+");

     for (int j=1; j<=n; j++)
{
          DIDD[j][k]=PZ[j]-PZ0[j];//DIDD[j][k]=PZ[j+jj]-PZ0[j+jj];
}
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+FloatToStr(DIDD[j][k]));
        if (SKV[10]<SKV[1])
{
          if (DDa[0][k]<=DDamax[k])
  {
            SKV[1]=SKV[10];
            np=k;
  }
}
        DDa[0][k]=DDa[0][k]-2.*d_DDa[k];

List1->Add("    a D+-2  ");
if (CheckBox47->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("    a D+-2  ");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k]));
        CALCULATION();
List1->Add(FloatToStr(CKV));
        SKV[11]=CKV;            //    SKV[11]  === skvm
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];  //for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DD-");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+FloatToStr(PZ[j]));

          for (int j=1; j<=n; j++)
{
        DIDD[j][k]=(DIDD[j][k]-PZ[j]+PZ0[j])/2.; // DIDD[j][k]=(DIDD[j][k]-PZ[j+jj]+PZ0[j+jj])/2.;
}
        if (SKV[11]<SKV[1])
{
          if (DDa[0][k]>=DDamin[k])
{
	    SKV[1]=SKV[11];
            np=-k;
 }
 }
        DDa[0][k]=DDa[0][k]+d_DDa[k];
        CHANGE[2][k]=SKV[11]-SKV[10];
        S2=S2+CHANGE[2][k]*CHANGE[2][k];
}   //241  end do

//      do 242 k=1,km
for (int k=1; k<=kma; k++)         //Стрибок у бік + або - по DLa:
{
        DLa[0][k]=DLa[0][k]+d_DL[k];
     hpl0=hpl0-d_DL[k];    ////        DL0=DL0-dDL;
List1->Add("    a L+-1  ");
if (CheckBox47->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("    a L+-1  ");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k]));
         CALCULATION();
List1->Add(FloatToStr(CKV));
        SKV[10]=CKV;            //    SKV[10]  === skvp
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];    //for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DL+");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+FloatToStr(PZ[j]));
for (int j=1; j<=n; j++)
{
	          DIDD[j][k+kma]=PZ[j]-PZ0[j];// DIDD[j][k+kma]=PZ[j+jj]-PZ0[j+jj];
}
        if (SKV[10]<SKV[1])
{
          SKV[1]=SKV[10];
          np=k+kma;
}
        DLa[0][k]=DLa[0][k]-2.*d_DL[k];
     hpl0=hpl0+2*d_DL[k];    ////        DL0=DL0+2*dDL;
        if (DLa[0][k]<=nenull_d_DL)             // В оригіналі  if (DLa[0][k]<0)
{
	        CHANGE[2][k+kma]=2.*(SKV[0]-SKV[10]);
          goto m54;
}
List1->Add("    a L+-2  ");
if (CheckBox47->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("    a L+-2  ");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method_lich][k])+'\t'+FloatToStr(DLa[method_lich][k]));
         CALCULATION();
List1->Add(FloatToStr(CKV));
        SKV[11]=CKV;            //    SKV[10]  === skvp
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];  //for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DL-");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+FloatToStr(PZ[j]));
        if (SKV[11]<SKV[1])
{
          if (DLa[0][k]>=nenull_d_DL)
{
            SKV[1]=SKV[11];
            np=-k-kma;
}
}
        CHANGE[2][k+kma]=SKV[11]-SKV[10];

        for (int j=1; j<=n; j++)
{
	          DIDD[j][k+kma]=(DIDD[j][k+kma]-PZ[j]+PZ0[j])/2.;//DIDD[j][k+kma]=(DIDD[j][k+kma]-PZ[j+jj]+PZ0[j+jj])/2.;
}
m54:      DLa[0][k]=DLa[0][k]+d_DL[k];
//         if (DLa[0][k]<=0) DLa[0][k]=DLa[0][k]+d_DL[k]/2.;       //   Щоб не було від'ємних розмірів
//         if (DLa[0][k]<=0) DLa[0][k]=DLa[0][k]+d_DL[k]/2.;       //   Щоб не було від'ємних розмірів
     hpl0=hpl0-d_DL[k];    ////        DL0=DL0-dDL;
        S2=S2+CHANGE[2][k+kma]*CHANGE[2][k+kma];
}    //242   end do
      S2=sqrt(S2);
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("  end  a D,L+-1,2  ");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[0][k])+'\t'+FloatToStr(DDa[0][k]));

//if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add("кінець змінам параметрів");
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DIDD");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(DIDD[j][1])+'\t'+FloatToStr(DIDD[j][2])+'\t'+FloatToStr(DIDD[j][3])+'\t'+FloatToStr(DIDD[j][4])+'\t'+FloatToStr(DIDD[j][5])+'\t'+FloatToStr(DIDD[j][6])+'\t'+FloatToStr(DIDD[j][7]));


//************************* 1
if (CheckBox49->Checked==true)
  {
  nenull=StrToFloat(Edit148->Text);
  for (int k=1; k<=m; k++)
    for (int j=1; j<=n; j++)
      if (fabs(DIDD[j][k])<=nenull)
        {
        if (DIDD[j][k]<0)  DIDD[j][k]=-nenull;
        if (DIDD[j][k]>=0)  DIDD[j][k]=nenull;
        }
  }
if (CheckBox30->Checked==true) nenull_d_DL=StrToFloat(Edit241->Text);
//*************************  2
//    Транспонована до DIDD матриця DIDDTRAN (А'):
Form2->Memo1-> Lines->Add("m/n");
Form2->Memo1-> Lines->Add(IntToStr(m)+'\t'+IntToStr(n));
for (int k=1; k<=m; k++)
{
for (int j=1; j<=n; j++)
{
//Form2->Memo1-> Lines->Add(IntToStr(k)+'\t'+IntToStr(j));
          DIDDTRAN[k][j]=DIDD[j][k];
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DIDDTRAN");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DIDDTRAN[j][1])+'\t'+FloatToStr(DIDDTRAN[j][2])+'\t'+FloatToStr(DIDDTRAN[j][3])+'\t'+FloatToStr(DIDDTRAN[j][4])+'\t'+FloatToStr(DIDDTRAN[j][5])+'\t'+FloatToStr(DIDDTRAN[j][6])+'\t'+FloatToStr(DIDDTRAN[j][7]));

//     Добуток А'xА=DOB1:
for (int j=1; j<=m; j++)
  {
  for (int k=1; k<=m; k++)
    {
    DOB1[j][k]=0;
    for (int i=1; i<=n; i++)
      {
      DOB1[j][k]=DOB1[j][k]+DIDDTRAN[j][i]*DIDD[i][k];
      }
    }
  }
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DOB1");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DOB1[j][1])+'\t'+FloatToStr(DOB1[j][2])+'\t'+FloatToStr(DOB1[j][3])+'\t'+FloatToStr(DOB1[j][4])+'\t'+FloatToStr(DOB1[j][5])+'\t'+FloatToStr(DOB1[j][6])+'\t'+FloatToStr(DOB1[j][7]));

//   Обурнена матриця  (A'xA)-1=OBER:
//   Пошук матриці X, оберненої до матриці AA розміром mxm,
//   методом Гауса з допомогою розширеної матриці AA(mx2m):
//   i(j) - Nr рядка (стовпчика) матриці AA
//   k"kkk"(ll) - Nr рядка (стовпчика) максимального елемента матриці AA
//   m1 -"m11" ранг решти матриці AA (нові координати максимального елемента матриці AA)

for (int i=1; i<=m; i++)
{
for (int j=1; j<=m; j++)
{
          AA[i][j]=DOB1[i][j];
}
}
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add("Utvorennya AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7]));

//      Утворюємо розширену матрицю  A:
for (int i=1; i<=m; i++)
  {
  for (int j=1; j<=m; j++)
    {
    if (j==i)
      {
      AA[i][j+m]=1;
      }
      else
      {
      AA[i][j+m]=0;
      }
    }
    NSA[i]=i;
    NRA[i]=i;
  }
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvor. rozshyrenu AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));

//      do 9914 m1=1,m
for (int m11=1; m11<=m; m11++)
{
	        if (m11==m) goto m9922;
//      Шукаємо AA(i,j)max:
        Amax=0;

for (int i=m11; i<=m; i++)
  {
  for (int j=m11; j<=m; j++)
    {
    if (Amax<fabs(AA[i][j]))
      {
      Amax=fabs(AA[i][j]);
      kkk=i;
      ll=j;
      }
    }
  }

//      Переставляємо А(i,j)max в кут:
        if (ll==m11) goto m9921;
//c       - перестановка стовпчиків:
for (int i=1; i<=m; i++)
{
          aAa=AA[i][m11];
          AA[i][m11]=AA[i][ll];
          AA[i][ll]=aAa;
}
        ns=NSA[m11];
        NSA[m11]=NSA[ll];
        NSA[ll]=ns;
m9921:
        if (kkk==m11) goto m9922;
//c       - перестановка рядків:
for (int j=m11; j<=m; j++)
{
          aAa=AA[m11][j];
          AA[m11][j]=AA[kkk][j];
          AA[kkk][j]=aAa;
}
for (int j=1; j<=m; j++)
{
	        aAa=AA[m11][j+m];
          AA[m11][j+m]=AA[kkk][j+m];
          AA[kkk][j+m]=aAa;
}
        nr=NRA[m11];
        NRA[m11]=NRA[kkk];
        NRA[kkk]=nr;
m9922:
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("After perestanovka v AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("  AA[m11][m11]= "+'\t'+FloatToStr(AA[m11][m11])+'\t'+FloatToStr (m11));

//       Ділимо m-тий рядок розштреної матриці на A на  A(m,m):
if (CheckBox49->Checked==true) if (AA[m11][m11]<=nenull)  AA[m11][m11]=nenull;      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for (int i=1; i<=m11; i++)
{
             nrai=NRA[i];
         AA[m11][nrai+m]=AA[m11][nrai+m]/AA[m11][m11];
}
        for (int j=m; j>=m11; j--)  AA[m11][j]=AA[m11][j]/AA[m11][m11];

        if (m11==m) goto m9914 ;
	      for (int i=m11+1; i<=m; i++)
{
	      for (int j=1; j<=m11; j++)
{
             nraj=NRA[j];
       	AA[i][nraj+m]=AA[i][nraj+m]-AA[i][m11]*AA[m11][nraj+m];
}
        for (int j=m; j>=1; j--)
{
	            AA[i][j]=AA[i][j]-AA[i][m11]*AA[m11][j];
}
m9918:
}
m9914:
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("end AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("endendendendend AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));

//     Пошук матриці X:
    for (int j=1; j<=m; j++)  X[m][j]=AA[m][j+m];
    for (int i=m-1; i>=1; i--)
{
    for (int j=1; j<=m; j++)
{
	          s=0.;
    for (int k=i+1; k<=m; k++)
{
	            s=s+AA[i][k]*X[k][j];
}
           X[i][j]=AA[i][j+m]-s;
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks X");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(X[j][1])+'\t'+FloatToStr(X[j][2])+'\t'+FloatToStr(X[j][3])+'\t'+FloatToStr(X[j][4])+'\t'+FloatToStr(X[j][5])+'\t'+FloatToStr(X[j][6])+'\t'+FloatToStr(X[j][7])+'\t'+FloatToStr(X[j][8]));

    for (int i=1; i<=m; i++)
{
	   for (int j=1; j<=m; j++)
{
             nsai=NSA[i];
          OBER[nsai][j]=X[i][j];
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks OBER");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(OBER[j][1])+'\t'+FloatToStr(OBER[j][2])+'\t'+FloatToStr(OBER[j][3])+'\t'+FloatToStr(OBER[j][4])+'\t'+FloatToStr(OBER[j][5])+'\t'+FloatToStr(OBER[j][6])+'\t'+FloatToStr(OBER[j][7])+'\t'+FloatToStr(OBER[j][8]));

//     Добуток (А'xА)-1xА'=DOB2:
     for (int j=1; j<=m; j++)
{
     for (int k=1; k<=n; k++)
{
	          DOB2[j][k]=0.;
     for (int i=1; i<=m; i++)
{
     DOB2[j][k]=DOB2[j][k]+OBER[j][i]*DIDDTRAN[i][k];
}
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks Dob2");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DOB2[j][1])+'\t'+FloatToStr(DOB2[j][2])+'\t'+FloatToStr(DOB2[j][3])+'\t'+FloatToStr(DOB2[j][4])+'\t'+FloatToStr(DOB2[j][5])+'\t'+FloatToStr(DOB2[j][6])+'\t'+FloatToStr(DOB2[j][7]));
if (CheckBox77->Checked==true) Form2->Memo1-> Lines->Add("Matriks Dob2");
if (CheckBox77->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DOB2[j][1])+'\t'+FloatToStr(DOB2[j][2])+'\t'+FloatToStr(DOB2[j][3])+'\t'+FloatToStr(DOB2[j][4])+'\t'+FloatToStr(DOB2[j][5])+'\t'+FloatToStr(DOB2[j][6])+'\t'+FloatToStr(DOB2[j][7]));

//      Добуток [(A'xA)-1xA']xY=TTTT:
//      TTTT - матриця поправок до профілю  dD/D(z)
      for (int i=1; i<=n; i++)    YYYY[i][1]=YYY[i];
      S3=0.;
     for (int k=1; k<=m; k++)
{
        TTTT[k][1]=0.;
     for (int i=1; i<=n; i++)
{
        TTTT[k][1]=TTTT[k][1]+DOB2[k][i]*YYYY[i][1];
}
        S3=S3+TTTT[k][1]*TTTT[k][1];
}
      S3=sqrt(S3);
     for (int k=1; k<=kma; k++)
{
    if (DLa[0][k]<=nenull_d_DL) TTTT[k][1]=0;           //!!!!!!!!!!!!!!!! Порівняння не цілих !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}					        	// А коли це можливе у програмі???
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("start CHANGE");
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add(FloatToStr(np)+'\t'+FloatToStr(S2)+'\t'+FloatToStr(S3));
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(d_DL[k])+'\t'+FloatToStr(d_DDa[k]));

//     Вектор зміни профілю для методу стрибків у сторону:
      for (int k=1; k<=m; k++)   CHANGE[1][k]=0;
      npn=fabs(np);
      if (npn<=kma)
{
        CHANGE[1][npn]=npn/np*d_DDa[npn];
}       else
{
        CHANGE[1][npn]=npn/np*d_DL[npn-kma];
}

if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("end CHANGE 1");

//     Вектор зміни профілю для градієнтного методу:
       for (int k=1; k<=kma; k++)
{
        CHANGE[2][k   ]=CHANGE[2][k   ]/S2;
        CHANGE[2][k+kma]=CHANGE[2][k+kma]/S2;
        CHANGE[2][k   ]=CHANGE[2][k   ]*d_DDa[k];
        CHANGE[2][k+kma]=CHANGE[2][k+kma]*d_DL[k];
}
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("end CHANGE 2");

//     Вектор зміни профілю для методу найменших квадратів Гауса:
      for (int k=1; k<=kma; k++)
{
        CHANGE[3][k   ]=TTTT[k   ][1]/S3;
        CHANGE[3][k+kma]=TTTT[k+kma][1]/S3;
        CHANGE[3][k   ]=CHANGE[3][k   ]*d_DDa[k];
        CHANGE[3][k+kma]=CHANGE[3][k+kma]*d_DL[k];
}
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("CHANGE");
if (CheckBox77->Checked==true) for (int k=1; k<=2*kma; k++) Form2->Memo1->Lines->Add(FloatToStr(1)+'\t'+FloatToStr(CHANGE[1][k]));
if (CheckBox77->Checked==true) for (int k=1; k<=2*kma; k++) Form2->Memo1->Lines->Add(FloatToStr(2)+'\t'+FloatToStr(CHANGE[2][k]));
if (CheckBox77->Checked==true) for (int k=1; k<=2*kma; k++) Form2->Memo1->Lines->Add(FloatToStr(3)+'\t'+FloatToStr(CHANGE[3][k]));

//     Пошук оптимального кроку у визначеному напрямку для кожного з методів:
for (int method=1; method<=3; method++)      // do 222 method=1,3
  {
  NN[method]=0;
  SKV[method]=SKV[0];
  for (int k=1; k<=kma; k++)
    {
    DDa[method][k]=DDa[0][k];
    DLa[method][k]=DLa[0][k];
    }

m221:  for (int k=1; k<=kma; k++)
    {
    DDa[4][k]=DDa[method][k]+CHANGE[method][k];
    DLa[4][k]=DLa[method][k]+CHANGE[method][k+kma];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("for (int method=1; method<=3; method++)");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(method)+'\t'+FloatToStr(CHANGE[method][k]));
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[4][k])+'\t'+FloatToStr(DDa[4][k]));

    if (DDa[4][k]<DDamin[k]) DDa[4][k]=DDamin[k];
    if (DDa[4][k]>DDamax[k]) DDa[4][k]=DDamax[k];
    if (DLa[4][k]<=nenull_d_DL) DLa[4][k]=DLa[method][k]-CHANGE[method][k+kma];       //   Щоб не було від'ємних розмірів    if (DLa[4][k]<0.) DLa[4][k]=0.;
    //          if (DLa[4][k]<=0) DLa[4][k]=DLa[4][k]+d_DL[k]/2.;       //   Щоб не було від'ємних розмірів    if (DLa[4][k]<0.) DLa[4][k]=0.;
    //          if (DLa[4][k]<=0) DLa[4][k]=DLa[4][k]+d_DL[k]/2.;       //   Щоб не було від'ємних розмірів    if (DLa[4][k]<0.) DLa[4][k]=0.;
    }
  L=0.;
  for (int k=1; k<=kma;k++) L=L+DLa[4][k] ;
  hpl0=hpl-L;
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[4][k])+'\t'+FloatToStr(DDa[4][k]));
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" END program 1");
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" END program 1");

  method_lich=4;
  CALCULATION();       //Обчисл. за модифікованими за певним методом парам. ПШ
List1->Add("         Обчисл. за модифікованими за певним методом ");
List1->Add(FloatToStr(method)+'\t'+FloatToStr(CKV));
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("  CALCULATION()  method   CKV");
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(FloatToStr(method)+'\t'+FloatToStr(CKV));

  SKV[4]=CKV;
  for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];  //for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];

  if (SKV[4]<SKV[method])
    {
    for (int k=1; k<=kma; k++)
      {
      DDa[method][k]=DDa[4][k];
      DLa[method][k]=DLa[4][k];
      }
      SKV[method]=SKV[4];
      NN[method]=NN[method]+1;
if (CheckBox77->Checked==true)  Form2->Memo1->Lines->Add("Іде у даному напрямку");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[method][k])+'\t'+FloatToStr(DDa[method][k]));
if (CheckBox78->Checked==true) Form2->Memo1->Lines->Add(IntToStr(lich_sikle+1)+'\t'+IntToStr(NN[method])+'\t'+IntToStr(method));
if (CheckBox78->Checked==true)for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add(FloatToStr(DLa[method][k])+'\t'+FloatToStr(DDa[method][k]));
//for (int k=1; k<=kma; k++) Memo1->Lines->Add("");
      if (CheckBox21->Checked==true)  goto m221;   //Іде у даному напрямку доки зменшується СКВ
    }

  if (CheckBox25->Checked==false) goto promunaepol;  //Проминає половинення!!!!!!!!!!!!!!
  if (NN[method]==0)    // Якщо метод не дав користі ні на 1 кроці, то половинить крок
    {
m223: NN[method]=NN[method]-1;
    for (int k=1; k<=m; k++)
      {
      CHANGE[method][k]=CHANGE[method][k]/2.;
      }
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" Кінець 6574657 1");
    for (int k=1; k<=kma; k++)
      {
      DDa[method][k]=DDa[0][k]+CHANGE[method][k];
      DLa[method][k]=DLa[0][k]+CHANGE[method][k+kma];
      if (DDa[method][k]<DDamin[k]) DDa[method][k]=DDamin[k];
      if (DDa[method][k]>DDamax[k]) DDa[method][k]=DDamax[k];
      if (DLa[method][k]<=nenull_d_DL) DLa[method][k]=DLa[0][k]-CHANGE[method][k+kma];       //   Щоб не було від'ємних розмірів    if (DLa[4][k]<0.) DLa[4][k]=0.;
//            if (DLa[method][k]<=0) DLa[method][k]=DLa[method][k]+d_DL[k]/2.;     //   Щоб не було від'ємних розмірів   if (DLa[method][k]<0.   ) DLa[method][k]=0.;
//            if (DLa[method][k]<=0) DLa[method][k]=DLa[method][k]+d_DL[k]/2.;     //   Щоб не було від'ємних розмірів   if (DLa[method][k]<0.   ) DLa[method][k]=0.;
      L=L+DLa[method][k];
      }
    L=0.;
    for (int k=1; k<=kma;k++) L=L+DLa[method][k] ;
    hpl0=hpl-L;


List1->Add("    a D  [4+method]  ");
for (int k=1; k<=kma; k++) List1->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[method][k])+'\t'+FloatToStr(DLa[method][k]));

if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" Кінець програми 2");
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" Кінець програми 2");

    method_lich=method;
    CALCULATION();
List1->Add(FloatToStr(CKV));
    SKV[method]=CKV;
    for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];   //        for (int i=0; i<=m1; i++) PZ[i]=R_vseZa[i];

    if (SKV[method]<SKV[0]) goto m222;        // Якщо дало покращення, то виходить з половинення і іде з поділеним кроком на новий круг
    if (NN[method]> -6) goto m223;         // Якщо не дало покращення, то половинить і перевіряє (до 6 разів) , а тоді виходить
    }
promunaepol:
m222:
  }
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("END  for (int method=1; method<=3; method++)");

//     Визначення оптимального профілю:
//////      SKV[11]=min(SKV[0],SKV[1],SKV[2],SKV[3]);
double  minSKV;
 minSKV=SKV[0];
if ( SKV[0]>SKV[1]) minSKV=SKV[1];
if ( SKV[1]>SKV[2]) minSKV=SKV[2];
if ( SKV[2]>SKV[3]) minSKV=SKV[3];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("optim profil");
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(FloatToStr(SKV[1])+'\t'+FloatToStr(SKV[2])+'\t'+FloatToStr(SKV[3]));

if (fabs(SKV[0]-minSKV)<riznSKV)      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Порівняння не цілих !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  {
  goto m99999;
  }
  else
  {
  for (int method=1; method<=3; method++)
    if (fabs(SKV[method]-minSKV)<riznSKV)
    {
    method_= method;
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("Large sikle");
    goto m225;  //Якщо результат покращився, то стартовому мінімальне СКВ і на наст. великий цикл
    }                                          //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Порівняння не цілих !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  }
m225:
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("Perelik metodiv");

if (method_==1) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод конфігурацій (стрибків) на " + IntToStr( NN[1])+" стрибки: СКВ="+FloatToStr( SKV[1]));

if (method_==2) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод градієнтний на " + IntToStr( NN[2])+" вектори: СКВ="+FloatToStr( SKV[2]));

if (method_==3) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод найм.квадратів Гаусса на " + IntToStr( NN[3])+" вектори: СКВ="+FloatToStr( SKV[3]));

      for (int k=1; k<=kma; k++)
{
        DDa[0][k]=DDa[method_][k];
        DLa[0][k]=DLa[method_][k];
}
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add("Na noll");
if (CheckBox77->Checked==true) for (int k=1; k<=kma; k++) Form2->Memo1->Lines->Add (FloatToStr(k)+'\t'+FloatToStr(DDa[method_][k])+'\t'+FloatToStr(DLa[method_][k]));

        lich_sikle=lich_sikle+1;
       if (lich_sikle==lich_sikleK)  goto m99999;

        goto m226;
m99999:

         method_lich=method_;
     vse=2;
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" Кінець обчислення");
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" Кінець обчислення");
       CALCULATION();
List1->Add(FloatToStr(CKV));
Memo1-> Lines->Add(" Кінець обчислення: СКВ="+FloatToStr(CKV));
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" !!!!Кінець обчислення");

// List1->SaveToFile("dataDD.dat");
  delete d_DDa, d_DL, DDamax, DDamin;
  delete PZ0, YYY, PZ;
  delete NSA, NRA;
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 111111");

for(int i=1; i<n; i++)
{
  delete[] DIDD[i];
  delete[] YYYY[i];
}
delete[] DIDD;
delete[] YYYY;
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222");

for(int i=1; i<m; i++)
{
  delete[] DIDDTRAN[i];
  delete[] DOB2[i];
  delete[] DOB1[i];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222aaaa1");
//  delete[] X[i];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222aaaa2");
  delete[] OBER[i];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222aaaa3");
  delete[] AA[i];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222aaaa4");
  delete[] TTTT[i];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222aaaa5");
}
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22222bbbbb");
  delete[] DIDDTRAN;
  delete[] DOB2;
  delete[] DOB1;
  delete[] OBER;
  delete[] AA;
  delete[] TTTT;

if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 11");
//  for(int i=2; i<m; i++)  delete[] X[i];
/*  delete[] X[0];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 22");
  delete[] X[1];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 33");
  delete[] X[3];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 44");
  delete[] X[4];
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 55");  */
  delete[] X;

if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 33333");
for(int i=1; i<=3; i++)
{
  delete[] CHANGE[i];
}
delete[] CHANGE;
if (CheckBox77->Checked==true) Form2->Memo1->Lines->Add(" 4444444");

}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//             ПІДПРОГРАМИ ДЛЯ   !Gausauto.for!
//---------------------------------------------------------------------------
void TForm1::CALCULATION_Gausauto()
{
//double EL[KM],DD0;
double Z_shod [2*KM],D_shod [2*KM],L_shod,nn_m1[100];
//double *Z_shod, *D_shod, L_shod, *nn_m1;

if (RadioButton30->Checked==true)
{       //Набл. профіль гаусіанами в ППШ (з врах. диф.розс. в  ід. част. монокр. та ППШ)
   Profil(km,DD,dl);
//  Z_shod    = new double[2*km+2];
//  D_shod    = new double[2*km+2];

	if (vse==2)    // запис результату наближення
	{
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;

for (int k=1; k<=2*km+1;k++) Series3->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clRed);
for (int k=1; k<=km;k++)
{
Series37->AddXY((km*dl-dl*k+dl/2.)/1e-8,DD[k],"",clRed);
Series38->AddXY((km*dl-dl*k+dl/2.)/1e-8,DDPL1[k],"",clRed);
Series39->AddXY((km*dl-dl*k+dl/2.)/1e-8,DDPL2[k],"",clRed);
}

Edit111 ->Text=FloatToStr(PARAM[method_][1]);
Edit112->Text=FloatToStr(PARAM[method_][2]);
Edit113->Text=FloatToStr(PARAM[method_][3]*1e8);
Edit114->Text=FloatToStr(PARAM[method_][4]*1e8);
Edit115->Text=FloatToStr(PARAM[method_][5]);
Edit116->Text=FloatToStr(PARAM[method_][6])*1e8;
Edit117->Text=FloatToStr(PARAM[method_][7])*1e8;
Edit118->Text=FloatToStr(PARAM[method_][8]);

Edit110->Text=FloatToStr(DDparam0[12]);          //   Dmax
Edit109->Text=FloatToStr(DDparam0[13]/1e-8);     //   L/1e-8
Edit108->Text=IntToStr((int)DDparam0[10]);       //   km

//Для запису в файл
PARAM[method_][3]=PARAM[method_][3]*1e8;
PARAM[method_][4]=PARAM[method_][4]*1e8;
PARAM[method_][6]=PARAM[method_][6]*1e8;
PARAM[method_][7]=PARAM[method_][7]*1e8;
PARAM[5][3]=PARAM[5][3]*1e8;
PARAM[5][4]=PARAM[5][4]*1e8;
PARAM[5][6]=PARAM[5][6]*1e8;
PARAM[5][7]=PARAM[5][7]*1e8;
STEP[3]=STEP[3]*1e8;
STEP[4]=STEP[4]*1e8;
STEP[6]=STEP[6]*1e8;
STEP[7]=STEP[7]*1e8;
}

	if (vse==1)    // //Запис вихідного профілю:
	{
//Profil(km,DD,dl);

L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
for (int k=1; k<=2*km+1;k++) Series34->AddXY(Z_shod[k]/1e-8,D_shod[k],"",clBlue);
for (int k=1; k<=km;k++)
{
Series40->AddXY((km*dl-dl*k+dl/2.)/1e-8,DD[k],"",clBlue);
Series41->AddXY((km*dl-dl*k+dl/2.)/1e-8,DDPL1[k],"",clBlue);
Series42->AddXY((km*dl-dl*k+dl/2.)/1e-8,DDPL2[k],"",clBlue);
}

}
}
if (RadioButton31->Checked==true)
{     //Набл. ф-ю розподілу розворотів блоків від кута у вигляді гаусіани (без врах. диф.розс. в  ід. част. монокр. та ППШ)
   Profil(km_rozv,nn_m1,dl);
km_rozv=km_rozv-1;
//nn_m1 = new double[km_rozv+2];
   for (int k=0; k<=km_rozv;k++) //  Перенумерація елементів масивів
   {
   nn_m[k]=nn_m1[km_rozv-k];
   DFi[k]=dl*1e8;
   }
	if (vse==2)    // запис результату наближення
	{
double    Snn,Afi,fff1[100],fi[100];
    Snn=0;                  // ii?ioa. ooieo?? ?iciia?eo ii eooao
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;
  Afi=StrToFloat(Edit80->Text); // Eiao. a  DD_rozv[kr] (fi[kr]);
  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
 for (int k=0; k<=km_rozv;k++)
  {
Series43->AddXY(fi[k],DD_rozv[k]*10000,"",clFuchsia);
Series22->AddXY(fi[k],fff1[k]*100,"",clFuchsia);
Memo2-> Lines->Add(FloatToStr(fi[k])+'\t'+FloatToStr(fff1[k]*100));
  }
//Запис вихідного профілю:
//lich_sikle=5;
//   Profil(km_rozv,nn_m1,dl);
km_rozv=km_rozv-1;
   for (int k=0; k<=km_rozv;k++) //  Перенумерація елементів масивів
   {
   nn_m[k]=nn_m1[km_rozv-k];
   DFi[k]=dl*1e8;
   }
    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;

  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
for (int k=0; k<=km_rozv;k++)
{
Series23->AddXY(fi[k],DD_rozv[k]*10000,"",clBlue);
Series44->AddXY(fi[k],fff1[k]*100,"",clBlue);
Memo5-> Lines->Add(FloatToStr(fi[k])+'\t'+FloatToStr(fff1[k]*100));
}

	}
}


//RozrachDiduz();
//RozrachKoger();
//Zgortka();
QuickStart();
//delete Z_shod, D_shod, nn_m1;
}
// -------------------      МІНІМІЗАЦІЯ СКВ   !Gausauto.for!    -----------------------
void __fastcall TForm1::Button17Click(TObject *Sender)
{
int  m,n, np, kkk,ll,ns,nr,nrai,nraj,nsai,npn, lich_sikleK, NN[4];
int k_param ,kchange ;
double  S2 , S3,Amax ,nenull,riznSKV ;
double   aAa, s ;
double   SKV[12];
double *PZ0, *YYY, *PZ;                 // [n]
double **DIDD, **DIDDTRAN;              // [n][m], [m][n]
double **DOB1, **X, **OBER;             // [m][m]
double **AA;                            // [m][2*m]
double *NSA, *NRA;                      // [m]
double **DOB2, **YYYY, **TTTT;          // [m][n], [n][1], [m][1]
double **CHANGE;                        // [3][m]
//TStringList *List1 = new TStringList;

fitting=10;
riznSKV=StrToFloat(Edit139->Text);

if (RadioButton31->Checked==false) RadioButton30->Checked=true;

lich_sikleK=StrToInt(Edit73->Text);
k_param=StrToInt(Edit127->Text);
PARAM[0][1]=StrToFloat(Edit98->Text);
PARAM[0][2]=StrToFloat(Edit101->Text);
PARAM[0][3]=StrToFloat(Edit102->Text)*1e-8;
PARAM[0][4]=StrToFloat(Edit103->Text)*1e-8;
PARAM[0][5]=StrToFloat(Edit104->Text);
PARAM[0][6]=StrToFloat(Edit105->Text)*1e-8;
PARAM[0][7]=StrToFloat(Edit106->Text)*1e-8;
PARAM[0][8]=StrToFloat(Edit128->Text);
PARAM[0][13]=StrToFloat(Edit95->Text);       //   DDamax
PARAM[0][14]=StrToFloat(Edit107->Text);       //   DDamin
dl=1e-8*StrToFloat(Edit97->Text);
kEW=StrToFloat(Edit128->Text);                     //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

STEP[1]=StrToFloat(Edit119->Text);
STEP[2]=StrToFloat(Edit120->Text);
STEP[3]=StrToFloat(Edit121->Text)*1e-8;
STEP[4]=StrToFloat(Edit122->Text)*1e-8;
STEP[5]=StrToFloat(Edit123->Text);
STEP[6]=StrToFloat(Edit124->Text)*1e-8;
STEP[7]=StrToFloat(Edit125->Text)*1e-8;
STEP[8]=StrToFloat(Edit126->Text);

if (CheckBox28->Checked==true)
{
PARAM[0][1]=StrToFloat(Edit35->Text);
PARAM[0][2]=StrToFloat(Edit36->Text);
PARAM[0][3]=StrToFloat(Edit37->Text)*1e-8;
PARAM[0][4]=StrToFloat(Edit38->Text)*1e-8;
PARAM[0][5]=StrToFloat(Edit39->Text);
PARAM[0][6]=StrToFloat(Edit40->Text)*1e-8;
PARAM[0][7]=StrToFloat(Edit41->Text)*1e-8;
PARAM[0][14]=StrToFloat(Edit42->Text);
dl=1e-8*StrToFloat(Edit33->Text);
//kEW=StrToFloat(Edit2->Text);

Edit98->Text=FloatToStr(PARAM[0][1]);
Edit101->Text=FloatToStr(PARAM[0][2]);
Edit102->Text=FloatToStr(PARAM[0][3])*1e8;
Edit103->Text=FloatToStr(PARAM[0][4])*1e8;
Edit104->Text=FloatToStr(PARAM[0][5]);
Edit105->Text=FloatToStr(PARAM[0][6])*1e8;
Edit106->Text=FloatToStr(PARAM[0][7])*1e8;
Edit107->Text=FloatToStr(PARAM[0][14]);
Edit97->Text=FloatToStr(dl)*1e8;
//Edit106->Text=FloatToStr(kEW);
}

vse=0;   //Для збереження стартових даних vse=1  // Для збереження остаточного результату vse=2;

for (int i=1; i<=k_param;i++)  PARAM[5][i]=PARAM[0][i];    //Стартовий профіль

lich_sikle=0;

m226g:
   Lich_na_DD=0;
   method_lich=0;
if (lich_sikle==0) vse=1;
//List1->Add("         00  ");
//List1->Add("    a D+-00  ");

        CALCULATION_Gausauto();
        SKV[0]=CKV;
if(vse==1)    // Задали розмір при першиму входженні в підпрограму
{
m=k_param;
n=nom;

  PZ0    = new double[n+1];        // [n]
  YYY    = new double[n+1];        // [n]
  PZ     = new double[n+1];        // [n]

    NSA = new double[m+1];         // [m]
    NRA = new double[m+1];         // [m]


DIDD = new double*[n+1];           // [n][m]
for(int i=0;i<=n; i++)
{
    DIDD[i]  = new double[m+1];
}
for (int k=1; k<=m; k++)          // Ініціалізація DIDD[j][k]
{
for (int j=1; j<=n; j++)
{
    DIDD[j][k]=1e-20;;
}}

DIDDTRAN = new double*[m+1];       // [m][m]
DOB2     = new double*[m+1];       // [m][m]
for(int i=0;i<=m; i++)
{
    DIDDTRAN[i] = new double[n+1];
    DOB2[i]     = new double[n+1];
}

DOB1 = new  double*[m+1];           // [m][m]
X    = new  double*[m+1];           // [m][m]
OBER = new  double*[m+1];           // [m][m]
for(int i=0;i<=m; i++)
{
    DOB1[i]  = new  double[m+1];
    X[i]     = new  double[m+1];
    OBER[i]  = new  double[m+1];
}

AA = new double*[m+1];             // [m][2*m]
for(int i=0;i<=m; i++)
{
    AA[i]  = new double[2*m+1];
}

YYYY = new double*[n+1];       // [n][1]
for(int i=0;i<=n; i++)
{
    YYYY[i]  = new double[2];
}

TTTT = new double*[m+1];       // [m][1]
for(int i=0;i<=m; i++)
{
    TTTT[i]  = new double[2];
}

CHANGE = new double*[4];       // [3][m]
for(int i=0;i<=3; i++)
{
    CHANGE[i]  = new double[m+1];
}
}

for (int i=1; i<=n; i++) PZ0[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ0");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(PZ0[j]));

if (lich_sikle==0)
{
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add(" n/m "+IntToStr(n)+'\t'+IntToStr(m));
Memo1-> Lines->Add(" Стартові параметри : СКВ="+FloatToStr( SKV[0]));
vse=0;
}
   Lich_na_DD=1;

//    Обчислення масиву Y.      Іексп-Ітеор, стартовий профіль
for (int j=1; j<=n; j++) YYY[j]=PE[j]-PZ0[j] ;//YYY[j]=PE[j+jj]-PZ0[j+jj] ;
//for (int j=1; j<=n; j++) Memo1-> Lines->Add(FloatToStr(PE[j])+'\t'+FloatToStr(PZ0[j]));
      np=0;
      SKV[1]=1000.;
      S2=0.;
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya YYY");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(YYY[j]));
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("PARAM[0][pp], STEP[pp]");
if (CheckBox47->Checked==true) for (int pp=1; pp<=k_param; pp++) Form2->Memo1-> Lines->Add(FloatToStr(PARAM[0][pp])+'\t'+FloatToStr(STEP[pp]));

//      do 241 k=1,km
for (int pp=1; pp<=k_param; pp++)   //Стрибок у бік + або - по DD:
{
        kchange=1;
        PARAM[0][pp]=PARAM[0][pp]+STEP[pp];
//List1->Add("    a D+-1  ");

if (CheckBox40->Checked==false)   //false -включити обмеження
   {
//c Якщо D01>=Dmax1, то СКВ не обчислюю, а вважаю, що skvp-SKV(0)=SKV(0)-skvm:
        if (PARAM[0][2]>=PARAM[0][1])
{
          SKV[10]=SKV[0];             //    SKV[10]  === skvp
          kchange=2;
 for (int j=1; j<=n; j++)  DIDD[j][2]=0;
          goto m227g;
}
//c Якщо Rp1>=L1, то СКВ не обчислюю, а вважаю, що skvp-SKV(0)=SKV(0)-skvm:
        if (PARAM[0][4]>=PARAM[0][3])
{
          SKV[10]=SKV[0];
          kchange=2;
     for (int j=1; j<=n; j++)   DIDD[j][4]=0;
          goto m227g;
}
//c Якщо L2>L1, то СКВ не обчислюю, а вважаю, що skvp-SKV(0)=SKV(0)-skvm:
        if (PARAM[0][6]>PARAM[0][3])
{
          SKV[10]=SKV[0];
          kchange=2;
     for (int j=1; j<=n; j++)  DIDD[j][6]=0;
          goto m227g;
}
//c Якщо Rp2>=0, то СКВ не обчислюю, а вважаю, що skvp-SKV(0)=SKV(0)-skvm:
         if (PARAM[0][7]>=0)
{
	  SKV[10]=SKV[0];
          kchange=2;
     for (int j=1; j<=n; j++)   DIDD[j][7]=0;
          goto m227g;
}
//c  Якщо kEW<0 чи kEW>1, то СКВ не обчислюю, а вважаю, що skvp-SKV(0)=SKV(0)-skvm:
         if (PARAM[0][8]<0 || PARAM[0][8]>1 )
{
	  SKV[10]=SKV[0];
          kchange=2;
     for (int j=1; j<=n; j++)   DIDD[j][8]=0;
          goto m227g;
}
   }

         CALCULATION_Gausauto();
        SKV[10]=CKV;            //    SKV[10]  === skvp
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DD+");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(pp)+'\t'+FloatToStr(PZ[j]));
//if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("DIDD[j][k]=PZ[j+jj]-PZ0[j+jj] DD+");

     for (int j=1; j<=n; j++)
{
          DIDD[j][pp]=PZ[j]-PZ0[j]; //DIDD[j][pp]=PZ[j+jj]-PZ0[j+jj];
}
        if (SKV[10]<SKV[1])
{
            SKV[1]=SKV[10];
            np=pp;
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DIDD-поч.");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(DIDD[j][1])+'\t'+FloatToStr(DIDD[j][2])+'\t'+FloatToStr(DIDD[j][3])+'\t'+FloatToStr(DIDD[j][4])+'\t'+FloatToStr(DIDD[j][5])+'\t'+FloatToStr(DIDD[j][6])+'\t'+FloatToStr(DIDD[j][7]));

m227g:         PARAM[0][pp]=PARAM[0][pp]-2.*STEP[pp];
if (CheckBox40->Checked==false)   //false -включити обмеження
   {
//c Якщо Dmax1=<D01 або L1=<Rp1 або Rp1<0 або D02=<Dmin або L1<L2 або L2=<0,
//c то СКВ не обчислюю, а вважаю, що skvm-SKV(0)=SKV(0)-skvp:
if ( PARAM[0][1]<=PARAM[0][2] || PARAM[0][3]<=PARAM[0][4] || PARAM[0][4]<0
 ||  PARAM[0][5]<=PARAM[0][14] || PARAM[0][3]<PARAM[0][6] || PARAM[0][6]<=0
 ||  PARAM[0][8]<0 || PARAM[0][8]>1)
  {
          SKV[11]=SKV[0];       //    SKV[11]  === skvm
          kchange=kchange+1;
          goto m228g;
   }
   }
   
       CALCULATION_Gausauto();
        SKV[11]=CKV;            //    SKV[11]  === skvm
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Vyvid PZ DD-");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(IntToStr(pp)+'\t'+FloatToStr(PZ[j]));

          for (int j=1; j<=n; j++)
{
        DIDD[j][pp]=(DIDD[j][pp]-PZ[j]+PZ0[j])/2.*kchange; //DIDD[j][pp]=(DIDD[j][pp]-PZ[j+jj]+PZ0[j+jj])/2.*kchange;
}
        if (SKV[11]<SKV[1])
{
        SKV[1]=SKV[11];
            np=-pp;
}
m228g:     PARAM[0][pp]=PARAM[0][pp]+STEP[pp];

        CHANGE[2][pp]=(SKV[11]-SKV[10])*kchange;
        S2=S2+CHANGE[2][pp]*CHANGE[2][pp];
}     //241  end do
      S2=sqrt(S2);

if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DIDD");
if (CheckBox47->Checked==true) for (int j=1; j<=n; j++) Form2->Memo1-> Lines->Add(FloatToStr(DIDD[j][1])+'\t'+FloatToStr(DIDD[j][2])+'\t'+FloatToStr(DIDD[j][3])+'\t'+FloatToStr(DIDD[j][4])+'\t'+FloatToStr(DIDD[j][5])+'\t'+FloatToStr(DIDD[j][6])+'\t'+FloatToStr(DIDD[j][7]));

//************************* 1
if (CheckBox49->Checked==true)
{
nenull=StrToFloat(Edit148->Text);
for (int k=1; k<=m; k++)
{
for (int j=1; j<=n; j++)
{
 if (fabs(DIDD[j][k])<=nenull)
{
 if (DIDD[j][k]<0)  DIDD[j][k]=-nenull;
 if (DIDD[j][k]>=0)  DIDD[j][k]=nenull;
}
}
}
}//*************************  2
//    Транспонована до DIDD матриця DIDDTRAN (А'):
for (int k=1; k<=m; k++)
{
for (int j=1; j<=n; j++)
{
          DIDDTRAN[k][j]=DIDD[j][k];
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DIDDTRAN");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DIDDTRAN[j][1])+'\t'+FloatToStr(DIDDTRAN[j][2])+'\t'+FloatToStr(DIDDTRAN[j][3])+'\t'+FloatToStr(DIDDTRAN[j][4])+'\t'+FloatToStr(DIDDTRAN[j][5])+'\t'+FloatToStr(DIDDTRAN[j][6])+'\t'+FloatToStr(DIDDTRAN[j][7]));

//     Добуток А'xА=DOB1:
for (int j=1; j<=m; j++)
{
for (int k=1; k<=m; k++)
{
          DOB1[j][k]=0;
for (int i=1; i<=n; i++)
{
       DOB1[j][k]=DOB1[j][k]+DIDDTRAN[j][i]*DIDD[i][k];
}
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvorennya DOB1");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DOB1[j][1])+'\t'+FloatToStr(DOB1[j][2])+'\t'+FloatToStr(DOB1[j][3])+'\t'+FloatToStr(DOB1[j][4])+'\t'+FloatToStr(DOB1[j][5])+'\t'+FloatToStr(DOB1[j][6])+'\t'+FloatToStr(DOB1[j][7]));

//   Обурнена матриця  (A'xA)-1=OBER:
//   Пошук матриці X, оберненої до матриці AA розміром mxm,
//   методом Гауса з допомогою розширеної матриці AA(mx2m):
//   i(j) - Nr рядка (стовпчика) матриці AA
//   k"kkk"(ll) - Nr рядка (стовпчика) максимального елемента матриці AA
//   m1 -"m11" ранг решти матриці AA (нові координати максимального елемента матриці AA)

for (int i=1; i<=m; i++)
{
for (int j=1; j<=m; j++)
{
          AA[i][j]=DOB1[i][j];
}
}
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add("Utvorennya AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7]));

//      Утворюємо розширену матрицю  A:
for (int i=1; i<=m; i++)
{
for (int j=1; j<=m; j++)
{
	   if (j==i)
{
            AA[i][j+m]=1;
}          else
{
	            AA[i][j+m]=0;
}
}
        NSA[i]=i;
        NRA[i]=i;
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Utvor. rozshyrenu AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));

//      do 9914 m1=1,m
for (int m11=1; m11<=m; m11++)
{
	        if (m11==m) goto m9922g;
//      Шукаємо AA(i,j)max:
        Amax=0;

for (int i=m11; i<=m; i++)
{
for (int j=m11; j<=m; j++)
{
             if (Amax<fabs(AA[i][j]))
{
            Amax=fabs(AA[i][j]);
             kkk=i;
              ll=j; 
}
}
}

//      Переставляємо А(i,j)max в кут:
        if (ll==m11) goto m9921g;
//c       - перестановка стовпчиків:
for (int i=1; i<=m; i++)
{
          aAa=AA[i][m11];
          AA[i][m11]=AA[i][ll];
          AA[i][ll]=aAa;
}
        ns=NSA[m11];
        NSA[m11]=NSA[ll];
        NSA[ll]=ns;
m9921g:
        if (kkk==m11) goto m9922g;
//c       - перестановка рядків:
for (int j=m11; j<=m; j++)
{
          aAa=AA[m11][j];
          AA[m11][j]=AA[kkk][j];
          AA[kkk][j]=aAa;
}
for (int j=1; j<=m; j++)
{
	        aAa=AA[m11][j+m];
          AA[m11][j+m]=AA[kkk][j+m];
          AA[kkk][j+m]=aAa;
}
        nr=NRA[m11];
        NRA[m11]=NRA[kkk];
        NRA[kkk]=nr;
m9922g:
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("After perestanovka v AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("  AA[m11][m11]= "+'\t'+FloatToStr(AA[m11][m11])+'\t'+FloatToStr (m11));

//       Ділимо m-тий рядок розштреної матриці на A на  A(m,m):
if (CheckBox49->Checked==true) if (AA[m11][m11]<=nenull)  AA[m11][m11]=nenull;      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for (int i=1; i<=m11; i++)
{
             nrai=NRA[i];
         AA[m11][nrai+m]=AA[m11][nrai+m]/AA[m11][m11];
}
        for (int j=m; j>=m11; j--)  AA[m11][j]=AA[m11][j]/AA[m11][m11];

        if (m11==m) goto m9914g ;
	      for (int i=m11+1; i<=m; i++)
{
	      for (int j=1; j<=m11; j++)
{
             nraj=NRA[j];
       	AA[i][nraj+m]=AA[i][nraj+m]-AA[i][m11]*AA[m11][nraj+m];
}
        for (int j=m; j>=1; j--)
{
	            AA[i][j]=AA[i][j]-AA[i][m11]*AA[m11][j];
}
m9918g:
}
m9914g:
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("end AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("endendendendend AA");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(AA[j][1])+'\t'+FloatToStr(AA[j][2])+'\t'+FloatToStr(AA[j][3])+'\t'+FloatToStr(AA[j][4])+'\t'+FloatToStr(AA[j][5])+'\t'+FloatToStr(AA[j][6])+'\t'+FloatToStr(AA[j][7])+'\t'+FloatToStr(AA[j][8]));

//     Пошук матриці X:
    for (int j=1; j<=m; j++)  X[m][j]=AA[m][j+m];
    for (int i=m-1; i>=1; i--)
{
    for (int j=1; j<=m; j++)
{
	          s=0.;
    for (int k=i+1; k<=m; k++)
{
	            s=s+AA[i][k]*X[k][j];
}
           X[i][j]=AA[i][j+m]-s;
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks X");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(X[j][1])+'\t'+FloatToStr(X[j][2])+'\t'+FloatToStr(X[j][3])+'\t'+FloatToStr(X[j][4])+'\t'+FloatToStr(X[j][5])+'\t'+FloatToStr(X[j][6])+'\t'+FloatToStr(X[j][7])+'\t'+FloatToStr(X[j][8]));

    for (int i=1; i<=m; i++)
{
	   for (int j=1; j<=m; j++)
{
             nsai=NSA[i];
          OBER[nsai][j]=X[i][j];
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks OBER");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(OBER[j][1])+'\t'+FloatToStr(OBER[j][2])+'\t'+FloatToStr(OBER[j][3])+'\t'+FloatToStr(OBER[j][4])+'\t'+FloatToStr(OBER[j][5])+'\t'+FloatToStr(OBER[j][6])+'\t'+FloatToStr(OBER[j][7])+'\t'+FloatToStr(OBER[j][8]));

//     Добуток (А'xА)-1xА'=DOB2:
     for (int j=1; j<=m; j++)
{
     for (int k=1; k<=n; k++)
{
	          DOB2[j][k]=0.;
     for (int i=1; i<=m; i++)
{
     DOB2[j][k]=DOB2[j][k]+OBER[j][i]*DIDDTRAN[i][k];
}
}
}
if (CheckBox47->Checked==true) Form2->Memo1-> Lines->Add("Matriks Dob2");
if (CheckBox47->Checked==true) for (int j=1; j<=m; j++) Form2->Memo1-> Lines->Add(FloatToStr(DOB2[j][1])+'\t'+FloatToStr(DOB2[j][2])+'\t'+FloatToStr(DOB2[j][3])+'\t'+FloatToStr(DOB2[j][4])+'\t'+FloatToStr(DOB2[j][5])+'\t'+FloatToStr(DOB2[j][6])+'\t'+FloatToStr(DOB2[j][7]));

//      Добуток [(A'xA)-1xA']xY=TTTT:
//      TTTT - матриця поправок до профілю  dD/D(z)
      for (int i=1; i<=n; i++)    YYYY[i][1]=YYY[i];
      S3=0.;
     for (int k=1; k<=m; k++)
{
        TTTT[k][1]=0.;
     for (int i=1; i<=n; i++)
{
        TTTT[k][1]=TTTT[k][1]+DOB2[k][i]*YYYY[i][1];
}
        S3=S3+TTTT[k][1]*TTTT[k][1];
}
      S3=sqrt(S3);


//     Вектор зміни профілю для методу стрибків у сторону:
      for (int k=1; k<=m; k++)   CHANGE[1][k]=0;
      npn=fabs(np);
      CHANGE[1][npn]=npn/np*STEP[npn];

//     Вектор зміни профілю для градієнтного методу:
       for (int k=1; k<=m; k++)    CHANGE[2][k ]=CHANGE[2][k ]/S2*STEP[k];

//     Вектор зміни профілю для методу найменших квадратів Гауса:
      for (int k=1; k<=m; k++)     CHANGE[3][k ]=TTTT[k ][1]/S3*STEP[k];

//     Пошук оптимального кроку у визначеному напрямку для кожного з методів:
      for (int method=1; method<=3; method++)      // do 222 method=1,3
{
        NN[method]=0;
        SKV[method]=SKV[0];
      for (int k=1; k<=m; k++)     PARAM[method][k]=PARAM[0][k];

m221g:  for (int k=1; k<=m; k++) PARAM[4][k]=PARAM[method][k]+CHANGE[method][k];

if (CheckBox40->Checked==false)   //false -включити обмеження
   {
//c       Якщо D01>=Dmax1 або Rp1>=L1 або L2>L1 або Rp2>=0 або Rp1<0 або
//c       D02=<Dmin або L2=<0 або Dmax1>DDmax, то:
if (PARAM[4][2]>=PARAM[4][1] || PARAM[4][4]>=PARAM[4][3] || PARAM[4][6]>PARAM[4][3]
 || PARAM[4][7]>=0 || PARAM[4][4]<0 || PARAM[4][5]<=PARAM[0][14] || PARAM[4][6]<=0
 || PARAM[4][1]>PARAM[0][13] ||  PARAM[4][8]<0 || PARAM[4][8]>1)
       {
          if (NN[method]>0) goto promunaepol_g;
          if (NN[method]<-6) goto m222g;
          if (NN[method]<=0)
       {
            NN[method]=NN[method]-1;
            for (int pp=1; pp<=m; pp++)      CHANGE[method][pp]=CHANGE[method][pp]/2.;
            goto m221g;
       }
       } 
   }
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" Кінець програми 1");

     method_lich=4;
        CALCULATION_Gausauto();       //Обчисл. за модифікованими за певним методом парам. ПШ
        SKV[4]=CKV;
for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];

        if (SKV[4]<SKV[method])
{
      for (int k=1; k<=m; k++)  PARAM[method][k]=PARAM[4][k];
          SKV[method]=SKV[4];
          if (NN[method]<0  && CheckBox25->Checked==false ) goto promunaepol_g;
          NN[method]=NN[method]+1;
 if (CheckBox21->Checked==true)     goto m221g;                      //Іде у даному напрямку доки зменшується СКВ
}
if (CheckBox25->Checked==false) goto promunaepol_g;  //Проминає половинення!!!!!!!!!!!!!!
        if (NN[method]==0)    // Якщо метод не дав користі ні на 1 кроці, то половинить крок
{
m223g:       NN[method]=NN[method]-1;
  for (int k=1; k<=m; k++) 	  CHANGE[method][k]=CHANGE[method][k]/2.;
  for (int k=1; k<=m; k++)    PARAM[method][k]=PARAM[0][k]+CHANGE[method][k];

//List1->Add("    a D  [4+method]  ");
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" Кінець програми 2");

        method_lich=method;
        CALCULATION_Gausauto();
        SKV[method]=CKV;
        for (int i=1; i<=n; i++) PZ[i]=R_vseZa[i];

         if (SKV[method]<SKV[0]) goto m222g;        // Якщо дало покращення, то виходить з половинення і іде з поділеним кроком на новий круг
        if (NN[method]> -6) goto m223g;         // Якщо не дало покращення, то половинить і перевіряє (до 6 разів) , а тоді виходить
}
promunaepol_g:
m222g:
}

//     Визначення оптимального профілю:
//////      SKV[11]=min(SKV[0],SKV[1],SKV[2],SKV[3]);
double  minSKV;
 minSKV=SKV[0];
if ( SKV[0]>SKV[1]) minSKV=SKV[1];
if ( SKV[1]>SKV[2]) minSKV=SKV[2];
if ( SKV[2]>SKV[3]) minSKV=SKV[3];

      if (fabs(SKV[0]-minSKV)<riznSKV)                //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Порівняння не цілих !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{        goto m9999g;
}      else
{      for (int method=1; method<=3; method++)   if (fabs(SKV[method]-minSKV)<riznSKV)
{
   method_= method;
goto m225g;  //Якщо результат покращився, то стартовому мінімальне СКВ і на наст. великий цикл
}                                   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Порівняння не цілих !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}
m225g:

if (method_==1) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод конфігурацій (стрибків) на " + IntToStr( NN[1])+" стрибки: СКВ="+FloatToStr( SKV[1]));

if (method_==2) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод градієнтний на " + IntToStr( NN[2])+" вектори: СКВ="+FloatToStr( SKV[2]));

if (method_==3) Memo1-> Lines->Add(IntToStr(lich_sikle+1)+" Метод найм.квадратів Гаусса на " + IntToStr( NN[3])+" вектори: СКВ="+FloatToStr( SKV[3]));

      for (int k=1; k<=m; k++)         PARAM[0][k]=PARAM[method_][k];


        lich_sikle=lich_sikle+1;
       if (lich_sikle==lich_sikleK)  goto m9999g;

        goto m226g;
m9999g:

         method_lich=method_;
     vse=2;
if (CheckBox47->Checked==true) Form2->Memo1->Lines->Add(" Кінець обчислення");
       CALCULATION_Gausauto();
Memo1-> Lines->Add(" Кінець обчислення: СКВ="+FloatToStr(CKV));

// List1->SaveToFile("dataDDag.dat");
  delete PZ0, YYY, PZ;
  delete NSA, NRA;

for(int i=0; i<n; i++)
{
  delete[] DIDD[i];
  delete[] YYYY[i];
}
delete[] DIDD;
delete[] YYYY;

for(int i=0; i<m; i++)
{
  delete[] DIDDTRAN[i];
  delete[] DOB1[i];
  delete[] X[i];
  delete[] OBER[i];
  delete[] AA[i];
  delete[] DOB2[i];
  delete[] TTTT[i];
}
  delete[] DIDDTRAN;
  delete[] DOB1;
  delete[] X;
  delete[] OBER;
  delete[] AA;
  delete[] DOB2;
  delete[] TTTT;

for(int i=0; i<=3; i++)
{
  delete[] CHANGE[i];
}
delete[] CHANGE;

}


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void TForm1::Xi_mon()         // Монокристал-підкладка
{
//----------
//void __fastcall TForm1::GGG_KlClick(TObject *Sender)
if (RadioButton37->Checked==true)     // По Кладьку (тепл.ф.ДВ-з полікр.)
{
double d;
a=StrToFloat(Edit267->Text)*1e-8;      // (см)
VelKom=a*a*a;                                       // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiR0=-3.68946e-5;
  ChiI0=-3.595136e-6;
  ModChiI0=3.595136e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRH=10.94764e-6;
  ImChiRH=1e-12;
  ModChiRH=10.94764e-6;
  ReChiIH[1]=2.84908e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=2.84908e-6;
  ReChiIH[2]=1.79083e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=1.79083e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=-6.193591e-6;
  ImChiRH=1e-12;
  ModChiRH=6.193591e-6;
  ReChiIH[1]=-1.98152e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=1.98152e-6;
  ReChiIH[2]=-0.962504e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=0.962504e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=-9.43210e-6;
  ImChiRH=1e-12;
  ModChiRH=9.43210e-6;
  ReChiIH[1]=-2.50512e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=2.50512e-6;
  ReChiIH[2]=-2.38149e-8;
  ImChiIH=1e-12;
  ModChiIH[2]=2.38149e-8;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);
  };
  }

//----------
//void __fastcall TForm1::GGG_KleClick(TObject *Sender)
if (RadioButton38->Checked==true)     // По Кладьку (тепл.ф.ДВ-з монокр.)
{
double d;
a=StrToFloat(Edit267->Text)*1e-8;      // (см)
VelKom=a*a*a;                                       // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiR0=-3.68946e-5;
  ChiI0=-3.595136e-6;
  ModChiI0=3.595136e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRH=12.66065e-6;
  ImChiRH=1e-12;
  ModChiRH=12.66065e-6;
  ReChiIH[1]=3.26115e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=3.26115e-6;
  ReChiIH[2]=2.04984e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=2.04984e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=-11.16454e-6;
  ImChiRH=1e-12;
  ModChiRH=11.16454e-6;
  ReChiIH[1]=-3.41817e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=3.41817e-06;
  ReChiIH[2]=-1.66035e-6;
  ImChiIH=-1e-12;
  ModChiIH[2]=1.66035e-06;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=-13.55918e-6;
  ImChiRH=1e-12;
  ModChiRH=13.55918e-6;
  ReChiIH[1]=-3.46209e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=3.46209e-06;
  ReChiIH[2]=-3.29124e-8;
  ImChiIH=1e-12;
  ModChiIH[2]=3.29124e-08;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);

  };
  }

//-----------
//  void __fastcall TForm1::GGG_KrClick(TObject *Sender)
if (RadioButton36->Checked==true)   // по Кравцю
{
double d;
a=StrToFloat(Edit267->Text)*1e-8;      // (см)
VelKom=a*a*a;                          // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiR0=-38.6135e-6;
  ChiI0=-4.2002e-6;
  ModChiI0=4.2002e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRH=14.5491e-6;
  ImChiRH=1e-12;
  ModChiRH=14.5491e-6;
  ReChiIH[1]=3.8603e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=3.8603e-6;
  ReChiIH[2]=3.8603e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=3.8603e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=-13.8930e-6;
  ImChiRH=1e-12;
  ModChiRH=13.8930e-6;
  ReChiIH[1]=-4.1562e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=4.1562e-6;
  ReChiIH[2]=-4.1562e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=4.1562e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=-15.7453e-6;
  ImChiRH=1e-12;
  ReChiIH[1]=-4.1528e-6;
  ImChiIH=1e-12;
  ModChiIH[1]=4.1528e-6;
  ReChiIH[2]=-4.1528e-6;
  ImChiIH=1e-12;
  ModChiIH[2]=4.1528e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);
  };
  }

//----------
//void __fastcall TForm1::GGG_KsiClick(TObject *Sender)
if (RadioButton39->Checked==true)    // Ідеальний з ст. Кисловського (Металоф.)
{                                    // МіНТ 2005. Т.27, №2.  Табл.2...С.221
double d;
a=StrToFloat(Edit267->Text)*1e-8;      // (см)
VelKom=a*a*a;                          // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiR0=36.681e-6;
  ChiI0=-7.006e-6;
  ModChiI0=7.006e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRH=10.796e-6;                 // !!! знак не відомий
  ImChiRH=1e-12;
  ModChiRH=10.796e-6;
  ReChiIH[1]=3.171e-6;               // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[1]=3.171e-6;
  ReChiIH[2]=1.993e-6;               // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[2]=1.993e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=6.098e-6;                        // !!!знак не відомий
  ImChiRH=1e-12;
  ModChiRH=6.098e-6;
  ReChiIH[1]=1.943e-6;                     // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[1]=1.943e-6;
  ReChiIH[2]=0.944e-6;                    // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[2]=0.944e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=0;
  ImChiRH=0;
  ModChiRH=0;
  ReChiIH[1]=0;
  ImChiIH=0;
  ModChiIH[1]=0;
  ReChiIH[2]=0;
  ImChiIH=0;
  ModChiIH[2]=0;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);
  };
  }

//----------
//void __fastcall TForm1::GGG_KsniClick(TObject *Sender)
if (RadioButton40->Checked==true)    // Дефектний з ст. Кисловського (Металоф.)
{                                    // МіНТ 2005. Т.27, №2.  Табл.2...С.221
double d;
a=StrToFloat(Edit267->Text)*1e-8;      // (см)
VelKom=a*a*a;                          // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiR0=0;
  ChiI0=-7.006e-6;
  ModChiI0=7.006e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRH=12.717e-6;                 // !!!знак не відомий
  ImChiRH=1e-12;
  ModChiRH=12.717e-6;
  ReChiIH[1]=3.756e-6;               // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[1]=3.756e-6;
  ReChiIH[2]=2.361e-6;               // !!!знак не відомий
  ImChiIH=1e-12;
  ModChiIH[2]=2.361e-6;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=0;
  ImChiRH=0;
  ModChiRH=0;
  ReChiIH[1]=0;
  ImChiIH=0;
  ModChiIH[1]=0;
  ReChiIH[2]=0;
  ImChiIH=0;
  ModChiIH[2]=0;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=0;
  ImChiRH=0;
  ModChiRH=0;
  ReChiIH[1]=0;
  ImChiIH=0;
  ModChiIH[1]=0;
  ReChiIH[2]=0;
  ImChiIH=0;
  ModChiIH[2]=0;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);
  };
  }

//----------
//void __fastcall TForm1::YIGClick(TObject *Sender)
if (RadioButton41->Checked==true)    // YIG
{
double d;
a=StrToFloat(Edit267->Text)*1e-8;    // (см)
VelKom=a*a*a;                        // (см^3)
Nu=StrToFloat(Edit268->Text);
  ChiI0=0;
  ChiI0=-2.00993e-6;
  ModChiI0=2.00993e-6;
  Edit4->Text=FloatToStr(a/1e-8);
  Edit8->Text=FloatToStr(ChiR0);
  Edit9->Text=FloatToStr(ChiI0);
  Edit10->Text=FloatToStr(ModChiI0);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRH=7.79560e-6;
  ImChiRH=1e-12;
  ModChiRH=7.79560e-6;
  ReChiIH[1]=6.98492e-7;
  ImChiIH=1e-12;
  ModChiIH[1]=6.98492e-07;
  ReChiIH[2]=4.38754e-7;
  ImChiIH=1e-12;
  ModChiIH[2]=4.38754e-07;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox36->Caption="(444)";
  Edit20->Text=FloatToStr(ReChiRH);
  Edit21->Text=FloatToStr(ImChiRH);
  Edit22->Text=FloatToStr(ModChiRH);
  Edit23->Text=FloatToStr(ReChiIH[1]);
  Edit259->Text=FloatToStr(ImChiIH);
  Edit260->Text=FloatToStr(ModChiIH[1]);
  Edit261->Text=FloatToStr(ReChiIH[2]);
  Edit262->Text=FloatToStr(ImChiIH);
  Edit263->Text=FloatToStr(ModChiIH[2]);
  Edit34->Text=FloatToStr(d/1e-8);
  };

if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRH=-4.67983e-6;
  ImChiRH=1e-12;
  ModChiRH=4.67983e-6;
  ReChiIH[1]=-9.36375e-7;
  ImChiIH=1e-12;
  ModChiIH[1]=9.36375e-7;
  ReChiIH[2]=-4.56409e-7;
  ImChiIH=1e-12;
  ModChiIH[2]=4.56409e-07;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox37->Caption="(888)";
  Edit275->Text=FloatToStr(ReChiRH);
  Edit272->Text=FloatToStr(ImChiRH);
  Edit274->Text=FloatToStr(ModChiRH);
  Edit273->Text=FloatToStr(ReChiIH[1]);
  Edit276->Text=FloatToStr(ImChiIH);
  Edit277->Text=FloatToStr(ModChiIH[1]);
  Edit278->Text=FloatToStr(ReChiIH[2]);
  Edit279->Text=FloatToStr(ImChiIH);
  Edit280->Text=FloatToStr(ModChiIH[2]);
  Edit282->Text=FloatToStr(d/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRH=0;
  ImChiRH=0;
  ModChiRH=0;
  ReChiIH[1]=0;
  ImChiIH=0;
  ModChiIH[1]=0;
  ReChiIH[2]=0;
  ImChiIH=0;
  ModChiIH[2]=0;
  d=a/sqrt(h*h+k*k+l*l);

  GroupBox38->Caption="(880)";
  Edit293->Text=FloatToStr(ReChiRH);
  Edit290->Text=FloatToStr(ImChiRH);
  Edit292->Text=FloatToStr(ModChiRH);
  Edit291->Text=FloatToStr(ReChiIH[1]);
  Edit294->Text=FloatToStr(ImChiIH);
  Edit295->Text=FloatToStr(ModChiIH[1]);
  Edit296->Text=FloatToStr(ReChiIH[2]);
  Edit297->Text=FloatToStr(ImChiIH);
  Edit298->Text=FloatToStr(ModChiIH[2]);
  Edit300->Text=FloatToStr(d/1e-8);
  };
  }
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void TForm1::Xi_pl()
{
//-----------
//void __fastcall TForm1::YIG1Click(TObject *Sender)
if (RadioButton43->Checked==true)   // YIG по Кладьку (тепл.ф.ДВ-з полікр.)
{
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.30806e-6;
  ChiI0pl=-2.00993e-6;
  ModChiI0pl=2.00993e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=7.73059e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=7.73059e-6;
  ReChiIHpl[1]=6.98020e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=6.98020e-07;
  ReChiIHpl[2]=4.38458e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=4.38458e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-4.68756e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=4.68756e-6;
  ReChiIHpl[1]=-9.36492e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=9.36492e-7;
  ReChiIHpl[2]=-4.56466e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=4.56466e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-6.91966e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=6.91966e-6;
  ReChiIHpl[1]=-1.22506e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.22506e-6;
  ReChiIHpl[2]=-1.02730e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.02730e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
//void __fastcall TForm1::YIG1Click(TObject *Sender)
if (RadioButton46->Checked==true)   // YIG по Кладьку (тепл.ф.ДВ-з монокр.)
{
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.30806e-6;
  ChiI0pl=-2.00993e-6;
  ModChiI0pl=2.00993e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=8.89309e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.89309e-6;
  ReChiIHpl[1]=8.11552e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=8.11552e-07;
  ReChiIHpl[2]=5.09772e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=5.09772e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.19246e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.19246e-6;
  ReChiIHpl[1]=-1.69143e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.69143e-6;
  ReChiIHpl[2]=-8.24437e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=8.24437e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-9.73517e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.73517e-6;
  ReChiIHpl[1]=-1.77749e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.77749e-6;
  ReChiIHpl[2]=-1.49055e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.49055e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
if (RadioButton51->Checked==true)   // YIG-5-x по Кладьку (тепл.ф.ДВ-з монокр.)
{                                   // S-5-x   Y2.8La0.2Fe4.2Ga0.8O12  a=12.3715A
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.99287e-6;
  ChiI0pl=-1.964144e-6;
  ModChiI0pl=1.9694144e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=9.44271e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.44271e-6;
  ReChiIHpl[1]=7.71247e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=7.71247e-7;
  ReChiIHpl[2]=4.84246e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=4.84246e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.53496e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.53496e-6;
  ReChiIHpl[1]=-1.67009e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.67009e-6;
  ReChiIHpl[2]=-8.15846e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=8.15846e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-10.15999e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=10.15999e-6;
  ReChiIHpl[1]=-1.74861e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.74861e-6;
  ReChiIHpl[2]=-1.34017e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.34017e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
if (RadioButton52->Checked==true)   // YIG-5-x по Кладьку (тепл.ф.ДВ-з монокр.)
{                                   // S-4-x  Y2.8La0.2Fe4.545Ga0.455O12  а=12,380А
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.77933e-6;
  ChiI0pl=-2.034841e-6;
  ModChiI0pl=2.034841e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=9.28461e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.28461e-6;
  ReChiIHpl[1]=8.408216e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=8.40821e-07;
  ReChiIHpl[2]=5.28359e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=5.28359e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.43975e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.43975e-6;
  ReChiIHpl[1]=-1.72833e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.72833e-6;
  ReChiIHpl[2]=-8.40766e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=8.40766e-7;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-10.03652e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=10.03652e-6;
  ReChiIHpl[1]=-1.81077e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.81077e-6;
  ReChiIHpl[2]=-1.63447e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.63447e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
if (RadioButton53->Checked==true)   // YIG-1-x по Кладьку (тепл.ф.ДВ-з монокр.)
{                                   // S-1-x  Y2.8La0.2Fe4.545Ga0.455O12  а=12,3747А
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.50222e-6;
  ChiI0pl=-2.04727e-6;
  ModChiI0pl=2.04727e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=9.03512e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.03512e-6;
  ReChiIHpl[1]=8.48829e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=8.48829e-07;
  ReChiIHpl[2]=5.33121e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=5.33121e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.28461e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.28461e-6;
  ReChiIHpl[1]=-1.72964e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.72964e-6;
  ReChiIHpl[2]=-8.43606e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=8.43606e-7;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-9.845636e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.845636e-6;
  ReChiIHpl[1]=-1.81547e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.81547e-6;
  ReChiIHpl[2]=-1.48457e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.48457e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
if (RadioButton54->Checked==true)   // YIG-2-x по Кладьку (тепл.ф.ДВ-з монокр.)
{                                   // S-2-x  Y2.8La0.2Fe4.545Ga0.455O12  а=12,377А
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-30.67820e-6;
  ChiI0pl=-1.987196e-6;
  ModChiI0pl=1.987196e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=9.20348e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.20348e-6;
  ReChiIHpl[1]=7.92873e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=7.92873e-07;
  ReChiIHpl[2]=4.98087e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=4.98087e-07;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.38568e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.38568e-6;
  ReChiIHpl[1]=-1.68261e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.68261e-6;
  ReChiIHpl[2]=-8.19734e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=8.19734e-7;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-9.974140e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.974140e-6;
  ReChiIHpl[1]=-1.76439e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.76439e-6;
  ReChiIHpl[2]=-1.50784e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.50784e-8;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit350->Text=FloatToStr(ReChiRHpl);
  Edit347->Text=FloatToStr(ImChiRHpl);
  Edit349->Text=FloatToStr(ModChiRHpl);
  Edit348->Text=FloatToStr(ReChiIHpl[1]);
  Edit351->Text=FloatToStr(ImChiIHpl);
  Edit352->Text=FloatToStr(ModChiIHpl[1]);
  Edit353->Text=FloatToStr(ReChiIHpl[2]);
  Edit354->Text=FloatToStr(ImChiIHpl);
  Edit355->Text=FloatToStr(ModChiIHpl[2]);
  Edit357->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
//void __fastcall TForm1::YIG1Click(TObject *Sender)
if (RadioButton47->Checked==true)   // YIG по Кравцю
{
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;        // (см)
VelKompl=apl*apl*apl;                      // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=-29.9207e-6;
  ChiI0pl=-2.18320e-6;
  ModChiI0pl=2.18320e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
  ReChiRHpl=8.82260e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.82260e-6;
  ReChiIHpl[1]=9.03900e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=9.03900e-7;
  ReChiIHpl[2]=9.03900e-7;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=9.03900e-7;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-8.96750e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=8.96750e-6;
  ReChiIHpl[1]=-2.14010e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=2.14010e-6;
  ReChiIHpl[2]=-2.14010e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=2.14010e-6;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);

};
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=-10.1584e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=10.1584e-6;
  ReChiIHpl[1]=-2.13550e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=2.13550e-6;
  ReChiIHpl[2]=-2.13550e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=2.13550e-6;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);
 };
}

//-----------
//void __fastcall TForm1::YIGKs1Click(TObject *Sender)
if (RadioButton44->Checked==true)   // YIG по Кисл.
{
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;   // (см)
VelKompl=apl*apl*apl;                 // (см^3)
Nu=StrToFloat(Edit269->Text);
  ChiR0pl=25.991e-6;                      // !!! знак не відомий
  ChiI0pl=-5.448e-6;
  ModChiI0pl=5.448e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRHpl=4.710e-6;                    // !!! знак не відомий
  ImChiRHpl=1e-12;
  ModChiRHpl=4.710e-6;               
  ReChiIHpl[1]=1.424e-6;                 // !!! знак не відомий
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.424e-06;          
  ReChiIHpl[2]=0.894e-6;                  // !!! знак не відомий
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=0.894e-06;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };
if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-1.649e-6;                  // !!! знак не відомий
  ImChiRHpl=1e-12;
  ModChiRHpl=1.649e-6;
  ReChiIHpl[1]=-0.468e-6;               // !!! знак не відомий
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=0.468e-6;
  ReChiIHpl[2]=-0.228e-6;               // !!! знак не відомий
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=0.228e-06;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);
  };
  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
{
  ReChiRHpl=0;
  ImChiRHpl=0;
  ModChiRHpl=0;
  ReChiIHpl[1]=0;
  ImChiIHpl=0;
  ModChiIHpl[1]=0;
  ReChiIHpl[2]=0;
  ImChiIHpl=0;
  ModChiIHpl[2]=0;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);
  };
  }

//-----------
//void __fastcall TForm1::GGG1Click(TObject *Sender)  //Плівка
if (RadioButton45->Checked==true)   // GGG по Кладьку.
{
double dpl;
apl=StrToFloat(Edit167->Text)*1e-8;     // (см)
VelKompl=apl*apl*apl;                   // (см^3)
Nu=StrToFloat(Edit269->Text);
ChiR0pl=0;
ChiI0pl=-3.595136e-6;
ModChiI0pl=3.595136e-6;
  Edit320->Text=FloatToStr(apl/1e-8);
  Edit309->Text=FloatToStr(ChiR0pl);
  Edit308->Text=FloatToStr(ChiI0pl);
  Edit310->Text=FloatToStr(ModChiI0pl);
  if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
  {
  ReChiRHpl=10.94764e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=10.94764e-6;
  ReChiIHpl[1]=2.84908e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=2.84908e-06;
  ReChiIHpl[2]=1.79083e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=1.79083e-06;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox40->Caption="(444)";
  Edit314->Text=FloatToStr(ReChiRHpl);
  Edit311->Text=FloatToStr(ImChiRHpl);
  Edit313->Text=FloatToStr(ModChiRHpl);
  Edit312->Text=FloatToStr(ReChiIHpl[1]);
  Edit315->Text=FloatToStr(ImChiIHpl);
  Edit316->Text=FloatToStr(ModChiIHpl[1]);
  Edit317->Text=FloatToStr(ReChiIHpl[2]);
  Edit318->Text=FloatToStr(ImChiIHpl);
  Edit319->Text=FloatToStr(ModChiIHpl[2]);
  Edit321->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==8)
  {
  ReChiRHpl=-6.193591e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=6.193591e-6;
  ReChiIHpl[1]=-1.98152e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[1]=1.98152e-06;
  ReChiIHpl[2]=-0.962504e-6;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=0.962504e-06;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox41->Caption="(888)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);
  };

  if (Lambda==1.5405*1e-8) if (h==8) if (k==8) if (l==0)
  {
  ReChiRHpl=-9.43210e-6;
  ImChiRHpl=1e-12;
  ModChiRHpl=9.43210e-6;
  ReChiIHpl[1]=-2.50512e-6;
  ImChiIHpl=1e-12;
  ReChiIHpl[2]=-2.38149e-8;
  ImChiIHpl=1e-12;
  ModChiIHpl[2]=2.38149e-08;
  dpl=apl/sqrt(h*h+k*k+l*l);

  GroupBox42->Caption="(880)";
  Edit332->Text=FloatToStr(ReChiRHpl);
  Edit329->Text=FloatToStr(ImChiRHpl);
  Edit331->Text=FloatToStr(ModChiRHpl);
  Edit330->Text=FloatToStr(ReChiIHpl[1]);
  Edit333->Text=FloatToStr(ImChiIHpl);
  Edit334->Text=FloatToStr(ModChiIHpl[1]);
  Edit335->Text=FloatToStr(ReChiIHpl[2]);
  Edit336->Text=FloatToStr(ImChiIHpl);
  Edit337->Text=FloatToStr(ModChiIHpl[2]);
  Edit339->Text=FloatToStr(dpl/1e-8);
  };
  }
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void TForm1::Xi_Si()         // Монокристал-підкладка
{}
/*
a=5.43*1e-8;
VelKom=2e-23;
Nu=0.215;
if (Lambda==1.5405*1e-8) if (h==1) if (k==1) if (l==1)
{
ReChiRH=-5.484e-6;
ImChiRH=5.484e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-1.724e-6;
ImChiIH=1.724e-6;
ModChiIH=2.438e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-1.51e-7;
ImChiIH=1.51e-7;
ModChiIH=2.1338e-7;
}
ModChiRH=7.755e-6;
ModChiI0=3.496e-7;
Mu0=144;
};
/////////////////
if (Lambda==0.70926*1e-8) if (h==1) if (k==1) if (l==1)
{
ReChiRH=-1.461e-6;
ImChiRH=1.461e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-7.92e-9;
ImChiIH=7.92e-9;
ModChiIH=1.12e-8;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-7.699e-9;
ImChiIH=7.699e-9;
ModChiIH=1.089e-8;
}
ModChiRH=1.621e-6;
ModChiI0=1.604e-8;
Mu0=14.64;
};
////////////////////////
if (Lambda==1.936*1e-8) if (h==1) if (k==1) if (l==1)
{
ReChiRH=-8.716e-6;
ImChiRH=8.716e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-4.202e-7;
ImChiIH=4.202e-7;
ModChiIH=5.942e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-3.38e-7;
ImChiIH=3.38e-7;
ModChiIH=4.78e-7;
}
ModChiRH=1.233e-5;
ModChiI0=8.5235e-7;
Mu0=276;
}
///////////////////
if (Lambda==0.559*1e-8) if (h==1) if (k==1) if (l==1)
{
ReChiRH=-7.11e-7;
ImChiRH=7.11e-7;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-3.045e-9;
ImChiIH=3.045e-9;
ModChiIH=4.306e-9;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-2.99e-9;
ImChiIH=2.99e-9;
ModChiIH=4.23e-9;
}
ModChiRH=1.005e-6;
ModChiI0=6.165e-9;
Mu0=7.39;
}
////////////////////
if (Lambda==1.5405*1e-8) if (h==3) if (k==3) if (l==3)
{
ReChiRH=-3.149e-6;
ImChiRH=-3.149e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-1.543e-7;
ImChiIH=-1.543e-7;
ModChiIH=2.183e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=1.635e-8;
ImChiIH=1.635e-8;
ModChiIH=2.312e-8;
}
ModChiRH=4.453e-6;
ModChiI0=3.496e-7;
Mu0=144;
};
/////////////////////
if (Lambda==0.70926*1e-8) if (h==3) if (k==3) if (l==3)
{
ReChiRH=-6.524e-7;
ImChiRH=-6.524e-7;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-7.157e-9;
ImChiIH=-7.157e-9;
ModChiIH=1.012e-8;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-5.372e-9;
ImChiIH=-5.372e-9;
ModChiIH=7.597e-9;
}
ModChiRH=9.226e-7;
ModChiI0=1.604e-8;
Mu0=14.64;
};
///////////////////
if (Lambda==1.936*1e-8) if (h==3) if (k==3) if (l==3)
{
ReChiRH=-5.027e-6;
ImChiRH=-5.027e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-3.75e-7;
ImChiIH=-3.75e-7;
ModChiIH=5.303e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=2.712e-7;
ImChiIH=2.712e-7;
ModChiIH=3.835e-7;
}
ModChiRH=7.11e-6;
ModChiI0=8.5245e-7;
Mu0=276;
};
//////////////////////////
if (Lambda==0.559*1e-8) if (h==3) if (k==3) if (l==3)
{
ReChiRH=-4.038e-7;
ImChiRH=-4.038e-7;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-2.76e-9;
ImChiIH=-2.76e-9;
ModChiIH=3.89e-9;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-2.32e-9;
ImChiIH=-2.32e-9;
ModChiIH=3.278e-9;
}
ModChiRH=5.711e-7;
ModChiI0=6.167e-9;
Mu0=7.39;
};
///////////////////////
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==4)
{
ReChiRH=-4.42e-6;
ImChiRH=4.879e-21;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-2.802e-7;
ImChiIH=3.088e-22;
ModChiIH=2.802e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=2.617e-7;
ImChiIH=-2.88e-22;
ModChiIH=2.618e-7;
}
ModChiRH=4.427e-6;
ModChiI0=3.496e-7;
Mu0=144;
};
//////////////
if (Lambda==0.70926*1e-8) if (h==4) if (k==4) if (l==4)
{
ReChiRH=-9.099e-7;
ImChiRH=1.002e-21;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-1.309e-8;
ImChiIH=1.44e-23;
ModChiIH=1.309e-8;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-7.33e-9;
ImChiIH=8.08e-24;
ModChiIH=7.33e-9;
}
ModChiRH=9.09e-7;
ModChiI0=1.604e-8;
Mu0=14.64;
};
///////
if (Lambda==0.559*1e-8) if (h==4) if (k==4) if (l==4)
{
ReChiRH=-5.623e-7;
ImChiRH=6.198e-22;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-5.05e-9;
ImCodChiRH=9.193e-6;
ModChiI0=3.496e-7;
Mu0=144;
};
if (Lambda==0.70926*1e-8) if (h==2) if (k==2) if (l==0)
{
ReChiRH=-1.917e-6;
ImChiRH=7.04e-22;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-1.5501e-8;
ImChiIH=5.6979e-24;
ModChiIH=1.55e-8;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-1.435e-8;
ImChiIH=5.274e-24;
ModChiIH=1.4356e-8;
}
ModChiRH=1.917e-6;
ModChiI0=1.604e-8;
Mu0=14.64;
};
///////
if (Lambda==1.936*1e-8) if (h==2) if (k==2) if (l==0)
{
ReChiRH=-1.463e-5;
ImChiRH=5.375e-21;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-8.207e-7;
ImChiIH=3.015e-22;
ModChiIH=8.207e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-3.945e-7;
ImChiIH=1.449e-22;
ModChiIH=3.945e-7;
}
ModChiRH=1.46e-5;
ModChiI0=8.5235e-7;
Mu0=276;
};
////////////
if (Lambda==0.559*1e-8) if (h==2) if (k==2) if (l==0)
{
ReChiRH=-1.1879e-6;
ImChiRH=4.364e-22;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-5.965e-9;
ImChiIH=2.191e-24;
ModChiIH=5.965e-9;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-5.682e-9;
ImChiIH=2.0875e-24;
ModChiIH=5.682e-9;
}
ModChiRH=1.188e-6;
ModChiI0=6.16e-9;
Mu0=7.39;
};
////////////
if (Lambda==1.5405*1e-8) if (h==4) if (k==4) if (l==0)
{
ReChiRH=-5.770e-6;
ImChiRH=4.239e-6;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-3.0169e-6;
ImChiIH=2.216e-22;
ModChiIH=3.0169e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=9.22e-8;
ImChiIH=-6.777e-23;
ModChiIH=9.224e-8;
}
ModChiRH=5.77e-6;
ModChiI0=3.496e-7;
Mu0=144;
};
////////////
if (Lambda==0.70926*1e-8) if (h==4) if (k==4) if (l==0)
{
ReChiRH=-1.1932e-6;
ImChiRH=8.764e-22;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-1.401e-8;
ImChiIH=1.0297e-23;
ModChiIH=1.401e-8;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-9.878e-9;
ImChiIH=7.258e-24;
ModChiIH=9.878e-9;
}
ModChiRH=1.193e-6;
ModChiI0=1.604e-8;
Mu0=14.64;
};
if (Lambda==0.559*1e-8) if (h==4) if (k==4) if (l==0)
{
ReChiRH=-7.384e-7;
ImChiRH=5.925e-22;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-5.400e-9;
ImChiIH=3.968e-24;
ModChiIH=5.400e-9;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-4.383e-9;
ImChiIH=3.22e-24;
ModChiIH=4.383e-9;
}
ModChiRH=7.384e-7;
ModChiI0=6.16e-9;
Mu0=7.39;
};
*/
/*{
}hiIH=5.57e-24;
ModChiIH=5.05e-9;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-3.63e-9;
ImChiIH=4.003e-24;
ModChiIH=3.632e-9;
}
ModChiRH=5.62e-7;
ModChiI0=6.166e-9;
Mu0=7.39;
};
///////////
if (Lambda==1.5405*1e-8) if (h==2) if (k==2) if (l==0)
{
ReChiRH=-9.193e-6;
ImChiRH=3.38e-21;
if (RadioButton1->Checked==true)  ////Sigma polar
{
ReChiIH=-3.369e-7;
ImChiIH=1.238e-22;
ModChiIH=3.369e-7;
}
if (RadioButton2->Checked==true)  ////Pi polar
{
ReChiIH=-2.249e-7;
ImChiIH=8.265e-23;
ModChiIH=2.249e-7;
}
              */

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void __fastcall TForm1::N9Click(TObject *Sender) //Відкр. експер. КДВ
{
 int nskv,kskv;         // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double TetaMin,DeltaTeta1,ik;  /*intIk[MM],*/   //!!!!!!!!!!!!!!!!!!!!!!!!!
  double *intIk;
  intIk  = new double[MM];

if (number_KDV==0)
{
MessageBox(0,"Ну і куди так спішиш?","???!", MB_OK + MB_ICONEXCLAMATION);
number_KDV=1;
//if (number_KDV==0) goto mexp;
}

AnsiString MyFName="";
if (OtkEt->Execute())
{
MyFName=OtkEt->FileName;
TStringList *List = new TStringList;
AnsiString Ds11;
//AnsiString Mas[MM], Mas1[MM];
//AnsiString Mas2[10000];
double  PEmax; // X[MM],
int m1e,m10e;
//double Y[10000];
  double *X;
  X  = new double[MM];
AnsiString *Mas, *Mas1;
  Mas  = new AnsiString[MM];
  Mas1 = new AnsiString[MM];


List->LoadFromFile(OtkEt->FileName);//зчитується файл NameFile

double       p11,p12,p13 ;
AnsiString   p1,p2,p3 ;
p1=List->Strings[0];// зчитуємо у масив рядки файла
p2=List->Strings[1];// зчитуємо у масив рядки файла
p3=List->Strings[2];// зчитуємо у масив рядки файла
p11=atof(p1.c_str());
p12=atof(p2.c_str());
p13=atof(p3.c_str());
m1e=p11-1;
ik=p12;
m10e=p13;
vved_exper=1;    //що дані вводяться при відкритті експ. КДВ
nskv=-m10e*ik;                  //100;
kskv=(m1e-m10e)*ik;  //(m1e-m10e)*ik-20.;      //20 angl sek.
if (RadioButton10->Checked==true)
{
Edit69->Text=FloatToStr(nskv);
Edit70->Text=FloatToStr(kskv);
}
if (RadioButton20->Checked==true)
{
Edit130->Text=FloatToStr(nskv);
Edit129->Text=FloatToStr(kskv);
}
if (RadioButton21->Checked==true)
{
Edit89->Text=FloatToStr(nskv);
Edit134->Text=FloatToStr(kskv);
}

for (int i=3;i<=m1e+3;i++)
{
Mas[i]=List->Strings[i];// зчитуємо у масив рядки файла
Mas1[i]="";
for (int k=1; k<=(Mas[i].Length());k++)
{
Ds11=Mas[i][k]; //допоміжна змінна типу AnsiString
if (Ds11!=("\t"))
if (Ds11!=(" "))
Mas1[i]=Mas1[i]+Ds11; //у масив Mas1 заносяться значення першого стовпця
else break;
else break;
}
X[i]=atof(Mas1[i].c_str());//перший стовбець переводиться із тексту в числа
}
for (int i=3;i<=m1e+3;i++) intIk[i-3]=X[i];
for (int i=0;i<=m1e;i++) if (intIk[i]<1e-14) intIk[i]=1e-14;

if (RadioButton10->Checked==true)
{
for (int i=0;i<=m1e;i++) intI02d[i][1]=intIk[i];
ekspk0=StrToFloat(Edit165->Text);
ekspk=StrToFloat(LabeledEdit1->Text);
m1_[1]=m1e;
m10_[1]=m10e;
ik_[1]=ik;
Edit202->Text=FloatToStr(ik_[1]);
Edit235->Text=FloatToStr(m1_[1]);
Edit238->Text=FloatToStr(m10_[1]);
}
if (RadioButton20->Checked==true)
{
for (int i=0;i<=m1e;i++) intI02d[i][2]=intIk[i];
ekspk0=StrToFloat(Edit75->Text);
ekspk=StrToFloat(LabeledEdit2->Text);
m1_[2]=m1e;
m10_[2]=m10e;
ik_[2]=ik;
Edit203->Text=FloatToStr(ik_[2]);
Edit236->Text=FloatToStr(m1_[2]);
Edit239->Text=FloatToStr(m10_[2]);
}
if (RadioButton21->Checked==true)
{
for (int i=0;i<=m1e;i++) intI02d[i][3]=intIk[i];
ekspk0=StrToFloat(Edit135->Text);
ekspk=StrToFloat(LabeledEdit3->Text);
m1_[3]=m1e;
m10_[3]=m10e;
ik_[3]=ik;
Edit204->Text=FloatToStr(ik_[3]);
Edit237->Text=FloatToStr(m1_[3]);
Edit240->Text=FloatToStr(m10_[3]);
}

for (int i=0; i<=m1e; i++)
{
 intIk[i]=intIk[i]-ekspk0;
if (intIk[i]<=0) intIk[i]=0.001*(ekspk0+1);
}
  PEmax=0;
for (int i=0; i<=m1e; i++) if (intIk[i]>PEmax) PEmax=intIk[i];
for (int i=0; i<=m1e; i++) intIk[i]=intIk[i]/PEmax*ekspk;

if (RadioButton10->Checked==true) for (int i=0; i<=m1e; i++) intIk2d[i][1]=intIk[i];
if (RadioButton20->Checked==true) for (int i=0; i<=m1e; i++) intIk2d[i][2]=intIk[i];
if (RadioButton21->Checked==true) for (int i=0; i<=m1e; i++) intIk2d[i][3]=intIk[i];

TetaMin=-(m10e)*ik;
for (int i=0;i<=m1e;i++)
{
DeltaTeta1=(TetaMin+i*ik);
//Application->ProcessMessages();
if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
if (number_KDV==2)
{
if (RadioButton10->Checked==true) Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
if (RadioButton20->Checked==true) Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
//if (RadioButton10->Checked==true) Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
//if (RadioButton20->Checked==true) Series25->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
if (number_KDV==3)
{
if (RadioButton10->Checked==true) Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
if (RadioButton20->Checked==true) Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
if (RadioButton21->Checked==true) Series45->AddXY(DeltaTeta1,intIk[i],"",clGreen);
//if (RadioButton10->Checked==true) Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
//if (RadioButton20->Checked==true) Series25->AddXY(DeltaTeta1,intIk[i],"",clGreen);
//if (RadioButton21->Checked==true) Series26->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}

}
delete X, Mas, Mas1;
}
//mexp:
delete intIk;
}

//---------------------------------------------------------------------------

void __fastcall TForm1::N8Click(TObject *Sender) //Зберегти розрахункову КДВ
{
//AnsiString GDDparam[20],param[20];
double w,TetaMin,z1;
double  L_shod;    // Z_shod[2*KM+1], D_shod[2*KM+1],
int number_KDV_;
//char ss[100];
AnsiString *GDDparam, *param;
  GDDparam  = new AnsiString[20];
  param = new AnsiString[20];
double *Z_shod, *D_shod;
  Z_shod  = new double[2*KM+1];
  D_shod  = new double[2*KM+1];
char *ss;
  ss  = new char[1000];

AnsiString MyFName3="";
if (SaveKDB1->Execute())
{
MyFName3=SaveKDB1->FileName;
TStringList *List3 = new TStringList;

if (CheckBox68->Checked==true)
  {
  List3->Add(" Монокристал / підкладка");
  if (RadioButton37->Checked==true) List3->Add("Хі - GGG Кл. (полі.)");
  if (RadioButton38->Checked==true) List3->Add("Хі - GGG Кл. (моно.)");
  if (RadioButton36->Checked==true) List3->Add("Хі - GGG Кр. (полі.)");
  if (RadioButton39->Checked==true) List3->Add("Хі - GGG Кc.ід. (полі.)");
  if (RadioButton40->Checked==true) List3->Add("Хі - GGG Кр.не ід. (полі.)");
  AnsiString nameapd1="  а (A) = ";
  List3->Add(nameapd1+'\t'+FloatToStr(a/1e-8));
  }
if (CheckBox31->Checked==true)
  {
  List3->Add(" Гетероструктура");
  if (RadioButton43->Checked==true) List3->Add("Хі pl - YIG Кл. (полі.)");
  if (RadioButton46->Checked==true) List3->Add("Хі pl - YIG Кл. (моно.)");
  if (RadioButton47->Checked==true) List3->Add("Хі pl - YIG Кр. (полі.)");
  if (RadioButton44->Checked==true) List3->Add("Хі pl - YIG Кc.ід. (полі.)");
  if (RadioButton45->Checked==true) List3->Add("Хі pl - GGG Кл. (полі.)");
  if (RadioButton53->Checked==true) List3->Add("Хі pl - S-1-x (моно.)");
  if (RadioButton54->Checked==true) List3->Add("Хі pl - S-2-x (моно.)");
  if (RadioButton57->Checked==true) List3->Add("Хі pl - S-3-x (моно.)");
  if (RadioButton52->Checked==true) List3->Add("Хі pl - S-4-x (моно.)");
  if (RadioButton51->Checked==true) List3->Add("Хі pl - S-5-x (моно.)");
  AnsiString nameapd1pl="  аpl (A) = ";
  AnsiString nameapd1pl2="  hpl (мкм) = ";
  List3->Add(nameapd1pl+'\t'+FloatToStr(apl/1e-8)+'\t'+nameapd1pl2+'\t'+FloatToStr(hpl/1e-4));
  }
if (CheckBox67->Checked==true) List3->Add(" Приповерхневий порушений шар");
//sprintf(ss,"h=% i,\tk=% i,\tl=% i", h,k,l);
//List3->Add(ss);
//sprintf(ss,"h= %i;\tk= \"%i\"; \t  l = \"%i\"; \n", h,3,0);
//sprintf(sss," h = \"%i\"; \t  k = \"%i\"; \t  l = \"%i\"; \n", h,k,l);
//List3->Add(ss);
//        sprintf(ss,"%3.6lf\t%.0lf", DD[k],Dl[k]/1e-8);
//fprintf(debug, "  \n");
//sprintf(S,"Сотрудник %s, %i г.p.",.Editl->Text, CSpinEditl->Value) ;
if (RadioButton5->Checked==true) List3->Add("Omega сканування");
if (RadioButton6->Checked==true) List3->Add("Teta-2teta сканування (тільки когер.)");
if (RadioButton1->Checked==true)  List3->Add("Sigma поляризація");
if (RadioButton55->Checked==true) List3->Add("Pi поляризація");
if (RadioButton2->Checked==true)  List3->Add("Sigma + Pi поляризація  Монохроматор ГГГ(444)");
if (RadioButton56->Checked==true) List3->Add("Sigma + Pi поляризація  Монохроматор Si(111)/Ge(111)");
if (RadioButton11->Checked==true) List3->Add("Моделювання ГБП за р-нями ТТ");
if (RadioButton13->Checked==true) List3->Add("Моделювання ГБП за р-нями УДТ");
if (RadioButton12->Checked==true) List3->Add("Моделювання ГБП за р-нями ТТ+кінем.ПШ");
if (RadioButton16->Checked==true) List3->Add("Моделювання ГБП за р-нями ТТ+кінем.ПШ+розворот");
if (RadioButton33->Checked==true) List3->Add("Моделювання ГБП за р-нями УДТ+кінем.ПШ(Мол.)");

if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==false) number_KDV_=1;
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==false) number_KDV_=1;
if (CheckBox42->Checked==false && CheckBox43->Checked==false && CheckBox44->Checked==true) number_KDV_=1;
if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==false) number_KDV_=2;
if (CheckBox42->Checked==true && CheckBox43->Checked==false && CheckBox44->Checked==true) number_KDV_=2;
if (CheckBox42->Checked==false && CheckBox43->Checked==true && CheckBox44->Checked==true) number_KDV_=2;
if (CheckBox42->Checked==true && CheckBox43->Checked==true && CheckBox44->Checked==true) number_KDV_=3;
sprintf(ss,"Кількість КДВ: \t% i",number_KDV_);
List3->Add(ss);

if (CheckBox42->Checked==true)  // (444)
{
List3->Add(" КДВ (444)");
if (CheckBox59->Checked==false)  List3->Add("Апаратна функція з файла AF-*");
if (CheckBox59->Checked==true)   // Параметри апар. ф-ї
{
List3->Add(" Парам. апаратної функції: ");
List3->Add(" Ymin  \t   A   \t   w  \t  Xmin  \t  К-ть точок: ");
List3->Add(FloatToStr(param_AF[0][1])+'\t'+FloatToStr(param_AF[1][1])+'\t'+FloatToStr(param_AF[2][1])+'\t'+FloatToStr(param_AF[3][1])+'\t'+FloatToStr(param_AF[4][1])+'\t'+FloatToStr(param_AF[5][1]));
}
AnsiString namekoef=" Коеф. на експер. КДВ: ";
List3->Add(namekoef+'\t'+LabeledEdit1->Text+'\t'+Edit165->Text);
}
if (CheckBox43->Checked==true)  // (888)
{
List3->Add(" КДВ (888)");
if (CheckBox60->Checked==false)  List3->Add("Апаратна функція з файла AF-*");
if (CheckBox60->Checked==true)   // Параметри апар. ф-ї
{
List3->Add(" Парам. апаратної функції: ");
List3->Add(" Ymin  \t   A   \t   w  \t  Xmin  \t  К-ть точок: ");
List3->Add(FloatToStr(param_AF[0][2])+'\t'+FloatToStr(param_AF[1][2])+'\t'+FloatToStr(param_AF[2][2])+'\t'+FloatToStr(param_AF[3][2])+'\t'+FloatToStr(param_AF[4][2])+'\t'+FloatToStr(param_AF[5][2]));
}
AnsiString namekoef=" Коеф. на експер. КДВ: ";
List3->Add(namekoef+'\t'+LabeledEdit2->Text+'\t'+Edit75->Text);
}
if (CheckBox44->Checked==true)  // (880)
{
List3->Add(" КДВ (880)");
if (CheckBox71->Checked==false)  List3->Add("Апаратна функція з файла AF-*");
if (CheckBox71->Checked==true)   // Параметри апар. ф-ї
{
List3->Add(" Парам. апаратної функції: ");
List3->Add(" Ymin  \t   A   \t   w  \t  Xmin  \t  К-ть точок: ");
List3->Add(FloatToStr(param_AF[0][3])+'\t'+FloatToStr(param_AF[1][3])+'\t'+FloatToStr(param_AF[2][3])+'\t'+FloatToStr(param_AF[3][3])+'\t'+FloatToStr(param_AF[4][3])+'\t'+FloatToStr(param_AF[5][3]));
}
AnsiString namekoef=" Коеф. на експер. КДВ: ";
List3->Add(namekoef+'\t'+LabeledEdit3->Text+'\t'+Edit135->Text);
}

//if (RadioButton3->Checked==true || RadioButton4->Checked==true) // ід.част. чи ід.част.+ ПШ
//{
List3->Add(" ");
if (RadioButton11->Checked==true || RadioButton12->Checked==true || RadioButton16->Checked==true) List3->Add("Когер. складова КДВ від ід.част.монокристалу розрах. за р-нями Такагі-Топена");
if (RadioButton13->Checked==true) List3->Add("Когер. складова КДВ від ід.част.монокристалу розрах. за узагальн. динам. теорією");
if (CheckBox13->Checked==false && CheckBox17->Checked==false && CheckBox12->Checked==false && CheckBox15->Checked==false && CheckBox16->Checked==false && CheckBox14->Checked==false)
List3->Add("Дефекти у ідеальній частині монокристалу не враховуються");
else
{
 List3->Add("Дефекти у ідеальній частині монокристалу:");
if (CheckBox13->Checked==true || CheckBox17->Checked==true)
   {
   if (RadioButton7->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/sqrt(2)");
   if (RadioButton8->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a");
   if (RadioButton9->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a*sqrt(2)");
   if (RadioButton28->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)");
   if (RadioButton32->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)");
   if (RadioButton29->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a*sqrt(3)");
AnsiString namekoef=" Коефіцієнт на L: ";
List3->Add(namekoef+'\t'+Edit149->Text);
   }
if (CheckBox13->Checked==true || CheckBox17->Checked==true) List3->Add("Конц.петель: \t Радіус петель:");
if (CheckBox13->Checked==true) List3->Add (Edit53->Text+'\t'+Edit54->Text);
if (CheckBox17->Checked==true) List3->Add (Edit64->Text+'\t'+Edit65->Text);
if (CheckBox12->Checked==true || CheckBox15->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - сферичні кластери");
if (CheckBox12->Checked==true || CheckBox15->Checked==true) List3->Add("Конц.сф.кластерів: \t Радіус сф.кластерів:");
if (CheckBox12->Checked==true) List3->Add (Edit50->Text+'\t'+Edit51->Text);
if (CheckBox15->Checked==true) List3->Add (Edit46->Text+'\t'+Edit47->Text);
if (CheckBox16->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - дископодібні кластери");
if (CheckBox16->Checked==true) List3->Add("Конц.диск.кластерів: \t Радіус диск.кластерів:");
if (CheckBox16->Checked==true) List3->Add (Edit55->Text+'\t'+Edit56->Text);
if (CheckBox14->Checked==true) List3->Add("Модель дефектів в ід. част. монокр. - сферичні кластери (точкові дефекти)");
if (CheckBox14->Checked==true) List3->Add("Конц.сф.кластерів (т.д.): \t Радіус сф.кластерів (т.д.):");
if (CheckBox14->Checked==true) List3->Add (Edit61->Text+'\t'+Edit62->Text);
}
List3->Add(" ");

if (CheckBox31->Checked==true)             // Плівка
{
if (RadioButton11->Checked==true || RadioButton13->Checked==true || RadioButton12->Checked==true || RadioButton16->Checked==true)
List3->Add("Когер. складова КДВ від ід.част.плівки розрах. за р-нями Такагі-Топена");
if (CheckBox34->Checked==false && CheckBox35->Checked==false && CheckBox32->Checked==false && CheckBox33->Checked==false && CheckBox37->Checked==false)
List3->Add("Дефекти у ідеальній частині плівки не враховуються");
else
{
 List3->Add("Дефекти у ідеальній частині плівки:");
if (CheckBox34->Checked==true || CheckBox35->Checked==true)
   {
   if (RadioButton7->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/sqrt(2)");
   if (RadioButton8->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a");
   if (RadioButton9->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a*sqrt(2)");
   if (RadioButton28->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)");
   if (RadioButton32->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)");
   if (RadioButton29->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a*sqrt(3)");
AnsiString namekoef=" Коефіцієнт на L: ";
List3->Add(namekoef+'\t'+Edit149->Text);
   }
if (CheckBox34->Checked==true || CheckBox35->Checked==true) List3->Add("Конц.петель: \t Радіус петель:");
if (CheckBox34->Checked==true) List3->Add (Edit174->Text+'\t'+Edit175->Text);
if (CheckBox35->Checked==true) List3->Add (Edit176->Text+'\t'+Edit177->Text);
if (CheckBox32->Checked==true || CheckBox33->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - сферичні кластери");
if (CheckBox32->Checked==true || CheckBox33->Checked==true) List3->Add("Конц.сф.кластерів: \t Радіус сф.кластерів:");
if (CheckBox32->Checked==true) List3->Add (Edit168->Text+'\t'+Edit169->Text);
if (CheckBox33->Checked==true) List3->Add (Edit171->Text+'\t'+Edit172->Text);
if (CheckBox36->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - дископодібні кластери");
if (CheckBox36->Checked==true) List3->Add("Конц.диск.кластерів: \t Радіус диск.кластерів:");
if (CheckBox36->Checked==true) List3->Add (Edit178->Text+'\t'+Edit179->Text);
if (CheckBox37->Checked==true) List3->Add("Модель дефектів в ід. част. плівки - сферичні кластери (точкові дефекти)");
if (CheckBox37->Checked==true) List3->Add("Конц.сф.кластерів (т.д.): \t Радіус сф.кластерів (т.д.):");
if (CheckBox37->Checked==true) List3->Add (Edit181->Text+'\t'+Edit182->Text);
}
}
//}

if (RadioButton16->Checked==true)  //Поверхня з мозаїчним шаром (без врах. диф.розс. в  ід. част. монокр. та ППШ)
{
List3->Add(" ");
List3->Add("Поверхня з мозаїчним шаром ");
double    Snn,Afi;  //,fff1[100],fi[100];
double *fff1, *fi;
  fff1  = new double[100];
  fi    = new double[100];
List3->Add("Параметри мозаїчного шару");
List3->Add("N \t nn_m,  \t DFi, A ");
for (int k=0; k<=km_rozv;k++)
{
sprintf(ss,"%3i\t%3.6lf\t%3.1lf",k,nn_m[k],DFi[k]);
List3->Add(ss);
}

List3->Add(" ");
List3->Add(" Ф-ї розподілу розворотів блоків від кута:");
    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;
  Afi=StrToFloat(Edit80->Text); // Коеф. в  DD_rozv[kr] (fi[kr]);
  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
for (int k=0; k<=km_rozv;k++)
{
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf",fi[k],DD_rozv[k]*10000,fff1[k]*100);
List3->Add(ss);
}
delete fff1, fi;
}





if (CheckBox67->Checked==true)        // Поруш. шар
{
List3->Add(" ");
if (RadioButton11->Checked==true || RadioButton13->Checked==true) List3->Add("Когер. складова КДВ від ППШ розрах. за р-нями Такагі-Топена");
if (RadioButton12->Checked==true || RadioButton16->Checked==true) List3->Add("Когер. складова КДВ від ППШ розрах. за кінематичною теорією");
if (CheckBox1->Checked==false && CheckBox2->Checked==false && CheckBox4->Checked==false && CheckBox26->Checked==false && CheckBox58->Checked==false)
List3->Add("Дефекти у ППШ не враховуються");
else
{
 List3->Add("Дефекти у ППШ:");
if (CheckBox1->Checked==true) // Дислокаційні петлі (усереднені)
{
if (RadioButton7->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/sqrt(2)");
if (RadioButton8->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a");
if (RadioButton9->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a*sqrt(2)");
if (RadioButton28->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)");
if (RadioButton32->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)");
if (RadioButton29->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a*sqrt(3)");
AnsiString namekoef=" Коефіцієнт на L: ";
List3->Add(namekoef+'\t'+Edit149->Text);
if (CheckBox6->Checked==true)  List3->Add("Концентрація дефектів пропорційна профілю деформації");
if (CheckBox6->Checked==false) List3->Add("Концентрація дефектів однакова по всьому профілю деформації");
if (CheckBox7->Checked==true)  List3->Add("Радіус дефектів пропорційний профілю деформації");
if (CheckBox7->Checked==false) List3->Add("Радіус дефектів однаковий по всьому профілю деформації");
List3->Add("Конц.петель: \t Радіус петель:");
List3->Add (Edit2->Text+'\t'+Edit3->Text);
}
if (CheckBox58->Checked==true)  // Дислокаційні петлі (анізотропія [111])
{
if (RadioButton7->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/sqrt(2)");
if (RadioButton8->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a");
if (RadioButton9->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a*sqrt(2)");
if (RadioButton28->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/3*sqrt(3)");
if (RadioButton32->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/2*sqrt(3)");
if (RadioButton29->Checked==true) List3->Add("Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a*sqrt(3)");
AnsiString namekoef=" Коефіцієнт на L: ";
List3->Add(namekoef+'\t'+Edit149->Text);
if (CheckBox39->Checked==true)  List3->Add("Концентрація дефектів пропорційна профілю деформації");
if (CheckBox39->Checked==false) List3->Add("Концентрація дефектів однакова по всьому профілю деформації");
if (CheckBox61->Checked==true)  List3->Add("Радіус дефектів пропорційний профілю деформації");
if (CheckBox61->Checked==false) List3->Add("Радіус дефектів однаковий по всьому профілю деформації");
List3->Add("Конц.петель: \t Радіус петель:");
List3->Add (Edit218->Text+'\t'+Edit219->Text);
}

if (CheckBox2->Checked==true)
{
List3->Add("Модель дефектів в ПШ - сферичні кластери");
if (CheckBox8->Checked==true)  List3->Add("Концентрація дефектів пропорційна профілю деформації");
if (CheckBox8->Checked==false) List3->Add("Концентрація дефектів однакова по всьому профілю деформації");
if (CheckBox9->Checked==true)  List3->Add("Радіус дефектів пропорційний профілю деформації");
if (CheckBox9->Checked==false) List3->Add("Радіус дефектів однаковий по всьому профілю деформації");
List3->Add("Конц.сф.кластерів: \t Радіус сф.кластерів:");
List3->Add (Edit14->Text+'\t'+Edit15->Text);
}
if (CheckBox26->Checked==true)
{
List3->Add("Модель дефектів в ПШ - сферичні кластери (точкові дефекти)");
if (CheckBox8->Checked==true)  List3->Add("Концентрація дефектів пропорційна профілю деформації");
if (CheckBox8->Checked==false) List3->Add("Концентрація дефектів однакова по всьому профілю деформації");
if (CheckBox9->Checked==true)  List3->Add("Радіус дефектів пропорційний профілю деформації");
if (CheckBox9->Checked==false) List3->Add("Радіус дефектів однаковий по всьому профілю деформації");
List3->Add("Конц.сф.кластерів (т.д.): \t Радіус сф.кластерів (т.д.):");
List3->Add (Edit250->Text+'\t'+Edit251->Text);
}
if (CheckBox4->Checked==true)
{
List3->Add("Модель дефектів в ПШ - дископодібні кластери");
if (CheckBox10->Checked==true)  List3->Add("Концентрація дефектів пропорційна профілю деформації");
if (CheckBox10->Checked==false) List3->Add("Концентрація дефектів однакова по всьому профілю деформації");
if (CheckBox11->Checked==true)  List3->Add("Радіус дефектів пропорційний профілю деформації");
if (CheckBox11->Checked==false) List3->Add("Радіус дефектів однаковий по всьому профілю деформації");
List3->Add("Конц.диск.кластерів: \t Радіус диск.кластерів:");
List3->Add (Edit24->Text+'\t'+Edit25->Text);
}
if (CheckBox3->Checked==true)
{
List3->Add("Додаткова аморфізація в ПШ");
List3->Add("Кофіцієнт: \t Степінь:");
List3->Add (Edit398->Text+'\t'+Edit399->Text);
List3->Add("Emin444): \t Emin(888): \t Emin(880):");
List3->Add (Edit131->Text+'\t'+Edit396->Text+'\t'+Edit397->Text);
}
if (CheckBox88->Checked==true)
{
List3->Add("Додаткова аморфізація в ПШ");
List3->Add("Кофіцієнт: \t Степінь:");
List3->Add (Edit407->Text+'\t'+Edit408->Text);
List3->Add("Emin444): \t Emin(888): \t Emin(880):");
List3->Add (Edit409->Text+'\t'+Edit410->Text+'\t'+Edit411->Text);
}
}
List3->Add(" ");
List3->Add(" Параметри  профілю:");

if (RadioButton3->Checked==true || RadioButton34->Checked==true)
{
if (RadioButton3->Checked==true) List3->Add(" Профіль - гаусіана з параметрами:");
if (RadioButton34->Checked==true) List3->Add(" Профіль обч. з дефектів, f - гаусіана з параметрами:");
param[1] ="Dmax1   ";
param[2] ="D01     ";
param[3] ="L1      ";
param[4] ="Rp1     ";
param[5] ="D02     ";
param[6] ="L2      ";
param[7] ="Rp2     ";
param[8] ="Dmin    ";
param[9] ="Emin    ";
param[10]="km      ";
param[11]="dl      ";
param[12]="Dmax    ";
param[13]="L       ";
for (int q=1; q<=kp; q++)
{
GDDparam[q]=FloatToStr(DDparam0[q]);
List3->Add(param[q]+'\t'+GDDparam[q]);
}

List3->Add(" ");
List3->Add(" Профілі деформації та аморфізації:");
sprintf(ss,"z \tDD1,% \tDD2,% \tDD,% \tE");
List3->Add(ss);
for (int q=1; q<=(int)DDparam0[10]; q++)
{
z1=DDparam0[13]-DDparam0[11]*q+DDparam0[11]/2.;
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf\t%3.6lf\t%3.6lf",z1,DDstart[1][q]*100,DDstart[2][q]*100,DDstart[0][q]*100,Esum_[q]);
List3->Add(ss);
}
//      Представлення б.-я. профілю сходинками
//L_shod=0;
L_shod=DDparam0[10]*DDparam0[11]; //km*dl
Z_shod[0]=0;
for (int k=1; k<=(int)DDparam0[10];k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+DDparam0[11];
D_shod[2*k-1]=DDstart[0][(int)DDparam0[10]-k+1];
D_shod[2*k  ]=DDstart[0][(int)DDparam0[10]-k+1];
}
Z_shod[2*(int)DDparam0[10]+1]=L_shod;
D_shod[2*(int)DDparam0[10]+1]=0;
List3->Add("   Профіль - гаусіана у вигляді сходинок:");
for (int q=1; q<=2*km+1; q++)
{
sprintf(ss,"%3.0lf\t%3.6lf",Z_shod[q],D_shod[q]*100);
List3->Add(ss);
}
}

if (RadioButton4->Checked==true)
{
List3->Add(" Профіль - сходинки:");
AnsiString kpid=" Кількість підшарів:";
km=StrToInt(Edit90->Text);
List3->Add(kpid+'\t'+IntToStr(km));
ReadMemo2stovp(Memo5,km,DD,Dl);
for (int k=1; k<=km;k++) Dl[k]=Dl[k]*1e-8;
double Ltovsh=0.;
for (int q=1; q<=km; q++)  Ltovsh=Ltovsh+Dl[q];
AnsiString tov=" Товщина порушеного шару (A): ";
List3->Add(tov+'\t'+FloatToStr(Ltovsh*1e8));
sprintf(ss,"№ \t Деформ. підшару (%) \t Товщина підшару (А)");
List3->Add(ss);
for (int q=1; q<=km; q++)
{
sprintf(ss,"%i\t%3.6lf\t%3.0lf",q,DD[q]*100,Dl[q]*1e8);
List3->Add(ss);
}
List3->Add(" Профілі деформації та аморфізації:");
sprintf(ss,"z, A \tDD, % \tE");
List3->Add(ss);
//z1=0.;
double zs=0.;
for (int q=1; q<=km; q++)
{
z1=Ltovsh-zs-Dl[q]+Dl[q]/2.;
zs=zs+Dl[q];
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf",z1*1e8,DD[q]*100,Esum_[q]);
List3->Add(ss);
}
//      Представлення б.-я. профілю сходинками
 L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+Dl[k];
Z_shod[0]=0;
Dl[km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+Dl[km-k+1];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
List3->Add("Профіль-сходинки у вигляді сходинок:");
for (int q=1; q<=2*km+1; q++)
{
sprintf(ss,"%3.0lf\t%3.6lf",Z_shod[q]*1e8,D_shod[q]*100);
List3->Add(ss);
}
}
}

if (fitting==1 || fitting==10)
{
List3->Add("");
List3->Add("*****************************************************************");
List3->Add("");
if (RadioButton22->Checked==true || RadioButton23->Checked==true || RadioButton24->Checked==true || RadioButton25->Checked==true || RadioButton27->Checked==true)
List3->Add("Результати наближення за програмами типу Auto");
if (RadioButton30->Checked==true) List3->Add("Результати наближення за програмами типу Gausauto");
List3->Add("Задана кількість циклів = "+Edit73->Text);
if (CheckBox21->Checked==true) List3->Add("Наближ. в даному напрямку");
if (CheckBox21->Checked==false) List3->Add("Наближ. в даному напрямку не викор.");
if (CheckBox25->Checked==true) List3->Add("Зменшення кроку");
if (CheckBox25->Checked==false) List3->Add("Крок не змінюється");
if (CheckBox29->Checked==true) List3->Add("Мінімізація ВСКВ");
if (CheckBox29->Checked==false) List3->Add("Мінімізація АСКВ");

if (RadioButton22->Checked==true)   // Наближення  диф.розс. в ід. част. монокр.
{
List3->Add(" ");
List3->Add("Наближення  диф.розс. в ід. част. монокр.");
List3->Add("Результати  наближення шляхом зміни параметрів дислокаційних петель:");
List3->Add("nL0, см-3 (старт.) \t R0, А (старт.) \t nL0, см-3 \t R0, А");
if (CheckBox79->Checked==true) {sprintf(ss,"%3.2le\t%3.0lf\t%3.2le\t%3.0lf",DDa[5][1],DLa[5][1]*1e8,DDa[method_][1],DLa[method_][1]*1e8); List3->Add(ss);}
if (CheckBox80->Checked==true) {sprintf(ss,"%3.2le\t%3.0lf\t%3.2le\t%3.0lf",DDa[5][2],DLa[5][2]*1e8,DDa[method_][2],DLa[method_][2]*1e8); List3->Add(ss);}
AnsiString dnLdR0=" dnL,  dR0 ";
List3->Add(dnLdR0+'\t'+Edit66->Text+'\t'+Edit67->Text);
}

if (RadioButton27->Checked==true)   // Наближення  диф.розс. в ід. част. плівки
{
List3->Add(" ");
List3->Add("Наближення  диф.розс. в ід. част. плівки");
List3->Add("Результати  наближення шляхом зміни параметрів дислокаційних петель:");
List3->Add("nL0, см-3 (старт.) \t R0, А (старт.) \t nL0, см-3 \t R0, А");
if (CheckBox34->Checked==true) {sprintf(ss,"%3.2le\t%3.0lf\t%3.2le\t%3.0lf",DDa[5][1],DLa[5][1]*1e8,DDa[method_][1],DLa[method_][1]*1e8); List3->Add(ss);}
if (CheckBox35->Checked==true) {sprintf(ss,"%3.2le\t%3.0lf\t%3.2le\t%3.0lf",DDa[5][2],DLa[5][2]*1e8,DDa[method_][2],DLa[method_][2]*1e8); List3->Add(ss);}
AnsiString dnLdR0=" dnL,  dR0 ";
List3->Add(dnLdR0+'\t'+Edit66->Text+'\t'+Edit67->Text);
}

if (RadioButton23->Checked==true)  // Набл. диф.розс. в ППШ (з врах. диф.розс. в  ід. част. монокр та ППШ (гаусіана).)
{
List3->Add(" ");
List3->Add("Наближення  диф.розс. в ППШ");
if (CheckBox1->Checked==true) List3->Add("Результати  наближення  шляхом зміни параметрів дислокаційних петель:");
if (CheckBox2->Checked==true) List3->Add("Результати  наближення  шляхом зміни параметрів сферичних кластерів:");
List3->Add("n_max, см-3 (старт.) \t R_max, А (старт.) \t n_max, см-3 \t R_max, А");
//List3->Add(FloatToStr(DDa[5][1])+'\t'+FloatToStr(DLa[5][1]*1e8)+'\t'+FloatToStr(DDa[method_][1])+'\t'+FloatToStr(DLa[method_][1]*1e8));
sprintf(ss,"%3.2le\t%3.0lf\t%3.2le\t%3.0lf",DDa[5][1],DLa[5][1]*1e8,DDa[method_][1],DLa[method_][1]*1e8);
List3->Add(ss);
AnsiString dnLdR0=" dnL,  dR0 ";
List3->Add(dnLdR0+'\t'+Edit66->Text+'\t'+Edit67->Text);
}

if (RadioButton24->Checked==true)  // Набл. профіль  сходинками в ППШ (з врах. диф.розс. в  ід. част. монокр. та ППШ)
{                  
List3->Add(" ");
if (CheckBox24->Checked==false) List3->Add("Наближення  профілю-сходинок  в ППШ");
if (CheckBox24->Checked==true) List3->Add("Наближення  профілю-сходинок в ППШ та парам.дефектів в ППШ)");
if ( CheckBox23->Checked==true) List3->Add(" Стартовий профіль - гаусіана");
if ( CheckBox23->Checked==false) List3->Add(" Стартовий профіль - профіль сходинками");
List3->Add("Результати  наближення профілю");
List3->Add("N \t DD, % (старт.) \t DL, A (старт.) \t DD, % \t DL, A");
for (int k=1; k<=km;k++)
{
//List3->Add(FloatToStr(k)+'\t'+FloatToStr(DDa[5][k])+'\t'+FloatToStr(DLa[5][k]*1e8)+'\t'+FloatToStr(DDa[method_][k])+'\t'+FloatToStr(DLa[method_][k]*1e8));
sprintf(ss,"%3i\t%3.6lf\t%3.1lf\t%3.6lf\t%3.1lf",k,DDa[5][k]*100,DLa[5][k]*1e8,DDa[method_][k]*100,DLa[method_][k]*1e8);
List3->Add(ss);
}
AnsiString dnLdR0=" dDD,  dL ";
List3->Add(dnLdR0+'\t'+Edit93->Text+'\t'+Edit94->Text);

List3->Add(" ");
List3->Add(" Наближені профілі деформації та аморфізації:");
double Ltovsh=0.;
for (int q=1; q<=km; q++)  Ltovsh=Ltovsh+DLa[method_][q];
List3->Add(" Товщина порушеного шару: "+'\t'+FloatToStr(Ltovsh*1e8));
//z1=0.;
List3->Add("z, A \t DD, %");
double zs=0.;
for (int q=1; q<=km; q++)
{
z1=Ltovsh-zs-DLa[method_][q]+DLa[method_][q]/2.;
zs=zs+DLa[method_][q];
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf",z1*1e8,DDa[method_][q]*100,Esum[q]);
List3->Add(ss);
}
//      Представлення б.-я. профілю сходинками
L_shod=0;
for (int k=1; k<=km;k++) L_shod=L_shod+DLa[method_][k];
Z_shod[0]=0;
DLa[method_][km+1]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+DLa[method_][km-k+1];
D_shod[2*k-1]=DDa[method_][km-k+1];
D_shod[2*k  ]=DDa[method_][km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
List3->Add("   Наближений профіль у вигляді сходинок:");
for (int q=1; q<=2*km+1; q++) 
{
sprintf(ss,"%3.0lf\t%3.6lf",Z_shod[q]*1e8,D_shod[q]*100);
List3->Add(ss);
}
}

if (RadioButton30->Checked==true)  // Наближення  профілю-гаусіни  в ППШ  за програмою Gausauto
{                  
List3->Add(" ");
List3->Add("Наближення  профілю-гаусіни  в ППШ");
List3->Add("Результати  наближення профілю шляхом зміни параметрів гаусіан");
List3->Add("Параметр:  \t Старт.  \t  Кінц.   \t  Крок");
int k_param=StrToInt(Edit127->Text);
for (int k=1; k<=k_param;k++) List3->Add(param[k]+'\t'+FloatToStr(PARAM[5][k])+'\t'+FloatToStr(PARAM[method_][k])+'\t'+FloatToStr(STEP[k]));

List3->Add("Всі параметри наближеного профілю:");
for (int q=1; q<=kp; q++)
{
GDDparam[q]=FloatToStr(DDparam[q]);
List3->Add(param[q]+'\t'+GDDparam[q]);
}
List3->Add(" ");
List3->Add(" Наближені профілі деформації та аморфізації:");
sprintf(ss,"z \tDD1,% \tDD2,% \tDD,% \tE");
List3->Add(ss);
for (int q=1; q<=km; q++)
{
z1=DDparam[13]-DDparam[11]*q+DDparam[11]/2.;
//zz=FloatToStr(z1);
//GDDPL1[q]=FloatToStr(DDPL1[q]*100);
//GDDPL2[q]=FloatToStr(DDPL2[q]*100);
//GDD[q]=FloatToStr((DDPL1[q]+DDPL2[q])*100);
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf\t%3.6lf\t%3.6lf",z1,DDPL1[q]*100,DDPL2[q]*100,(DDPL1[q]+DDPL2[q])*100,Esum[q]);
List3->Add(ss);
//GDDvse[q]=(zz+'\t'+GDDPL1[q]+'\t'+GDDPL2[q]+'\t'+GDD[q]+'\t'+FloatToStr(EL[q]));
//List3->Add(GDDvse[q]);
}
//      Представлення б.-я. профілю сходинками
//L_shod=0;
 L_shod=DDparam[10]*DDparam[11]; //km*dl
Z_shod[0]=0;
for (int k=1; k<=km;k++)
{
Z_shod[2*k-1]=Z_shod[2*k-2];
Z_shod[2*k  ]=Z_shod[2*k-1]+DDparam[11];
D_shod[2*k-1]=DD[km-k+1];
D_shod[2*k  ]=DD[km-k+1];
}
Z_shod[2*km+1]=L_shod;
D_shod[2*km+1]=0;
List3->Add("   Наближений профіль-гаусіана у вигляді сходинок:");
for (int q=1; q<=2*km+1; q++)
{
// List3->Add(FloatToStr(Z_shod[q])+'\t'+FloatToStr(D_shod[q]*100));
sprintf(ss,"%3.0lf\t%3.6lf",Z_shod[q],D_shod[q]*100);
List3->Add(ss);
}
}

if (RadioButton25->Checked==true)  //Набл. ф-ю розподілу розворотів блоків від кута (без врах. диф.розс. в  ід. част. монокр. та ППШ)
{
List3->Add(" ");
List3->Add("Наближення ф-ї розподілу розворотів блоків від кута");
double    Snn,Afi;  //,fff1[100],fi[100];
double *fff1, *fi;
  fff1  = new double[100];
  fi    = new double[100];
for (int k=0; k<=km_rozv; k++)
{
nn_m[k]=DDa[method_lich][k+1];
DFi[k]=DLa[method_lich][k+1];
}
List3->Add("Результати  наближення");
List3->Add("N \t nn_m, % (старт.) \t DFi, A (старт.) \t nn_m, % \t DFi, A");
for (int k=1; k<=km_rozv;k++)
{
sprintf(ss,"%3i\t%3.6lf\t%3.1lf\t%3.6lf\t%3.1lf",k,DDa[5][k]*100,DLa[5][k]*1e8,DDa[method_][k]*100,DLa[method_][k]*1e8);
List3->Add(ss);
}
AnsiString dnLdR0=" dnn_m,  dDFi ";
List3->Add(dnLdR0+'\t'+Edit133->Text+'\t'+Edit136->Text);

List3->Add(" ");
List3->Add(" Наближені ф-ї розподілу розворотів блоків від кута:");
    Snn=0;                  // нормув. функції розподілу по кутах
    for (int kr=0; kr<=km_rozv;kr++) Snn=Snn+nn_m[kr];
    for (int kr=0; kr<=km_rozv;kr++)    fff1[kr]=nn_m[kr]/Snn;
  Afi=StrToFloat(Edit80->Text); // Коеф. в  DD_rozv[kr] (fi[kr]);
  fi[0]=0;
  for (int kr=1; kr<=km_rozv;kr++)
  {
//    DD_rozv[kr]=DD_rozv[kr-1]+0.0000162;
//    fi[kr]=DD_rozv[kr]*DD_rozv[kr]/(Afi*Afi)/M_PI*180*3600;
    fi[kr]=fi[kr-1]+DFi[kr];
    DD_rozv[kr]=Afi*sqrt(fi[kr]/3600*M_PI/180);
  }
for (int k=0; k<=km_rozv;k++)
{
sprintf(ss,"%3.0lf\t%3.6lf\t%3.6lf",fi[k],DD_rozv[k]*10000,fff1[k]*100);
List3->Add(ss);
}
delete fff1, fi;
}

}
List3->Add(" ");
List3->Add(" Теоретичні та експериментальна КДВ:");
AnsiString Skv=" CKB(загальне)= ";
List3->Add(Skv+'\t'+FloatToStr(CKV));
AnsiString kut=" Крок кута КДВ (с): ";
AnsiString nskv_=" nskv (с)= ";
AnsiString kskv_=" kskv (с)= ";
AnsiString koefCKV_=" koefCKV= ";
AnsiString nskv_r_=" nskv_r (с)= ";
AnsiString kskv_r_=" kskv_r (с)= ";

if (CheckBox42->Checked==true)  // (444)
{
List3->Add(" ");
List3->Add(" КДВ (444)");
if (CheckBox54->Checked==false) List3->Add(kut+'\t'+FloatToStr(ik_[1]));
if (CheckBox54->Checked==true) List3->Add(kut+'\t'+FloatToStr(ik_[1])+'\t'+Edit213->Text);
sprintf(ss,"m1= % i\tm10= % i", m1_[1],m10_[1]);
List3->Add(ss);
List3->Add(nskv_+'\t'+Edit69->Text+'\t'+kskv_+'\t'+Edit70->Text+'\t'+koefCKV_+'\t'+Edit400->Text);
if (CheckBox22->Checked==true)
List3->Add(nskv_r_+'\t'+Edit140->Text+'\t'+kskv_r_+'\t'+Edit141->Text);
AnsiString Skv=" CKB= ";
List3->Add(Skv+'\t'+FloatToStr(CKV_[1]));
if (CheckBox62->Checked==true) List3->Add(" Дифузна складова від ПШ не обчислювалася");
List3->Add(" ");
sprintf(ss,"Кут (с) \tІ (диф.ід.кр.)\tІ (диф.ід.част.пл.)\tІ (диф.ПШ)\tІ (диф.)\tІ (когер.)\tІ (теор.)\tІ (теор.+АФ)\tІ (експ.)\tІ (експ.0)");
List3->Add(ss);
TetaMin=-(m10_[1])*ik_[1];
for (int q=0; q<=m1_[1]; q++)
{
w=TetaMin+q*ik_[1];
sprintf(ss,"%3.6lf\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.61f",w,R_dif_0_[q][1],R_dif_0pl_[q][1],R_dif_dl_[q][1],R_dif_[q][1],R_cogerTT_[q][1],R_vse_[q][1],R_vseZg[q][1],intIk2d[q][1],intI02d[q][1],DeltaTeta[q]/M_PI*(3600.*180.));
List3->Add(ss);
}
}
if (CheckBox43->Checked==true)  // (888)
{
List3->Add(" ");
List3->Add(" КДВ (888)");
//List3->Add(kut+'\t'+FloatToStr(ik_[2]));
if (CheckBox54->Checked==false) List3->Add(kut+'\t'+FloatToStr(ik_[2]));
if (CheckBox54->Checked==true) List3->Add(kut+'\t'+FloatToStr(ik_[2])+'\t'+Edit214->Text);
sprintf(ss,"m1= % i\tm10= % i", m1_[2],m10_[2]);
List3->Add(ss);
List3->Add(nskv_+'\t'+Edit130->Text+'\t'+kskv_+'\t'+Edit129->Text+'\t'+koefCKV_+'\t'+Edit401->Text);
if (CheckBox45->Checked==true)
List3->Add(nskv_r_+'\t'+Edit142->Text+'\t'+kskv_r_+'\t'+Edit143->Text);
AnsiString Skv=" CKB= ";
List3->Add(Skv+'\t'+FloatToStr(CKV_[2]));
if (CheckBox63->Checked==true) List3->Add(" Дифузна складова від ПШ не обчислювалася");
List3->Add(" ");
sprintf(ss,"Кут (с) \tІ (диф.ід.кр.)\tІ (диф.ід.част.пл.)\tІ (диф.ПШ)\tІ (диф.)\tІ (когер.)\tІ (теор.)\tІ (теор.+АФ)\tІ (експ.)\tІ (експ.0)");
List3->Add(ss);
TetaMin=-(m10_[2])*ik_[2];
for (int q=0; q<=m1_[2]; q++)
{
w=TetaMin+q*ik_[2];
sprintf(ss,"%3.6lf\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.61f",w,R_dif_0_[q][2],R_dif_0pl_[q][2],R_dif_dl_[q][2],R_dif_[q][2],R_cogerTT_[q][2],R_vse_[q][2],R_vseZg[q][2],intIk2d[q][2],intI02d[q][2],DeltaTeta[q]/M_PI*(3600.*180.));
List3->Add(ss);
}
}
if (CheckBox44->Checked==true)  // (880)
{
List3->Add(" ");
List3->Add(" КДВ (880)");
//List3->Add(kut+'\t'+FloatToStr(ik_[3]));
if (CheckBox54->Checked==false) List3->Add(kut+'\t'+FloatToStr(ik_[3]));
if (CheckBox54->Checked==true) List3->Add(kut+'\t'+FloatToStr(ik_[3])+'\t'+Edit215->Text);
sprintf(ss,"m1= % i\tm10= % i", m1_[3],m10_[3]);
List3->Add(ss);
List3->Add(nskv_+'\t'+Edit89->Text+'\t'+kskv_+'\t'+Edit134->Text+'\t'+koefCKV_+'\t'+Edit402->Text);
if (CheckBox46->Checked==true)
List3->Add(nskv_r_+'\t'+Edit144->Text+'\t'+kskv_r_+'\t'+Edit145->Text);
AnsiString Skv=" CKB= ";
List3->Add(Skv+'\t'+FloatToStr(CKV_[3]));
if (CheckBox64->Checked==true) List3->Add(" Дифузна складова від ПШ не обчислювалася");
List3->Add(" ");
sprintf(ss,"Кут (с) \tІ (диф.ід.кр.)\tІ (диф.ід.част.пл.)\tІ (диф.ПШ)\tІ (диф.)\tІ (когер.)\tІ (теор.)\tІ (теор.+АФ)\tІ (експ.)\tІ (експ.0)");
List3->Add(ss);
TetaMin=-(m10_[3])*ik_[3];
for (int q=0; q<=m1_[3]; q++)
{
w=TetaMin+q*ik_[3];
sprintf(ss,"%3.6lf\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6le\t%3.6lf",w,R_dif_0_[q][3],R_dif_0pl_[q][3],R_dif_dl_[q][3],R_dif_[q][3],R_cogerTT_[q][3],R_vse_[q][3],R_vseZg[q][3],intIk2d[q][3],intI02d[q][3],DeltaTeta[q]/M_PI*(3600.*180.));
List3->Add(ss);
}           
}

List3->SaveToFile(SaveKDB1->FileName);        
}
delete GDDparam, param;
delete Z_shod, D_shod;
delete ss;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::N6Click(TObject *Sender) //Відкр. апар. ф-ю
{
int m1ez, m10ez;
double ik, TetaMin,DeltaTeta1,Ap_sum;
//double  DeltaTetaAF[MM],PO[MM],POk[MM];
//double  nkoef_dTeta, kkoef_dTeta;
int  koef_dTeta;
double  *DeltaTetaAF, *PO, *POk;
  DeltaTetaAF  = new double[MM];
  PO   = new double[MM];
  POk  = new double[MM];

if ((RadioButton10->Checked==true && CheckBox59->Checked==true)||
 (RadioButton20->Checked==true && CheckBox60->Checked==true)||
 (RadioButton21->Checked==true && CheckBox71->Checked==true))
{
double y0,xc,w,A,Ymin,Xmin,x,Xmin1,Xmin2,eta;
if (vved_exper==0) ik=StrToFloat(Edit19->Text);
if (vved_exper==1 || vved_exper==2)
{
if (RadioButton10->Checked==true) ik=ik_[1];
if (RadioButton20->Checked==true) ik=ik_[2];
if (RadioButton21->Checked==true) ik=ik_[3];
}

if (RadioButton10->Checked==true)     // (444)
{
w=StrToFloat(Edit161->Text);
A=StrToFloat(Edit160->Text);
Ymin=StrToFloat(Edit159->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit162->Text=FloatToStr(Xmin);
Edit163->Text=IntToStr(m1ez+1);
param_AF[0][1]=Ymin;
param_AF[1][1]=A;
param_AF[2][1]=w;
param_AF[3][1]=Xmin;
param_AF[4][1]=m1ez+1;
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit152->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][1]=POk[i];
m1_[11]=m1ez;
m10_[11]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series2->AddXY(DeltaTeta1,POk[i],"",clWhite);
}
}

if (RadioButton20->Checked==true)        // (888)
{
w=StrToFloat(Edit227->Text);
A=StrToFloat(Edit226->Text);
Ymin=StrToFloat(Edit225->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit228->Text=FloatToStr(Xmin);
Edit229->Text=IntToStr(m1ez+1);
param_AF[0][2]=Ymin;
param_AF[1][2]=A;
param_AF[2][2]=w;
param_AF[3][2]=Xmin;
param_AF[4][2]=m1ez+1;
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit385->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][2]=POk[i];
m1_[12]=m1ez;
m10_[12]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series51->AddXY(DeltaTeta1,POk[i],"",clWhite);
}
}

if (RadioButton21->Checked==true)     // (880)
{
w=StrToFloat(Edit232->Text);
A=StrToFloat(Edit231->Text);
Ymin=StrToFloat(Edit230->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit233->Text=FloatToStr(Xmin);
Edit234->Text=IntToStr(m1ez+1);
param_AF[0][3]=Ymin;
param_AF[1][3]=A;
param_AF[2][3]=w;
param_AF[3][3]=Xmin;
param_AF[4][3]=m1ez+1;
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit391->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][3]=POk[i];
m1_[13]=m1ez;
m10_[13]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series53->AddXY(DeltaTeta1,POk[i],"",clWhite);
}
}


/*
if (RadioButton10->Checked==true)
{
w=StrToFloat(Edit161->Text);
A=StrToFloat(Edit160->Text);
Ymin=StrToFloat(Edit159->Text);
}
if (RadioButton20->Checked==true)
{
w=StrToFloat(Edit227->Text);
A=StrToFloat(Edit226->Text);
Ymin=StrToFloat(Edit225->Text);
}
if (RadioButton21->Checked==true)
{
w=StrToFloat(Edit232->Text);
A=StrToFloat(Edit231->Text);
Ymin=StrToFloat(Edit230->Text);
}
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
if (RadioButton10->Checked==true)
{
param_AF[0][1]=Ymin;
param_AF[1][1]=A;
param_AF[2][1]=w;
param_AF[3][1]=Xmin;
param_AF[4][1]=m1ez+1;
}
if (RadioButton20->Checked==true)
{
param_AF[0][2]=Ymin;
param_AF[1][2]=A;
param_AF[2][2]=w;
param_AF[3][2]=Xmin;
param_AF[4][2]=m1ez+1;
}
if (RadioButton21->Checked==true)
{
param_AF[0][3]=Ymin;
param_AF[1][3]=A;
param_AF[2][3]=w;
param_AF[3][3]=Xmin;
param_AF[4][3]=m1ez+1;
}

TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
if (RadioButton10->Checked==true) koef_dTeta=StrToInt(Edit152->Text);
if (RadioButton20->Checked==true) koef_dTeta=StrToInt(Edit385->Text);
if (RadioButton21->Checked==true) koef_dTeta=StrToInt(Edit391->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
//Edit162->Text=FloatToStr(Xmin);
//Edit163->Text=IntToStr(m1ez+1);
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)  DeltaTetaAF[i]=(TetaMin+i*ik/koef_dTeta);

for (int i=0; i<=m1ez; i++)
{
x=DeltaTetaAF[i];
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}

if (RadioButton10->Checked==true)
{
Edit162->Text=FloatToStr(Xmin);
Edit163->Text=IntToStr(m1ez+1);
}
if (RadioButton20->Checked==true)
{
Edit228->Text=FloatToStr(Xmin);
Edit229->Text=IntToStr(m1ez+1);
}
if (RadioButton21->Checked==true)
{
Edit233->Text=FloatToStr(Xmin);
Edit234->Text=IntToStr(m1ez+1);
}
}*/
}

if ((RadioButton10->Checked==true && CheckBox59->Checked==false)||
 (RadioButton20->Checked==true && CheckBox60->Checked==false)||
 (RadioButton21->Checked==true && CheckBox71->Checked==false))
{
AnsiString MyFName="";
if (OtkAf->Execute())
{
MyFName=OtkAf->FileName;
TStringList *List = new TStringList;
AnsiString Ds11;
//AnsiString Mas[MM], Mas1[MM];
//AnsiString Mas2[10000];
//double X[MM];
//double Y[10000];
double *X;
  X  = new double[MM];
AnsiString *Mas, *Mas1;
  Mas  = new AnsiString[MM];
  Mas1 = new AnsiString[MM];


List->LoadFromFile(OtkAf->FileName);//зчитується файл NameFile

double       p11,p12,p13 ;
AnsiString   p1,p2,p3 ;
p1=List->Strings[0];// зчитуємо у масив рядки файла
p2=List->Strings[1];// зчитуємо у масив рядки файла
p3=List->Strings[2];// зчитуємо у масив рядки файла
p11=atof(p1.c_str());
p12=atof(p2.c_str());
p13=atof(p3.c_str());
m1ez=p11-1;
if (ik!=p12) MessageBox(0,"Крок в АФ та експ.КДВ чи програмі не співпадають","Увага!", MB_OK + MB_ICONEXCLAMATION);
//ik=p12;
m10ez=p13;


// потрібно визначити к-ть рядків у файлі
for (int i=3;i<=m1ez+3;i++)
{
Mas[i]=List->Strings[i];// зчитуємо у масив рядки файла
Mas1[i]="";
for (int k=1; k<=(Mas[i].Length());k++)
{
Ds11=Mas[i][k]; //допоміжна змінна типу AnsiString
if (Ds11!=("\t"))
if (Ds11!=(" "))
Mas1[i]=Mas1[i]+Ds11; //у масив Mas1 заносяться значення
//першого стовпця файла Test0
else break;
else break;
}
X[i]=atof(Mas1[i].c_str());//перший стовбець переводиться із тексту в числа
}
for (int i=3;i<=m1ez+3;i++) PO[i-3]=X[i];

delete X, Mas, Mas1;
}
}

 Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;

if (RadioButton10->Checked==true)
{
for (int i=0; i<=m1ez; i++) POk2d[i][1]=POk[i];
m1_[11]=m1ez;
m10_[11]=m10ez;
}
if (RadioButton20->Checked==true)
{
for (int i=0; i<=m1ez; i++) POk2d[i][2]=POk[i];
m1_[12]=m1ez;
m10_[12]=m10ez;
}
if (RadioButton21->Checked==true)
{
for (int i=0; i<=m1ez; i++) POk2d[i][3]=POk[i];
m1_[13]=m1ez;
m10_[13]=m10ez;
}

TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
//Application->ProcessMessages();
if (RadioButton10->Checked==true) Series2->AddXY(DeltaTeta1,POk[i],"",clWhite);
if (CheckBox51->Checked==true)  Series51->AddXY(DeltaTetaAF[i],POk[i],"",clBlue);
if (RadioButton20->Checked==true) Series51->AddXY(DeltaTeta1,POk[i],"",clWhite);
if (RadioButton21->Checked==true) Series53->AddXY(DeltaTeta1,POk[i],"",clWhite);
//Memo1->Lines->Add(IntToStr(i)+'\t'+FloatToStr(DeltaTeta1)+'\t'+FloatToStr(DeltaTeta[i])+'\t'+FloatToStr(POk[i]));
}
delete  DeltaTetaAF, PO, POk;
}

//---------------------------------------------------------------------------

void TForm1::OpenAF_Lorenz() //Відкр. апар. ф-ю автоматично
{
int m1ez, m10ez;
double ik, TetaMin,DeltaTeta1;
//double  DeltaTetaAF[MM],PO[MM],POk[MM];
//double  nkoef_dTeta, kkoef_dTeta;
int  koef_dTeta, Ap_sum;
int  nsd,ep, ek,  op,  ok,jp,jk,m1ez_teor,m10ez_teor ;
double  *DeltaTetaAF, *PO, *POk;
double *POG,*POG1, *POL, *POPV,*POV;
  DeltaTetaAF  = new double[MM];
  PO    = new double[MM];
  POk   = new double[MM];
  POG   = new double[MM];
  POG1  = new double[MM];
  POL   = new double[MM];
  POPV  = new double[MM];
  POV   = new double[MM];
double y0,xc,w,A,Ymin,Xmin,x,Xmin1,Xmin2,eta,y0L;

if (vved_exper==0)
  {
  if (CheckBox42->Checked==true && CheckBox59->Checked==true) ik_[1]=StrToFloat(Edit19->Text);
  if (CheckBox43->Checked==true && CheckBox60->Checked==true) ik_[2]=StrToFloat(Edit19->Text);
  if (CheckBox44->Checked==true && CheckBox71->Checked==true) ik_[3]=StrToFloat(Edit19->Text);
  }

if (CheckBox42->Checked==true && CheckBox59->Checked==true)     // (444)
{
ik=ik_[1];
eta=StrToFloat(Edit289->Text);
w=StrToFloat(Edit161->Text);
A=StrToFloat(Edit160->Text);
Ymin=StrToFloat(Edit159->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
//Xmin=xc+0.5*sqrt(2*A*w/(Ymin-y0L)/M_PI-w*w);
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit162->Text=FloatToStr(Xmin);
Edit163->Text=IntToStr(m1ez+1);
param_AF[0][1]=Ymin;
param_AF[1][1]=A;
param_AF[2][1]=w;
param_AF[3][1]=Xmin;
param_AF[4][1]=m1ez+1;
param_AF[5][1]=eta;
y0L=-(0 + (2*A*w/M_PI)/(4*(Xmin-xc)*(Xmin-xc)+w*w));     // =0; !!!!!!!!!!!!!!!
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
POG[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
POPV[i]=eta*POG[i]+(1-eta)*POL[i];
//Memo8->Lines->Add(FloatToStr(POG[i])+'\t'+FloatToStr(POL[i])+'\t'+FloatToStr(POPV[i])+'\t'+FloatToStr(1111) );
}
//Memo9->Lines->Add( "aaa02 пройшло");

if (eta==1)
  for (int i=0; i<=m1ez; i++) PO[i]=POG[i];
if (eta==0)
  for (int i=0; i<=m1ez; i++) PO[i]=POL[i];
//for (int i=0; i<=m1ez; i++)  PO[i]=POL[i]-0.999*POL[m1ez];
if (eta>0 && eta<1)
  for (int i=0; i<=m1ez; i++) PO[i]=POPV[i];
//Memo9->Lines->Add( "aaa03 пройшло");
if (eta==2)
{
 ep=-m10ez;   //-m10;
 ek=m1ez-m10ez;     ///m1-m10;
 op=-m10ez;    //-m10z;      // АФ вся уточнена при розрахунку
 ok=m1ez-m10ez;     //m1z-m10z;   // АФ вся уточнена при розрахунку
 jp=ep-ok;
 jk=ek-op;
m10ez_teor=-jp;        // =m10+(m1z-m10z)
m1ez_teor=-jp+jk;      // =m10+(m1z-m10z)+m1-m10-(-m10z)=m1+m1z      =260
//Memo9->Lines->Add( "aaa2 пройшло");

TetaMin=-(m10ez_teor)*ik;
for (int i=0; i<=m1ez_teor; i++)
{
x=(TetaMin+i*ik);
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
}
 jp=-m10ez_teor;   //-m10_teor;             // 260/275
 jk=m1ez_teor-m10ez_teor;   //m1_teor-m10_teor;
 ep=jp+ok;     //бо АФ вся уточнена
 ek=jk+op;
 nsd=-jp;

for (int i=m1ez; i>=0; i--) POG1[i+nsd-m10ez]=POG[i];
for (int j=ep; j<=ek; j++)                     //   do 13 j=ep,ek+jkd
{
     POV[j+nsd]=0;                            //   PZ(j)=0.
for (int i=op; i<=ok; i++)          //   do i=op,ok
{
  POV[j+nsd]= POV[j+nsd]+ POL[j-i+nsd]*POG1[i+nsd];    //   PZ(j)=PZ(j)+PR(j-i)*PO(i);
}
}
for (int i=0; i<=m1ez_teor-(op+ok); i++) PO[i]=POV[i+ok]; //   Зсув КДВ до початку інформативної області
for (int i=0; i<=m1ez; i++)  PO[i]=PO[i]-0.999*PO[m1ez];
}

// Memo9->Lines->Add( "aaa3 пройшло");

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit152->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][1]=POk[i];
m1_[11]=m1ez;
m10_[11]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series2->AddXY(DeltaTeta1,POk[i],"",clWhite);
//Memo8->Lines->Add(FloatToStr(POk[i]));
}
//Memo9->Lines->Add( "aaa4 пройшло");
}

if (CheckBox43->Checked==true && CheckBox60->Checked==true)        // (888)
{
ik=ik_[2];
eta=StrToFloat(Edit299->Text);
w=StrToFloat(Edit227->Text);
A=StrToFloat(Edit226->Text);
Ymin=StrToFloat(Edit225->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit228->Text=FloatToStr(Xmin);
Edit229->Text=IntToStr(m1ez+1);
param_AF[0][2]=Ymin;
param_AF[1][2]=A;
param_AF[2][2]=w;
param_AF[3][2]=Xmin;
param_AF[4][2]=m1ez+1;
param_AF[5][2]=eta;
y0L=-(0 + (2*A*w/M_PI)/(4*(Xmin-xc)*(Xmin-xc)+w*w));     // =0; !!!!!!!!!!!!!!!
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
POG[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
POPV[i]=eta*POG[i]+(1-eta)*POL[i];
//Memo8->Lines->Add(FloatToStr(POG[i])+'\t'+FloatToStr(POL[i])+'\t'+FloatToStr(POPV[i])+'\t'+FloatToStr(1111) );
}

if (eta==1)
  for (int i=0; i<=m1ez; i++) PO[i]=POG[i];
if (eta==0)
  for (int i=0; i<=m1ez; i++) PO[i]=POL[i];
//for (int i=0; i<=m1ez; i++)  PO[i]=POL[i]-0.999*POL[m1ez];
if (eta>0 && eta<1)
  for (int i=0; i<=m1ez; i++) PO[i]=POPV[i];
if (eta==2)
{
 ep=-m10ez;   //-m10;
 ek=m1ez-m10ez;     ///m1-m10;
 op=-m10ez;    //-m10z;      // АФ вся уточнена при розрахунку
 ok=m1ez-m10ez;     //m1z-m10z;   // АФ вся уточнена при розрахунку
 jp=ep-ok;
 jk=ek-op;
m10ez_teor=-jp;        // =m10+(m1z-m10z)
m1ez_teor=-jp+jk;      // =m10+(m1z-m10z)+m1-m10-(-m10z)=m1+m1z      =260

TetaMin=-(m10ez_teor)*ik;
for (int i=0; i<=m1ez_teor; i++)
{
x=(TetaMin+i*ik);
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
}
 jp=-m10ez_teor;   //-m10_teor;             // 260/275
 jk=m1ez_teor-m10ez_teor;   //m1_teor-m10_teor;
 ep=jp+ok;     //бо АФ вся уточнена
 ek=jk+op;
 nsd=-jp;

for (int i=m1ez; i>=0; i--) POG1[i+nsd-m10ez]=POG[i];
for (int j=ep; j<=ek; j++)                     //   do 13 j=ep,ek+jkd
{
     POV[j+nsd]=0;                            //   PZ(j)=0.
for (int i=op; i<=ok; i++)          //   do i=op,ok
{
  POV[j+nsd]= POV[j+nsd]+ POL[j-i+nsd]*POG1[i+nsd];    //   PZ(j)=PZ(j)+PR(j-i)*PO(i);
}
}
for (int i=0; i<=m1ez_teor-(op+ok); i++) PO[i]=POV[i+ok]; //   Зсув КДВ до початку інформативної області
for (int i=0; i<=m1ez; i++)  PO[i]=PO[i]-0.999*PO[m1ez];
}

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit385->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][2]=POk[i];
m1_[12]=m1ez;
m10_[12]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series51->AddXY(DeltaTeta1,POk[i],"",clWhite);
}
}

if (CheckBox44->Checked==true && CheckBox71->Checked==true)     // (880)
{
ik=ik_[3];
eta=StrToFloat(Edit304->Text);
w=StrToFloat(Edit232->Text);
A=StrToFloat(Edit231->Text);
Ymin=StrToFloat(Edit230->Text);
y0=0.;
xc=0.;
Xmin=xc+sqrt(-w*w/(2*log(4.))*log((Ymin-y0)/(A/w)/sqrt(2*log(4.)/M_PI)));
m10ez=int(Xmin/ik);
m1ez=m10ez*2;
Edit233->Text=FloatToStr(Xmin);
Edit234->Text=IntToStr(m1ez+1);
param_AF[0][3]=Ymin;
param_AF[1][3]=A;
param_AF[2][3]=w;
param_AF[3][3]=Xmin;
param_AF[4][3]=m1ez+1;
param_AF[5][3]=eta;
y0L=-(0 + (2*A*w/M_PI)/(4*(Xmin-xc)*(Xmin-xc)+w*w));     // =0; !!!!!!!!!!!!!!!
TetaMin=-(m10ez)*ik;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik);
POG[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
POPV[i]=eta*POG[i]+(1-eta)*POL[i];
//Memo8->Lines->Add(FloatToStr(POG[i])+'\t'+FloatToStr(POL[i])+'\t'+FloatToStr(POPV[i])+'\t'+FloatToStr(1111) );
}

if (eta==1)
  for (int i=0; i<=m1ez; i++) PO[i]=POG[i];
if (eta==0)
  for (int i=0; i<=m1ez; i++) PO[i]=POL[i];
//for (int i=0; i<=m1ez; i++)  PO[i]=POL[i]-0.999*POL[m1ez];
if (eta>0 && eta<1)
  for (int i=0; i<=m1ez; i++) PO[i]=POPV[i];
if (eta==2)
{
 ep=-m10ez;   //-m10;
 ek=m1ez-m10ez;     ///m1-m10;
 op=-m10ez;    //-m10z;      // АФ вся уточнена при розрахунку
 ok=m1ez-m10ez;     //m1z-m10z;   // АФ вся уточнена при розрахунку
 jp=ep-ok;
 jk=ek-op;
m10ez_teor=-jp;        // =m10+(m1z-m10z)
m1ez_teor=-jp+jk;      // =m10+(m1z-m10z)+m1-m10-(-m10z)=m1+m1z      =260

TetaMin=-(m10ez_teor)*ik;
for (int i=0; i<=m1ez_teor; i++)
{
x=(TetaMin+i*ik);
POL[i]=y0L + (2*A*w/M_PI)/(4*(x-xc)*(x-xc)+w*w);
}
 jp=-m10ez_teor;   //-m10_teor;             // 260/275
 jk=m1ez_teor-m10ez_teor;   //m1_teor-m10_teor;
 ep=jp+ok;     //бо АФ вся уточнена
 ek=jk+op;
 nsd=-jp;

for (int i=m1ez; i>=0; i--) POG1[i+nsd-m10ez]=POG[i];
for (int j=ep; j<=ek; j++)                     //   do 13 j=ep,ek+jkd
{
     POV[j+nsd]=0;                            //   PZ(j)=0.
for (int i=op; i<=ok; i++)          //   do i=op,ok
{
  POV[j+nsd]= POV[j+nsd]+ POL[j-i+nsd]*POG1[i+nsd];    //   PZ(j)=PZ(j)+PR(j-i)*PO(i);
}
}
for (int i=0; i<=m1ez_teor-(op+ok); i++) PO[i]=POV[i+ok]; //   Зсув КДВ до початку інформативної області
for (int i=0; i<=m1ez; i++)  PO[i]=PO[i]-0.999*PO[m1ez];
}

if (CheckBox51->Checked==true) // Уточнення ділянки КДВ
{
koef_dTeta=StrToInt(Edit391->Text);
m10ez=m10ez*koef_dTeta;
m1ez=m10ez*2;
TetaMin=-(m10ez)*ik/koef_dTeta;
for (int i=0; i<=m1ez; i++)
{
x=(TetaMin+i*ik/koef_dTeta);
PO[i]=y0 + (A/w*sqrt(2*log(4.)/M_PI)*exp(-2*(x-xc)*(x-xc)*log(4.)/w/w));
}
}
Ap_sum=0.;
for (int i=0; i<=m1ez; i++)  Ap_sum=Ap_sum+PO[i];
for (int i=0; i<=m1ez; i++)  POk[i]=PO[i]/Ap_sum;
for (int i=0; i<=m1ez; i++) POk2d[i][3]=POk[i];
m1_[13]=m1ez;
m10_[13]=m10ez;
TetaMin=-(m10ez)*ik;
for (int i=0;i<=m1ez;i++)
{
DeltaTeta1=(TetaMin+i*ik);
Series53->AddXY(DeltaTeta1,POk[i],"",clWhite);
}
}

delete  DeltaTetaAF, PO, POk;
delete POG,POG1, POL, POPV,POV;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button5Click(TObject *Sender) //Очистити профіль
{
for (int i=0; i<=km; i++)  DD[i]=0;
Series5->Clear();
Series30->Clear();
Series31->Clear();
Series33->Clear();
Series27->Clear();
Series28->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button2Click(TObject *Sender) //Очистити диф. КДВ (справа)
{
Series18->Clear();
Series19->Clear();
Series20->Clear();
Series21->Clear();
/*Series22->Clear();
Series23->Clear();
Series27->Clear();
Series28->Clear();
Series29->Clear();   */
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button7Click(TObject *Sender)  // Очистити когер. КДВ (справа)
{
Series8->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button6Click(TObject *Sender) // Очистити КДВ та АФ (справа)
{
Series2->Clear();
Series51->Clear();
Series53->Clear();
//Series24->Clear();
//Series25->Clear();
//Series26->Clear();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Button16Click(TObject *Sender) // очистити АФ
{
if (RadioButton10->Checked==true)
{
Edit162->Clear();
Edit163->Clear();
for (int i=0; i<=m1_[11]; i++) POk2d[i][1]=0;
Series2->Clear();
CheckBox59->Checked=false;
}
if (RadioButton20->Checked==true)
{
Edit228->Clear();
Edit229->Clear();
for (int i=0; i<=m1_[12]; i++) POk2d[i][2]=0;
Series51->Clear();
CheckBox60->Checked=false;
}
if (RadioButton21->Checked==true)
{
Edit233->Clear();
Edit234->Clear();
for (int i=0; i<=m1_[13]; i++) POk2d[i][3]=0;
Series53->Clear();
CheckBox71->Checked=false;
}
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button4Click(TObject *Sender) // Очистити експер. КДВ (центр)
{
if (CheckBox42->Checked==false && CheckBox43->Checked==false && CheckBox44->Checked==false)
vved_exper=0;
if (RadioButton10->Checked==true)
{
for (int i=0; i<=m1_[1]; i++) intIk2d[i][1]=0;
if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
else Series11->Clear();
}
if (RadioButton20->Checked==true)
{
for (int i=0; i<=m1_[2]; i++) intIk2d[i][2]=0;
if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
else Series1->Clear();
}
if (RadioButton21->Checked==true)
{
for (int i=0; i<=m1_[3]; i++) intIk2d[i][3]=0;
if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
else Series45->Clear();
}
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button8Click(TObject *Sender)  // Очистити диф. КДВ (центр)
{
Series9->Clear();
Series15->Clear();
Series46->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button9Click(TObject *Sender)  // Очистити когер. КДВ (центр)
{
Series4->Clear();
Series13->Clear();
Series47->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button10Click(TObject *Sender)  // Очистити повну КДВ (центр)
{
Series6->Clear();
Series14->Clear();
Series48->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button11Click(TObject *Sender)  // Очистити згортку КДВ (центр)
{
Series10->Clear();
Series12->Clear();
Series49->Clear();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button12Click(TObject *Sender) // Коеф. на експ. КДВ 1
{
//number_KDV=StrToInt(Edit133->Text);   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
int m1e,m10e;
double ik,ik_new,TetaMin,DeltaTeta1; // intIk[MM],intIk_[MM],intIktmp[MM],  //!!!!!!!!!!!!!!!!!!!!!!!!!
  double *intIk, *intIk_, *intIktmp;
  intIk    = new double[MM];
  intIk_   = new double[MM];
  intIktmp = new double[MM];


if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
if (number_KDV==2 || number_KDV==3)
{
//Series1->Clear();
Series11->Clear();
Series24->Clear();
}
ik=ik_[1];
m1e=m1_[1];
m10e=m10_[1];
for (int i=0;i<=m1e;i++) intIk[i]=intI02d[i][1];
ekspk0=StrToFloat(Edit165->Text);
ekspk=StrToFloat(LabeledEdit1->Text);

for (int i=0; i<=m1e; i++)
{
 intIk[i]=intIk[i]-ekspk0;
if (intIk[i]<=0) intIk[i]=0.001*(ekspk0+1);
}
double PEmax=0;
for (int i=0; i<=m1e; i++) if (intIk[i]>PEmax) PEmax=intIk[i];
for (int i=0; i<=m1e; i++) intIk[i]=intIk[i]/PEmax*ekspk;
for (int i=0; i<=m1e; i++) intIk2d[i][1]=intIk[i];

if (CheckBox54->Checked==true)   // Уточнення кроку
{
ik_new=StrToFloat(Edit213->Text);
int zsuv=int(0.5+m10e*(ik_new-ik)/ik_new);
int m1e_ukor=m1e-int(0.5+m1e*(ik_new-ik)/ik_new);
for (int i=0; i<=m1e_ukor; i++)intIk_[i]=intIk[i+int(0.5+(ik_new*i-ik*i)/ik)]+ (intIk[i+1+int(0.5+(ik_new*i-ik*i)/ik)]-intIk[i+int(0.5+(ik_new*i-ik*i)/ik)])*
(ik_new*i-ik*(i+(ik_new*i-ik*i)/ik));

for (int i=0; i<=m1e; i++)
{
if (i<zsuv) intIktmp[i]=1e-6;
if (i>=zsuv && i<=zsuv+m1e_ukor) intIktmp[i]=intIk_[i-zsuv];
if (i>zsuv+m1e_ukor) intIktmp[i]=2e-6;
}
//for (int i=0; i<=m1e; i++)Memo8->Lines->Add(FloatToStr(intIk2d[i][1])+'\t'+FloatToStr(intIk_[i])+'\t'+FloatToStr(intIktmp[i])+'\t'+FloatToStr(m1e_ukor)+'\t'+FloatToStr(zsuv)+'\t'+FloatToStr(ik)+'\t'+FloatToStr(ik_new));
for (int i=0; i<=m1e; i++) intIk[i]=intIktmp[i];
for (int i=0; i<=m1e; i++) intIk2d[i][1]=intIk[i];
}

TetaMin=-(m10e)*ik;
for (int i=0;i<=m1e;i++)
{
DeltaTeta1=(TetaMin+i*ik);

if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
}
  delete intIk, intIk_, intIktmp;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Button28Click(TObject *Sender) // Коеф. на експ. КДВ 2
{
//number_KDV=StrToInt(Edit133->Text);   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
int m1e,m10e;
double ik, ik_new,TetaMin,DeltaTeta1;   //intIk[MM],intIk_[MM],intIktmp[MM], //!!!!!!!!!!!!!!!!!!!!!!!!!
  double *intIk, *intIk_, *intIktmp;
  intIk    = new double[MM];
  intIk_   = new double[MM];
  intIktmp = new double[MM];
if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
if (number_KDV==2 || number_KDV==3)
{
Series1->Clear();
//Series11->Clear();
Series25->Clear();
}
ik=ik_[2];
m1e=m1_[2];
m10e=m10_[2];
for (int i=0;i<=m1e;i++) intIk[i]=intI02d[i][2];
ekspk0=StrToFloat(Edit75->Text);
ekspk=StrToFloat(LabeledEdit2->Text);

for (int i=0; i<=m1e; i++)
{
 intIk[i]=intIk[i]-ekspk0;
if (intIk[i]<=0) intIk[i]=0.001*(ekspk0+1);
}
double PEmax=0;
for (int i=0; i<=m1e; i++) if (intIk[i]>PEmax) PEmax=intIk[i];
for (int i=0; i<=m1e; i++) intIk[i]=intIk[i]/PEmax*ekspk;
for (int i=0; i<=m1e; i++) intIk2d[i][2]=intIk[i];

if (CheckBox54->Checked==true)   // Уточнення кроку
{
 ik_new=StrToFloat(Edit214->Text);
int zsuv=int(0.5+m10e*(ik_new-ik)/ik_new);
int m1e_ukor=m1e-int(0.5+m1e*(ik_new-ik)/ik_new);
for (int i=0; i<=m1e_ukor; i++)intIk_[i]=intIk[i+int(0.5+(ik_new*i-ik*i)/ik)]+ (intIk[i+1+int(0.5+(ik_new*i-ik*i)/ik)]-intIk[i+int(0.5+(ik_new*i-ik*i)/ik)])*
(ik_new*i-ik*(i+(ik_new*i-ik*i)/ik));

for (int i=0; i<=m1e; i++)
{
if (i<zsuv) intIktmp[i]=1e-6;
if (i>=zsuv && i<=zsuv+m1e_ukor) intIktmp[i]=intIk_[i-zsuv];
if (i>zsuv+m1e_ukor) intIktmp[i]=2e-6;
}
//for (int i=0; i<=m1e; i++)Memo8->Lines->Add(FloatToStr(intIk2d[i][1])+'\t'+FloatToStr(intIk[i])+'\t'+FloatToStr(intIktmp[i])+'\t'+FloatToStr(m1e_ukor)+'\t'+FloatToStr(zsuv)+'\t'+FloatToStr(ik)+'\t'+FloatToStr(ik_new));
for (int i=0; i<=m1e; i++) intIk[i]=intIktmp[i];
for (int i=0; i<=m1e; i++) intIk2d[i][2]=intIk[i];
}


TetaMin=-(m10e)*ik;
for (int i=0;i<=m1e;i++)
{
DeltaTeta1=(TetaMin+i*ik);
if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series25->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
}
  delete intIk, intIk_, intIktmp;
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Button29Click(TObject *Sender) // Коеф. на експ. КДВ 3
{
//number_KDV=StrToInt(Edit133->Text);   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
int m1e,m10e;
double ik,ik_new,TetaMin,DeltaTeta1;   //intIk[MM],intIk_[MM],intIktmp[MM], //!!!!!!!!!!!!!!!!!!!!!!!!!
  double *intIk, *intIk_, *intIktmp;
  intIk    = new double[MM];
  intIk_   = new double[MM];
  intIktmp = new double[MM];
if (number_KDV==1)
{
Series1->Clear();
Series11->Clear();
Series24->Clear();
}
if (number_KDV==2 || number_KDV==3)
{
//Series1->Clear();
Series45->Clear();
Series26->Clear();
}
ik=ik_[3];
m1e=m1_[3];
m10e=m10_[3];
for (int i=0;i<=m1e;i++) intIk[i]=intI02d[i][3];
ekspk0=StrToFloat(Edit135->Text);
ekspk=StrToFloat(LabeledEdit3->Text);

for (int i=0; i<=m1e; i++)
{
 intIk[i]=intIk[i]-ekspk0;
if (intIk[i]<=0) intIk[i]=0.001*(ekspk0+1);
}
double PEmax=0;
for (int i=0; i<=m1e; i++) if (intIk[i]>PEmax) PEmax=intIk[i];
for (int i=0; i<=m1e; i++) intIk[i]=intIk[i]/PEmax*ekspk;
for (int i=0; i<=m1e; i++) intIk2d[i][3]=intIk[i];

if (CheckBox54->Checked==true)   // Уточнення кроку
{
 ik_new=StrToFloat(Edit215->Text);
int zsuv=int(0.5+m10e*(ik_new-ik)/ik_new);
int m1e_ukor=m1e-int(0.5+m1e*(ik_new-ik)/ik_new);
for (int i=0; i<=m1e_ukor; i++)intIk_[i]=intIk[i+int(0.5+(ik_new*i-ik*i)/ik)]+ (intIk[i+1+int(0.5+(ik_new*i-ik*i)/ik)]-intIk[i+int(0.5+(ik_new*i-ik*i)/ik)])*
(ik_new*i-ik*(i+(ik_new*i-ik*i)/ik));

for (int i=0; i<=m1e; i++)
{
if (i<zsuv) intIktmp[i]=1e-6;
if (i>=zsuv && i<=zsuv+m1e_ukor) intIktmp[i]=intIk_[i-zsuv];
if (i>zsuv+m1e_ukor) intIktmp[i]=2e-6;
}
//for (int i=0; i<=m1e; i++)Memo8->Lines->Add(FloatToStr(intIk2d[i][1])+'\t'+FloatToStr(intIk[i])+'\t'+FloatToStr(intIktmp[i])+'\t'+FloatToStr(m1e_ukor)+'\t'+FloatToStr(zsuv)+'\t'+FloatToStr(ik)+'\t'+FloatToStr(ik_new));
for (int i=0; i<=m1e; i++) intIk[i]=intIktmp[i];
for (int i=0; i<=m1e; i++) intIk2d[i][3]=intIk[i];
}

TetaMin=-(m10e)*ik;
for (int i=0;i<=m1e;i++)
{
DeltaTeta1=(TetaMin+i*ik);
if (number_KDV==1)
{
Series11->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series1->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series45->AddXY(DeltaTeta1,intIk[i],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk[i],"",clGreen);
}
}
  delete intIk, intIk_, intIktmp;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void __fastcall TForm1::rez1Click(TObject *Sender) // Відкрити файл *.rez
{
  if (OtkEt->Execute())
  {
vved_exper=2;                //що  вводяться  експ. КДВ з *.rez
// Очищуємо дані по замовчуванню про дефекти:
CheckBox1->Checked=false;
CheckBox58->Checked=false;
CheckBox2->Checked=false;
CheckBox26->Checked=false;
CheckBox4->Checked=false;

CheckBox32->Checked=false;
CheckBox33->Checked=false;
CheckBox34->Checked=false;
CheckBox35->Checked=false;
CheckBox36->Checked=false;
CheckBox37->Checked=false;

CheckBox12->Checked=false;
CheckBox15->Checked=false;
CheckBox13->Checked=false;
CheckBox17->Checked=false;
CheckBox55->Checked=false;
CheckBox16->Checked=false;
CheckBox14->Checked=false;


//    Memo2->Clear();
//   Memo2->Lines->Add("# " + OtkEt->FileName);
//    Memo2->Lines->Add("");

    // Визначаємо розмір файлу
    unsigned long file_size = fsize( OtkEt->FileName.c_str() );
    // Виділяємо пам`ять під вміст файлу
    char *data = new char[ file_size + 1 ];
    // Читаємо увесьфайл у змінну
    fread_all( OtkEt->FileName.c_str(), data );
//    Memo2->Lines->Add( data);
// Form2->Memo1-> Lines->Add(data);
    // Визначаємо кількість рядків
    unsigned long lines_count = count_substr( data, "\n", 0);
    long offset,xx,xx_,xx2,xx_line,xx_line1,xx_line2;

	// !!!! find_first=-4 якщо співпадінь не знайдено !!!
    xx =  find_first ( data, "Монокристал",0);
if (xx!=-4) 
  {
  CheckBox68->Checked=true; //else CheckBox68->Checked=false;
  if (xx>0) offset=xx;
  xx =  find_first ( data, "Хі - GGG Кл. (моно.)",0);
  if (xx!=-4) RadioButton38->Checked=true; 
  }
  
  xx =  find_first ( data, "Гетероструктура",0);
if (xx==-4)
  CheckBox31->Checked=false;
else
  {
  CheckBox31->Checked=true;
  if (xx>0) offset=xx;
  xx =  find_first ( data, "Хі pl - YIG Кл. (моно.)",0);
  if (xx!=-4) RadioButton46->Checked=true; 
  xx =  find_first ( data, "Хі pl - S-4-x (моно.)",0);
  if (xx!=-4) RadioButton52->Checked=true; 
  xx =  find_first ( data, "Хі pl - S-5-x (моно.)",0);
  if (xx!=-4) RadioButton51->Checked=true; 

  xx_line2 = find_line  ( data, "Гетероструктура");
    rectangle data_range3;
    data_range3.top = xx_line2+2;
    data_range3.bottom = lines_count - data_range3.top -1; // 1 - кількість рядків із даними
    data_range3.left = 12;
    data_range3.right = 17;
    J_get_narray master3( OtkEt->FileName.c_str(), "\t", data_range3);
    if( master3.get_err_code() != ALL_OK ){ return; }
  apl=master3.get_element(0,0);
  Edit167->Text=FloatToStr(apl);
  int zs=0;
  hpll:
    rectangle data_range4;
    data_range4.top = xx_line2+2;
    data_range4.bottom = lines_count - data_range4.top -1; // 1 - кількість рядків із даними
    data_range4.left = 35+zs;
    data_range4.right = 0;
    J_get_narray master4( OtkEt->FileName.c_str(), "\t", data_range4);
    if( master4.get_err_code() != ALL_OK ){ return; }
  hpl=master4.get_element(0,0);
  Edit166->Text=FloatToStr(hpl);
  if (StrToFloat(Edit166->Text)<0.001) {zs=1; goto hpll;}
  }
if (xx>0) offset=xx;
  
    xx =  find_first ( data, "Приповерхневий порушений шар",0);
if (xx!=-4) CheckBox67->Checked=true; else CheckBox67->Checked=false;
if (xx>0) offset=xx;
Memo9->Lines->Add( FloatToStr(xx)+'\t'+FloatToStr(offset));

    xx =  find_first ( data, "Omega сканування",offset-1);
if (xx!=-4)RadioButton5->Checked=true;  else RadioButton5->Checked=false;
if (xx>0) offset=xx;
    xx =  find_first ( data, "Teta-2teta сканування",offset-1);
if (xx!=-4)RadioButton6->Checked=true;  else RadioButton6->Checked=false;
if (xx>0) offset=xx;
Memo9->Lines->Add( FloatToStr(xx)+'\t'+FloatToStr(offset));

/*    xx =  find_first ( data, "Sigma + Pi поляризація",offset-1);
if (xx!=-4)RadioButton2->Checked=true;  else RadioButton1->Checked=true;
if (xx>0) offset=xx;
Memo9->Lines->Add( FloatToStr(xx)+'\t'+FloatToStr(offset));   */

    xx_ =  find_first ( data, "Sigma + Pi поляризація",offset-1);
if (xx_!=-4)RadioButton2->Checked=true;
    xx =  find_first ( data, "Монохроматор Si(111)/Ge(111)",offset-1);
if (xx!=-4){RadioButton2->Checked=false; RadioButton56->Checked=true;}
if (xx_==-4)
  {
    xx =  find_first ( data, "Sigma поляризація",offset-1);
  if (xx!=-4)RadioButton1->Checked=true; else RadioButton55->Checked=true;
  }
Memo9->Lines->Add( FloatToStr(xx_)+'\t'+FloatToStr(xx)+'\t'+FloatToStr(offset));

    xx =  find_first ( data, "Моделювання ГБП за р-нями ТТ",offset-1);
if (xx!=-4)RadioButton11->Checked=true;
if (xx>0) offset=xx;
Memo9->Lines->Add( FloatToStr(xx)+'\t'+FloatToStr(offset));
    xx =  find_first ( data, "Моделювання ГБП за р-нями УДТ",offset-1);
if (xx!=-4)RadioButton13->Checked=true;
if (xx>0) offset=xx;
    xx =  find_first ( data, "Моделювання ГБП за р-нями ТТ+кінем.ПШ",offset-1);
if (xx!=-4)RadioButton12->Checked=true;
if (xx>0) offset=xx;
    xx =  find_first ( data, "Моделювання ГБП за р-нями ТТ+кінем.ПШ+розворот",offset-1);
if (xx!=-4)RadioButton16->Checked=true;
if (xx>0) offset=xx;
    xx =  find_first ( data, "Моделювання ГБП за р-нями УДТ+кінем.ПШ(Мол.)",offset-1);
if (xx!=-4)RadioButton33->Checked=true;
if (xx>0) offset=xx;

xx_line =find_line_another ( data, "Кількість КДВ:",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line+0;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 15;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
number_KDV=master2.get_element(0,0);      
Edit146->Text=FloatToStr(number_KDV);    // Кількість КДВ



// КДВ (444)        Парам. апаратної функції:
xx=find_first ( data, " КДВ (444)",offset-1);;
if (xx!=-4)
{
CheckBox42->Checked=true;
offset=xx;
xx_line =  find_line_another ( data, " КДВ (444)",offset-1);
xx_line2=find_line_another ( data, "Апаратна функція з файла",offset-1);
if (xx_line+1==xx_line2)
  {
  //MessageBox(0,"Апаратна функція для (444) з файла","Увага!", MB_OK + MB_ICONEXCLAMATION);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
  CheckBox59->Checked=true;
  LabeledEdit1->Text=FloatToStr(master2.get_element(0,0));
  Edit165->Text=FloatToStr(master2.get_element(0,1));
  }
xx_line2=find_line_another ( data, "Парам. апаратної функції:",offset-1);
//Memo9->Lines->Add( "Парам. апаратної функції 444:");
//Memo9->Lines->Add( FloatToStr(xx_line)+'\t'+FloatToStr(xx_line2)+'\t'+FloatToStr(offset));
if (xx_line+1==xx_line2)
  {
    rectangle data_range;
    data_range.top = xx_line2+2;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
  CheckBox59->Checked=true;
  Edit159->Text=FloatToStr(master.get_element(0,0));
  Edit160->Text=FloatToStr(master.get_element(0,1));
  Edit161->Text=FloatToStr(master.get_element(0,2));
  if (master.get_columns()==6)Edit289->Text=FloatToStr(master.get_element(0,5));
    rectangle data_range2;
    data_range2.top = xx_line2+3;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
  LabeledEdit1->Text=FloatToStr(master2.get_element(0,0));
  Edit165->Text=FloatToStr(master2.get_element(0,1));
  }
} else CheckBox42->Checked=false;
/*
xx_line =  find_line ( data, "Sigma");                      //4
xx_line2 =  find_line_another ( data, " КДВ (999)",0);      //664    663
offset =  find_first ( data, "кристал",0);                  //5
xx=count_substr(data, "КДВ (999)", 0);                      //0
Memo9->Lines->Add( FloatToStr(xx_line)+'\t'+FloatToStr(xx_line2)+'\t'+FloatToStr(offset)+'\t'+FloatToStr(xx));
    */

// КДВ (888)        Парам. апаратної функції:
xx=find_first ( data, " КДВ (888)",offset-1);;
if (xx!=-4)
{
CheckBox43->Checked=true;
offset=xx;
xx_line =find_line_another ( data, " КДВ (888)",offset-1);
xx_line2=find_line_another ( data, "Апаратна функція з файла",offset-1);
if (xx_line+1==xx_line2)
  {
  //if (xx_line+1==xx_line2) MessageBox(0,"Апаратна функція для (888) з файла","Увага!", MB_OK + MB_ICONEXCLAMATION);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
  CheckBox60->Checked=true;
  LabeledEdit2->Text=FloatToStr(master2.get_element(0,0));
  Edit75->Text=FloatToStr(master2.get_element(0,1));
  }
xx_line2=find_line_another ( data, "Парам. апаратної функції:",offset-1);
//Memo9->Lines->Add( "Парам. апаратної функції 888:");
//Memo9->Lines->Add( FloatToStr(xx_line)+'\t'+FloatToStr(xx_line2)+'\t'+FloatToStr(offset));
if (xx_line+1==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+2;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
CheckBox60->Checked=true;
Edit225->Text=FloatToStr(master.get_element(0,0));
Edit226->Text=FloatToStr(master.get_element(0,1));
Edit227->Text=FloatToStr(master.get_element(0,2));
if (master.get_columns()==6) Edit299->Text=FloatToStr(master.get_element(0,5));
    rectangle data_range2;
    data_range2.top = xx_line2+3;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
LabeledEdit2->Text=FloatToStr(master2.get_element(0,0));
Edit75->Text=FloatToStr(master2.get_element(0,1));
}
} else CheckBox43->Checked=false;

// КДВ (880)        Парам. апаратної функції:
xx=find_first ( data, " КДВ (880)",offset-1);;
if (xx!=-4)
{
CheckBox44->Checked=true;
offset=xx;
xx_line =find_line_another ( data, " КДВ (880)",offset-1);
xx_line2=find_line_another ( data, "Апаратна функція з файла",offset-1);
if (xx_line+1==xx_line2)
  {
  //if (xx_line+1==xx_line2) MessageBox(0,"Апаратна функція для (880) з файла","Увага!", MB_OK + MB_ICONEXCLAMATION);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
  CheckBox71->Checked=true;
  LabeledEdit3->Text=FloatToStr(master2.get_element(0,0));
  Edit135->Text=FloatToStr(master2.get_element(0,1));
  }
xx_line2=find_line_another ( data, "Парам. апаратної функції:",offset-1);
Memo9->Lines->Add( "Парам. апаратної функції 880:");
Memo9->Lines->Add( FloatToStr(xx_line)+'\t'+FloatToStr(xx_line2)+'\t'+FloatToStr(offset));
if (xx_line+1==xx_line2)
  {
    rectangle data_range;
    data_range.top = xx_line2+2;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
CheckBox71->Checked=true;
Memo9->Lines->Add( FloatToStr(data_range.top)+'\t'+FloatToStr(data_range.bottom)+'\t'+FloatToStr(offset));
Edit230->Text=FloatToStr(master.get_element(0,0));
Edit231->Text=FloatToStr(master.get_element(0,1));
Edit232->Text=FloatToStr(master.get_element(0,2));
if (master.get_columns()==6) Edit304->Text=FloatToStr(master.get_element(0,5));
    rectangle data_range2;
    data_range2.top = xx_line2+3;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 23;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
LabeledEdit3->Text=FloatToStr(master2.get_element(0,0));
Edit135->Text=FloatToStr(master2.get_element(0,1));
  }
} else CheckBox44->Checked=false;

xx = find_first ( data, "Дефекти у ідеальній частині монокристалу",offset-1);
if (xx!=-4) offset=xx;
if (find_first  ( data, "Дефекти у ідеальній частині монокристалу не враховуються",offset-1)>=0)
  {
  CheckBox13->Checked=false; CheckBox17->Checked=false; CheckBox12->Checked=false;
  CheckBox15->Checked=false; CheckBox16->Checked==false; CheckBox14->Checked==false;
  }
else
  {
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі",offset-1)>=0)
  {
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/sqrt(2)",offset-1)>=0) RadioButton7->Checked=true;
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a",offset-1)>=0)         RadioButton8->Checked=true;
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a*sqrt(2)",offset-1)>=0) RadioButton9->Checked=true;
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)",offset-1)>=0) RadioButton28->Checked=true;
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)",offset-1)>=0) RadioButton32->Checked=true;
  if (find_first  ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі з вект. Бюргерса b=a*sqrt(3)",offset-1)>=0) RadioButton29->Checked=true;
  xx_line =find_line_another ( data, "Модель дефектів в ід. част. монокр. - дисл. петлі",offset-1);
  xx_line2=find_line_another ( data, "Коефіцієнт на L:",offset-1);
  if (xx_line+1==xx_line2)
    {
    rectangle data_range2;
    data_range2.top = xx_line+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 18;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
    Edit149->Text=FloatToStr(master2.get_element(0,0));    // Коефіцієнт на L
    }
  xx_line2=find_line_another ( data, "Конц.петель:",offset-1);
  Memo9->Lines->Add("LLL");
  if (xx_line+2==xx_line2)
    {
    Memo9->Lines->Add( "kon");
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -2; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
    Memo9->Lines->Add( FloatToStr(master.get_element(0,0))+'\t'+FloatToStr(master.get_element(0,1))+'\t'+FloatToStr(master.get_element(1,0))+'\t'+FloatToStr(master.get_element(1,1)));
    Edit53->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
    Edit54->Text=FloatToStr(master.get_element(0,1));
    Edit64->Text=FloatToStr(master.get_element(1,0));
    Edit65->Text=FloatToStr(master.get_element(1,1));
    CheckBox13->Checked=true; CheckBox17->Checked=true;
    if (StrToFloat(Edit64->Text)<0.001 && StrToFloat(Edit65->Text)<0.001) CheckBox17->Checked=false;
    }
  }
if (find_first  ( data, "Модель дефектів в ід. част. монокр. - сферичні кластери",offset-1)>=0)
{
xx_line =find_line_another ( data, "Модель дефектів в ід. част. монокр. - сферичні кластери",offset-1);
xx_line2=find_line_another ( data, "Конц.сф.кластерів:",offset-1);
if (xx_line+2==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -2; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Edit50->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
Edit51->Text=FloatToStr(master.get_element(0,1));
Edit46->Text=FloatToStr(master.get_element(1,0));
Edit47->Text=FloatToStr(master.get_element(1,1));
CheckBox12->Checked=true; CheckBox15->Checked=true;
if (StrToFloat(Edit46->Text)<0.001 && StrToFloat(Edit47->Text)<0.001) CheckBox15->Checked=false;
}
}
 // !!!!!!!!!!!   +інші типи дефектів
}


xx = find_first ( data, "Дефекти у ідеальній частині плівки:",offset-1);
if (xx!=-4) offset=xx;
if (find_first  ( data, "Дефекти у ідеальній частині плівки не враховуються",offset-1)>=0)
{
CheckBox32->Checked=false; CheckBox33->Checked=false; CheckBox34->Checked=false;
CheckBox35->Checked=false; CheckBox36->Checked==false; CheckBox37->Checked==false;
}
else
{
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі",offset-1)>=0)
{
/*if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/sqrt(2)",offset-1)>=0) RadioButton7->Checked=true;
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a",offset-1)>=0)         RadioButton8->Checked=true;
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a*sqrt(2)",offset-1)>=0) RadioButton9->Checked=true;
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)",offset-1)>=0) RadioButton28->Checked=true;
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)",offset-1)>=0) RadioButton32->Checked=true;
if (find_first  ( data, "Модель дефектів в ід. част. плівки - дисл. петлі з вект. Бюргерса b=a*sqrt(3)",offset-1)>=0) RadioButton29->Checked=true;
*/
xx_line =find_line_another ( data, "Модель дефектів в ід. част. плівки - дисл. петлі",offset-1);
/*xx_line2=find_line_another ( data, "Коефіцієнт на L:",offset-1);
if (xx_line+1==xx_line2)
{
    rectangle data_range2;
    data_range2.top = xx_line+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 18;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
Edit149->Text=FloatToStr(master2.get_element(0,0));    // Коефіцієнт на L
}      */
xx_line2=find_line_another ( data, "Конц.петель:",offset-1);
Memo9->Lines->Add("kon pl");
if (xx_line+2==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -2; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Memo9->Lines->Add( FloatToStr(master.get_element(0,0))+'\t'+FloatToStr(master.get_element(0,1))+'\t'+FloatToStr(master.get_element(1,0))+'\t'+FloatToStr(master.get_element(1,1)));
Edit174->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
Edit175->Text=FloatToStr(master.get_element(0,1));
Edit176->Text=FloatToStr(master.get_element(1,0));
Edit177->Text=FloatToStr(master.get_element(1,1));
Memo9->Lines->Add("kon pl end");
CheckBox34->Checked=true; CheckBox35->Checked=true;
if (StrToFloat(Edit176->Text)<0.001 && StrToFloat(Edit177->Text)<0.001) CheckBox35->Checked=false;
}
}

if (find_first  ( data, "Модель дефектів в ід. част. плівки - сферичні кластери",offset-1)>=0)
{
xx_line =find_line_another ( data, "Модель дефектів в ід. част. плівки - сферичні кластери",offset-1);
xx_line2=find_line_another ( data, "Конц.сф.кластерів:",offset-1);
if (xx_line+2==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -2; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Edit168->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
Edit169->Text=FloatToStr(master.get_element(0,1));
Edit171->Text=FloatToStr(master.get_element(1,0));
Edit172->Text=FloatToStr(master.get_element(1,1));
CheckBox32->Checked=true; CheckBox33->Checked=true;
if (StrToFloat(Edit171->Text)<0.001 && StrToFloat(Edit172->Text)<0.001) CheckBox33->Checked=false;
}
}
 // !!!!!!!!!!!   +інші типи дефектів
}





xx = find_first ( data, "Дефекти у ППШ",offset-1);
if (xx!=-4) offset=xx;
if (find_first  ( data, "Дефекти у ППШ не враховуються",offset-1)>=0)
  {
  CheckBox1->Checked=false; CheckBox58->Checked=false; CheckBox2->Checked=false;
  CheckBox26->Checked=false; CheckBox4->Checked=false;
  }
else
{
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=",offset-1)>=0)
{
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/sqrt(2)",offset-1)>=0) RadioButton7->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a",offset-1)>=0)         RadioButton8->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a*sqrt(2)",offset-1)>=0) RadioButton9->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/3*sqrt(3)",offset-1)>=0) RadioButton28->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a/2*sqrt(3)",offset-1)>=0) RadioButton32->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=a*sqrt(3)",offset-1)>=0) RadioButton29->Checked=true;

xx = find_first ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса",offset-1);
if (xx!=-4) offset=xx;

xx_line =find_line_another ( data, "Модель дефектів в ПШ - дисл. петлі з вект. Бюргерса b=",offset-1);
xx_line2=find_line_another ( data, "Коефіцієнт на L:",offset-1);
if (xx_line+1==xx_line2)
{
    rectangle data_range2;
    data_range2.top = xx_line+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 18;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
Edit149->Text=FloatToStr(master2.get_element(0,0));    // Коефіцієнт на L
}
if (find_first ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1)>=0)
{
xx_line2 =find_line_another ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1);
if (xx_line+2==xx_line2) CheckBox6->Checked=true;
}
else
{
xx_line2 =find_line_another ( data, "Концентрація дефектів однакова по всьому профілю деформації",offset-1);
if (xx_line+2==xx_line2) CheckBox6->Checked=false;
}

if (find_first ( data, "Радіус дефектів пропорційний профілю деформації",offset-1)>=0)
{
xx_line2 =find_line_another ( data, "Радіус дефектів пропорційний профілю деформації",offset-1);
if (xx_line+3==xx_line2) CheckBox7->Checked=true;
}
else
{
xx_line2 =find_line_another ( data, "Радіус дефектів однаковий по всьому профілю деформації",offset-1);
if (xx_line+3==xx_line2) CheckBox7->Checked=false;
}
xx_line2=find_line_another ( data, "Конц.петель:",offset-1);
if (xx_line+4==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -1; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Edit2->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
Edit3->Text=FloatToStr(master.get_element(0,1));
CheckBox1->Checked=true;
}
}

if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111)",offset-1)>=0)
{
Memo8->Lines->Add("1111");
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/sqrt(2)",offset-1)>=0) RadioButton7->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a",offset-1)>=0)         RadioButton8->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a*sqrt(2)",offset-1)>=0) RadioButton9->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/3*sqrt(3)",offset-1)>=0) RadioButton28->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a/2*sqrt(3)",offset-1)>=0) RadioButton32->Checked=true;
if (find_first  ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111) з вект. Бюргерса b=a*sqrt(3)",offset-1)>=0) RadioButton29->Checked=true;

xx = find_first ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111)",offset-1);
if (xx!=-4) offset=xx;
xx_line =find_line_another ( data, "Модель дефектів в ПШ - дисл. петлі в площині (111)",offset-1);
xx_line2=find_line_another ( data, "Коефіцієнт на L:",offset-1);
Memo8->Lines->Add(IntToStr(xx_line)+'\t'+IntToStr(xx_line2));
if (xx_line+1==xx_line2)
{
    rectangle data_range2;
    data_range2.top = xx_line+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 18;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
Edit149->Text=FloatToStr(master2.get_element(0,0));    // Коефіцієнт на L
    Memo8->Lines->Add(FloatToStr( master2.get_element(0,0)) );
Memo8->Lines->Add(Edit149->Text);
Memo8->Lines->Add(IntToStr(xx)+'\t'+IntToStr(xx_));
}
if (find_first ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1)>=0)
{
xx_line2 =find_line_another ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1);
if (xx_line+2==xx_line2) CheckBox39->Checked=true;
}
else
{
xx_line2 =find_line_another ( data, "Концентрація дефектів однакова по всьому профілю деформації",offset-1);
if (xx_line+2==xx_line2) CheckBox39->Checked=false;
}

if (find_first ( data, "Радіус дефектів пропорційний профілю деформації",offset-1)>=0)
{
xx_line2 =find_line_another ( data, "Радіус дефектів пропорційний профілю деформації",offset-1);
if (xx_line+3==xx_line2) CheckBox61->Checked=true;
}
else
{
xx_line2 =find_line_another ( data, "Радіус дефектів однаковий по всьому профілю деформації",offset-1);
if (xx_line+3==xx_line2) CheckBox61->Checked=false;
}
xx_line2=find_line_another ( data, "Конц.петель:",offset-1);
if (xx_line+4==xx_line2)
{
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -1; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Edit218->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
Edit219->Text=FloatToStr(master.get_element(0,1));
CheckBox58->Checked=true;
RadioButton63->Checked=true;
}
}

if (find_first  ( data, "Модель дефектів в ПШ - сферичні кластери (точкові дефекти)",offset-1)>=0)
  {
  xx = find_first ( data, "Модель дефектів в ПШ - сферичні кластери (точкові дефекти)",offset-1);
  if (xx!=-4) offset=xx;
  xx_line =find_line_another ( data, "Модель дефектів в ПШ - сферичні кластери (точкові дефекти)",offset-1);
  //  xx_line2=find_line_another ( data, "Конц.сф.кластерів:",offset-1);
  if (find_first ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1)>=0)
    {
    xx_line2 =find_line_another ( data, "Концентрація дефектів пропорційна профілю деформації",offset-1);
    if (xx_line+1==xx_line2) CheckBox69->Checked=true;
    }
    else
    {
    xx_line2 =find_line_another ( data, "Концентрація дефектів однакова по всьому профілю деформації",offset-1);
    if (xx_line+1==xx_line2) CheckBox69->Checked=false;
    }

  if (find_first ( data, "Радіус дефектів пропорційний профілю деформації",offset-1)>=0)
    {
    xx_line2 =find_line_another ( data, "Радіус дефектів пропорційний профілю деформації",offset-1);
    if (xx_line+2==xx_line2) CheckBox70->Checked=true;
    }
    else
    {
    xx_line2 =find_line_another ( data, "Радіус дефектів однаковий по всьому профілю деформації",offset-1);
    if (xx_line+2==xx_line2) CheckBox70->Checked=false;
    }
  xx_line2=find_line_another ( data, "Конц.сф.кластерів (т.д.):",offset-1);
  if (xx_line+3==xx_line2)
    {
    rectangle data_range;
    data_range.top = xx_line2+1;
    data_range.bottom = lines_count - data_range.top -1; // 2 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
    Edit250->Text=FloatToStr(master.get_element(0,0));   // Конц.петель: Радіус петель:
    Edit251->Text=FloatToStr(master.get_element(0,1));
    CheckBox26->Checked=true;
    }  
  }

 // !!!!!!!!!!!   +інші типи дефектів
}

xx = find_first ( data, "Параметри  профілю:",offset-1);
if (xx!=-4) {offset=xx; CheckBox67->Checked=true;}

if (find_first ( data, "Профіль - гаусіана з параметрами:",offset-1)>=0)
{
RadioButton3->Checked=true;
xx_line=find_line_another ( data, "Профіль - гаусіана з параметрами:",offset-1);
    rectangle data_range;
    data_range.top = xx_line+1;
    data_range.bottom = lines_count - data_range.top -13; // 13 - кількість рядків із даними
    data_range.left = 5;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), " ", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }



      /*
Memo9->Lines->Add("# " + OtkEt->FileName);
    // Зчитуємо і обробляємо файл
//    J_get_narray master( OtkEt->FileName.c_str(), " ");
    rectangle data_range;
     char *searching = new char [100];
     searching ="Параметри " ;
//     char *searching = Edit164->Text;    // доробити

    long  vidstup =  find_line ( data, searching) + StrToInt(Edit72->Text);
     // Звільняємо пам`ять
   delete data, searching;

 // int vidstup=StrToInt(Edit72->Text);
    data_range.top = vidstup;      //14;
    data_range.bottom = lines_count - data_range.top - 13; // 13 - кількість рядків із даними
    data_range.left = 5;
    data_range.right = 0;

    J_get_narray master( OtkEt->FileName.c_str(), " ", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }

    // Динамічно виділяємо пам'ять під масив
    double *zz = new double [ master.get_rows() ];

//    Memo2->Lines->Add(FloatToStrF( master.get_rows(), ffGeneral,5,2 ));
//    Memo2->Lines->Add(FloatToStrF( master.get_columns(), ffGeneral,1,1  ) );

    // Пробігаємося по всіх новоутворених елементах
    for( long i = 0; i < master.get_rows(); i++ )
    {
     for( long j = 0; j < master.get_columns(); j++){
       // Записуємо у файл у рядок дані, попутно перетворивши їх у експоненційну форму.
       //zz[i] = master.get_element(i,j);

//       Memo2->Lines->Add(FloatToStrF( master.get_element(i,j), ffGeneral, 10, 4 ) );
//     Memo2->Lines->Add(FloatToStrF( master.get_element(i,j), ffExponent, 10, 4 ) );
         //якщо master.get_element(i,0)  то тільки стов.
    }
    }
  // Чистимо виділену пам'ять
  delete zz;                     */

Edit35->Text=FloatToStr(master.get_element(0,0));    // Dmax1
Edit36->Text=FloatToStr(master.get_element(1,0));    // D01
Edit37->Text=FloatToStr(master.get_element(2,0));    // L1
Edit38->Text=FloatToStr(master.get_element(3,0));    // Rp1
Edit39->Text=FloatToStr(master.get_element(4,0));    // D02
Edit40->Text=FloatToStr(master.get_element(5,0));    // L2
Edit41->Text=FloatToStr(master.get_element(6,0));    // Rp2
Edit42->Text=FloatToStr(master.get_element(7,0));    // Dmin
Edit33->Text=FloatToStr(master.get_element(10,0));   // dl
}

if (find_first ( data, "Профіль обч. з дефектів, f - гаусіана з параметрами:",offset-1)>=0)
{
RadioButton34->Checked=true;
xx_line=find_line_another ( data, "Профіль обч. з дефектів, f - гаусіана з параметрами:",offset-1);
    rectangle data_range;
    data_range.top = xx_line+1;
    data_range.bottom = lines_count - data_range.top -13; // 13 - кількість рядків із даними
    data_range.left = 5;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), " ", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }

Edit35->Text=FloatToStr(master.get_element(0,0));    // Dmax1
Edit36->Text=FloatToStr(master.get_element(1,0));    // D01
Edit37->Text=FloatToStr(master.get_element(2,0));    // L1
Edit38->Text=FloatToStr(master.get_element(3,0));    // Rp1
Edit39->Text=FloatToStr(master.get_element(4,0));    // D02
Edit40->Text=FloatToStr(master.get_element(5,0));    // L2
Edit41->Text=FloatToStr(master.get_element(6,0));    // Rp2
Edit42->Text=FloatToStr(master.get_element(7,0));    // Dmin
Edit33->Text=FloatToStr(master.get_element(10,0));   // dl
}


if (find_first ( data, "Профіль - сходинки:",offset-1)>=0)
{
RadioButton4->Checked=true;
xx_line=find_line_another ( data, "Профіль - сходинки:",offset-1);

    rectangle data_range2;
    data_range2.top = xx_line+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 20;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
    km= master2.get_element(0,0);
    Edit90->Text=IntToStr(km);    // Кількість підшарів

xx_line=find_line_another ( data, "№ 	 Деформ. підшару (%) 	 Товщина підшару (А)",offset-1);
    rectangle data_range6;
    data_range6.top = xx_line+1;
    data_range6.bottom = lines_count - data_range6.top -km-1; // 1 - кількість рядків із даними
    data_range6.left = 0;
    data_range6.right = 0;
    J_get_narray master6( OtkEt->FileName.c_str(), "\t", data_range6);
    if( master6.get_err_code() != ALL_OK ){ return; }
    Memo5->Clear();
for (int i=0; i<=km-1; i++)
{
Memo5->Lines->Add(FloatToStr(master6.get_element(i,1)/100)+'\t'+FloatToStr(master6.get_element(i,2)));
}


}

xx = find_first ( data, "*****************************************************************",offset-1);
if (xx!=-4) offset=xx;
if (xx>=0)
        {
if (find_first ( data,"Результати наближення за програмами типу Auto",offset-1)>=0)
        MessageBox(0,"Результати наближення за програмами типу Auto","Увага!", MB_OK + MB_ICONEXCLAMATION);
if (find_first ( data,"Результати наближення за програмами типу Gausauto",offset-1)>=0)
{
MessageBox(0,"Результати наближення за програмами типу Gausauto","Увага!", MB_OK + MB_ICONEXCLAMATION);
RadioButton30->Checked=true;
}

xx_line =find_line_another ( data, "Задана кількість циклів =",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line+0;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 26;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
Edit73->Text=FloatToStr(master2.get_element(0,0));    // Задана кількість циклів

if (find_first ( data,"Наближ. в даному напрямку",offset-1)>=0)
        CheckBox21->Checked=true; else CheckBox21->Checked=false;
if (find_first ( data,"Зменшення кроку",offset-1)>=0)
        CheckBox25->Checked=true; else CheckBox25->Checked=false;
if (find_first ( data,"Мінімізація ВСКВ",offset-1)>=0)
        CheckBox25->Checked=true; else CheckBox25->Checked=false;


offset=xx;
if (find_first  ( data, "Наближення  профілю-гаусіни  в ППШ",offset-1)>=0)
{
CheckBox28->Checked=false;
xx_line =find_line_another ( data, "Результати  наближення профілю шляхом зміни параметрів гаусіан",offset-1);
    rectangle data_range;
    data_range.top = xx_line+2;
    data_range.bottom = lines_count - data_range.top -7; // 13 - кількість рядків із даними
    data_range.left = 9;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
Edit98->Text=FloatToStr(master.get_element(0,0));    // Dmax1
Edit101->Text=FloatToStr(master.get_element(1,0));    // D01
Edit102->Text=FloatToStr(master.get_element(2,0));    // L1
Edit103->Text=FloatToStr(master.get_element(3,0));    // Rp1
Edit104->Text=FloatToStr(master.get_element(4,0));    // D02
Edit105->Text=FloatToStr(master.get_element(5,0));    // L2
Edit106->Text=FloatToStr(master.get_element(6,0));    // Rp2
//Edit128->Text=FloatToStr(master.get_element(7,0));    // Dmin

Edit111->Text=FloatToStr(master.get_element(0,1));    // Dmax1
Edit112->Text=FloatToStr(master.get_element(1,1));    // D01
Edit113->Text=FloatToStr(master.get_element(2,1));    // L1
Edit114->Text=FloatToStr(master.get_element(3,1));    // Rp1
Edit115->Text=FloatToStr(master.get_element(4,1));    // D02
Edit116->Text=FloatToStr(master.get_element(5,1));    // L2
Edit117->Text=FloatToStr(master.get_element(6,1));    // Rp2
//Edit118->Text=FloatToStr(master.get_element(7,1));    // Dmin

Edit119->Text=FloatToStr(master.get_element(0,2));    // stDmax1
Edit120->Text=FloatToStr(master.get_element(1,2));    // stD01
Edit121->Text=FloatToStr(master.get_element(2,2));    // stL1
Edit122->Text=FloatToStr(master.get_element(3,2));    // stRp1
Edit123->Text=FloatToStr(master.get_element(4,2));    // stD02
Edit124->Text=FloatToStr(master.get_element(5,2));    // stL2
Edit125->Text=FloatToStr(master.get_element(6,2));    // stRp2
//Edit126->Text=FloatToStr(master.get_element(7,2));    // stDmin

xx_line2 =find_line_another ( data, "Всі параметри наближеного профілю:",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -13; // 13 - кількість рядків із даними
    data_range2.left = 5;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), " ", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
Edit107->Text=FloatToStr(master2.get_element(7,0));   // Dmin
Edit97->Text=FloatToStr(master2.get_element(10,0));   // dl
//Edit127->Text=FloatToStr(vidstup2-vidstup-2);         //К-ть змінних параметрів
}


offset=xx;
if (find_first  ( data, "Наближення  профілю-сходинок  в ППШ",offset-1)>=0)
{
if (find_first  ( data, "Стартовий профіль - профіль сходинками",offset-1)>=0)
                  CheckBox38->Checked=false;
                  else  CheckBox38->Checked=true;
xx_line =find_line_another ( data, "Результати  наближення профілю",offset-1);
    rectangle data_range;
    data_range.top = xx_line+2;
    data_range.bottom = lines_count - data_range.top -km; // 13 - кількість рядків із даними
    data_range.left = 0;
    data_range.right = 0;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
    Memo2->Clear();
for (int i=0; i<=km-1; i++)
{
Memo2->Lines->Add(FloatToStr(master.get_element(i,3)/100)+'\t'+FloatToStr(master.get_element(i,4)));
}

}






        }
/*long posTE_KDV=find_first ( data, " Теоретичні та експериментальна КДВ:",0);
//long lineTE_KDV=find_line ( data, " Теоретичні та експериментальна КДВ:");
xx=find_first  ( data, "КДВ (444)",posTE_KDV-1);
unsigned long linesCountTotal = count_substr( data, "\n", 0);
unsigned long linesCountNext = count_substr( data, "\n", xx + 1 );
long vidstup =linesCountTotal - linesCountNext;   */

xx = find_first ( data, " Теоретичні та експериментальна КДВ:",offset-1);
if (xx!=-4) offset=xx;

xx = find_first ( data, "КДВ (444)",offset-1);
if (xx>=0)
{
 offset=xx;
 xx_line2 = find_line_another  ( data, "КДВ (444)",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 20;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
ik_[1]=master2.get_element(0,0);
Edit202->Text=FloatToStr(ik_[1]);    // крок (444)
if (master2.get_columns()==2)
  {
  Edit213->Text=FloatToStr(master2.get_element(0,1));
  CheckBox54->Checked=true;
  }

 xx = find_first  ( data, "m1=",offset-1);
 xx_ =find_first  ( data, "m10=",offset-1);
//Edit230->Text=FloatToStr(xx);    // крок (444)
//Edit233->Text=FloatToStr(xx_line1);    // крок (444)
//Edit234->Text=FloatToStr(xx_line1-xx);    // крок (444)
    rectangle data_range3;
    data_range3.top = xx_line2+2;
    data_range3.bottom = lines_count - data_range3.top -1; // 1 - кількість рядків із даними
    data_range3.left = 3;
    data_range3.right = 8; //xx;
    J_get_narray master3( OtkEt->FileName.c_str(), "\t", data_range3);
    if( master3.get_err_code() != ALL_OK ){ return; }
m1_[1]=master3.get_element(0,0);
Edit235->Text=FloatToStr(m1_[1]);
    rectangle data_range4;
    data_range4.top = xx_line2+2;
    data_range4.bottom = lines_count - data_range4.top -1; // 1 - кількість рядків із даними
    data_range4.left = xx_-xx+4; // 13; //xx+3;
    data_range4.right = 0;
    J_get_narray master4( OtkEt->FileName.c_str(), "\t", data_range4);
    if( master4.get_err_code() != ALL_OK ){ return; }
m10_[1]=master4.get_element(0,0);
Edit238->Text=FloatToStr(m10_[1]);

 xx = find_first  ( data, "nskv (с)=",offset-1);
 xx_ =find_first  ( data, "kskv (с)=",offset-1);
    rectangle data_range;
    data_range.top = xx_line2+3;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 11;
    data_range.right = 14;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
double nskv1=master.get_element(0,0);
Edit69->Text=FloatToStr(nskv1);
    rectangle data_range5;
    data_range5.top = xx_line2+3;
    data_range5.bottom = lines_count - data_range5.top -1; // 1 - кількість рядків із даними
    data_range5.left = xx_-xx+11;  //27;
    data_range5.right = 0;
    J_get_narray master5( OtkEt->FileName.c_str(), "\t", data_range5);
    if( master5.get_err_code() != ALL_OK ){ return; }
double kskv1=master5.get_element(0,0);
Edit70->Text=FloatToStr(kskv1);

   xx_line = find_line_another  ( data, "Дифузна складова від ПШ не обчислювалася",offset-1);
if (xx_line2+5==xx_line) CheckBox62->Checked=true;

  xx_line = find_line_another  ( data, "Кут (с)",offset-1);
    rectangle data_range6;
    data_range6.top = xx_line+1;
    data_range6.bottom = lines_count - data_range6.top -m1_[1]-2; // 1 - кількість рядків із даними
    data_range6.left = 0;
    data_range6.right = 0;
    J_get_narray master6( OtkEt->FileName.c_str(), "\t", data_range6);
    if( master6.get_err_code() != ALL_OK ){ return; }

//if (CheckBox31->Checked==false)
  for (int i=0; i<=m1_[1]; i++)
    {
    intIk2d[i][1]=master6.get_element(i,8);
    intI02d[i][1]=master6.get_element(i,9);
    }
//if (CheckBox31->Checked==true)
//  for (int i=0; i<=m1_[1]; i++)
//    {
//    intIk2d[i][1]=master6.get_element(i,9);
//    intI02d[i][1]=master6.get_element(i,10);
//    }                     
//for (int i=0; i<=m1_[1]; i++)
//          Memo8->Lines->Add(FloatToStr(i)+'\t'+FloatToStr(intIk2d[i][1])+'\t'+FloatToStr(intI02d[i][1]));
double TetaMin,  DeltaTeta1;
TetaMin=-(m10_[1])*ik_[1];
for (int i=0;i<=m1_[1];i++)
{
DeltaTeta1=(TetaMin+i*ik_[1]);

if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk2d[i][1],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk2d[i][1],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk2d[i][1],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series11->AddXY(DeltaTeta1,intIk2d[i][1],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk2d[i][1],"",clGreen);
}
}
}
Memo9->Lines->Add( "444 пройшло");


xx = find_first ( data, "КДВ (888)",offset-1);
if (xx>=0)
{
 offset=xx;
 xx_line2 = find_line_another  ( data, "КДВ (888)",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 20;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
ik_[2]=master2.get_element(0,0);
Edit203->Text=FloatToStr(ik_[2]);    // крок (888)
if (master2.get_columns()==2)
  {
  Edit214->Text=FloatToStr(master2.get_element(0,1));
  CheckBox54->Checked=true;
  }

 xx = find_first  ( data, "m1=",offset-1);
 xx_= find_first  ( data, "m10=",offset-1);

    rectangle data_range3;
    data_range3.top = xx_line2+2;
    data_range3.bottom = lines_count - data_range3.top -1; // 1 - кількість рядків із даними
    data_range3.left = 3;
    data_range3.right = 8; //xx;
    J_get_narray master3( OtkEt->FileName.c_str(), "\t", data_range3);
    if( master3.get_err_code() != ALL_OK ){ return; }
m1_[2]=master3.get_element(0,0);
Edit236->Text=FloatToStr(m1_[2]);
    rectangle data_range4;
    data_range4.top = xx_line2+2;
    data_range4.bottom = lines_count - data_range4.top -1; // 1 - кількість рядків із даними
    data_range4.left = xx_-xx+4; //14; //xx+3;
    data_range4.right = 0;
    J_get_narray master4( OtkEt->FileName.c_str(), "\t", data_range4);
    if( master4.get_err_code() != ALL_OK ){ return; }
m10_[2]=master4.get_element(0,0);
Edit239->Text=FloatToStr(m10_[2]);

 xx = find_first  ( data, "nskv (с)=",offset-1);
 xx_ =find_first  ( data, "kskv (с)=",offset-1);
    rectangle data_range;
    data_range.top = xx_line2+3;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 11;
    data_range.right = 14;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
double nskv2=master.get_element(0,0);
Edit130->Text=FloatToStr(nskv2);
    rectangle data_range5;
    data_range5.top = xx_line2+3;
    data_range5.bottom = lines_count - data_range5.top -1; // 1 - кількість рядків із даними
    data_range5.left = xx_-xx+11;  //27;
    data_range5.right = 0;
    J_get_narray master5( OtkEt->FileName.c_str(), "\t", data_range5);
    if( master5.get_err_code() != ALL_OK ){ return; }
double kskv2=master5.get_element(0,0);
Edit129->Text=FloatToStr(kskv2);

 xx_line = find_line_another  ( data, "Дифузна складова від ПШ не обчислювалася",offset-1);
if (xx_line2+5==xx_line) CheckBox63->Checked=true;
 xx_line = find_line_another  ( data, "Кут (с)",offset-1);
    rectangle data_range6;
    data_range6.top = xx_line+1;
    data_range6.bottom = lines_count - data_range6.top -m1_[2]-2; // 1 - кількість рядків із даними
    data_range6.left = 0;
    data_range6.right = 0;
    J_get_narray master6( OtkEt->FileName.c_str(), "\t", data_range6);
    if( master6.get_err_code() != ALL_OK ){ return; }
for (int i=0; i<=m1_[2]; i++)
{
intIk2d[i][2]=master6.get_element(i,8);
intI02d[i][2]=master6.get_element(i,9);
}

double TetaMin,  DeltaTeta1;
TetaMin=-(m10_[2])*ik_[2];
for (int i=0;i<=m1_[2];i++)
{
DeltaTeta1=(TetaMin+i*ik_[2]);

if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk2d[i][2],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk2d[i][2],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk2d[i][2],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series1->AddXY(DeltaTeta1,intIk2d[i][2],"",clGreen);
Series25->AddXY(DeltaTeta1,intIk2d[i][2],"",clGreen);
}
}
}
Memo9->Lines->Add( "888 пройшло");


xx = find_first ( data, "КДВ (880)",offset-1);
if (xx>=0)
{
 offset=xx;
 xx_line2 = find_line_another  ( data, "КДВ (880)",offset-1);
    rectangle data_range2;
    data_range2.top = xx_line2+1;
    data_range2.bottom = lines_count - data_range2.top -1; // 1 - кількість рядків із даними
    data_range2.left = 20;
    data_range2.right = 0;
    J_get_narray master2( OtkEt->FileName.c_str(), "\t", data_range2);
    if( master2.get_err_code() != ALL_OK ){ return; }
ik_[3]=master2.get_element(0,0);
Edit204->Text=FloatToStr(ik_[3]);    // крок (880)
if (master2.get_columns()==2)
  {
  Edit215->Text=FloatToStr(master2.get_element(0,1));
  CheckBox54->Checked=true;
  }

 xx = find_first  ( data, "m1=",offset-1);
 xx_= find_first  ( data, "m10=",offset-1);

    rectangle data_range3;
    data_range3.top = xx_line2+2;
    data_range3.bottom = lines_count - data_range3.top -1; // 1 - кількість рядків із даними
    data_range3.left = 3;
    data_range3.right = 8; //xx;
    J_get_narray master3( OtkEt->FileName.c_str(), "\t", data_range3);
    if( master3.get_err_code() != ALL_OK ){ return; }
m1_[3]=master3.get_element(0,0);
Edit237->Text=FloatToStr(m1_[3]);
    rectangle data_range4;
    data_range4.top = xx_line2+2;
    data_range4.bottom = lines_count - data_range4.top -1; // 1 - кількість рядків із даними
    data_range4.left = xx_-xx+4; //xx+3;
    data_range4.right = 0;
    J_get_narray master4( OtkEt->FileName.c_str(), "\t", data_range4);
    if( master4.get_err_code() != ALL_OK ){ return; }
m10_[3]=master4.get_element(0,0);
Edit240->Text=FloatToStr(m10_[3]);

 xx = find_first  ( data, "nskv (с)=",offset-1);
 xx_ =find_first  ( data, "kskv (с)=",offset-1);
    rectangle data_range;
    data_range.top = xx_line2+3;
    data_range.bottom = lines_count - data_range.top -1; // 1 - кількість рядків із даними
    data_range.left = 11;
    data_range.right = 14;
    J_get_narray master( OtkEt->FileName.c_str(), "\t", data_range);
    if( master.get_err_code() != ALL_OK ){ return; }
double nskv3=master.get_element(0,0);
Edit89->Text=FloatToStr(nskv3);
    rectangle data_range5;
    data_range5.top = xx_line2+3;
    data_range5.bottom = lines_count - data_range5.top -1; // 1 - кількість рядків із даними
    data_range5.left = xx_-xx+11;  //27;
    data_range5.right = 0;
    J_get_narray master5( OtkEt->FileName.c_str(), "\t", data_range5);
    if( master5.get_err_code() != ALL_OK ){ return; }
double kskv3=master5.get_element(0,0);
Edit134->Text=FloatToStr(kskv3);

 xx_line = find_line_another  ( data, "Дифузна складова від ПШ не обчислювалася",offset-1);
if (xx_line2+5==xx_line) CheckBox64->Checked=true;
 xx_line = find_line_another  ( data, "Кут (с)",offset-1);
    rectangle data_range6;
    data_range6.top = xx_line+1;
    data_range6.bottom = lines_count - data_range6.top -m1_[3]-2; // 1 - кількість рядків із даними
    data_range6.left = 0;
    data_range6.right = 0;
    J_get_narray master6( OtkEt->FileName.c_str(), "\t", data_range6);
    if( master6.get_err_code() != ALL_OK ){ return; }
for (int i=0; i<=m1_[3]; i++)
{
intIk2d[i][3]=master6.get_element(i,8);
intI02d[i][3]=master6.get_element(i,9);
}
Memo9->Lines->Add( "880 графік");

double TetaMin,  DeltaTeta1;
TetaMin=-(m10_[3])*ik_[3];
for (int i=0;i<=m1_[3];i++)
{
DeltaTeta1=(TetaMin+i*ik_[3]);

if (number_KDV==1)
{
Series1->AddXY(DeltaTeta1,intIk2d[i][3],"",clGreen);
Series11->AddXY(DeltaTeta1,intIk2d[i][3],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk2d[i][3],"",clGreen);
}
if (number_KDV==2 || number_KDV==3)
{
//Chart6->LeftAxis->Logarithmic=false;
Series45->AddXY(DeltaTeta1,intIk2d[i][3],"",clGreen);
Series24->AddXY(DeltaTeta1,intIk2d[i][3],"",clGreen);
}
}
}
Memo9->Lines->Add( "880 пройшло");



 }
 }
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------


void __fastcall TForm1::Button18Click(TObject *Sender)
{
Edit35->Text=Edit111->Text;
Edit36->Text=Edit112->Text;
Edit37->Text=Edit113->Text;
Edit38->Text=Edit114->Text;
Edit39->Text=Edit115->Text;
Edit40->Text=Edit116->Text;
Edit41->Text=Edit117->Text;
Edit42->Text=Edit107->Text;
Edit33->Text=Edit97->Text;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button19Click(TObject *Sender)
{
Memo5->Clear();
km=StrToInt(Edit90->Text);
ReadMemo2stovp(Memo2,km, DD,Dl);   //       Зчитуємо профіль з Memo2
for (int k=1; k<=km;k++) Memo5->Lines->Add(FloatToStr(DD[k])+'\t'+FloatToStr(Dl[k]));
}
//---------------------------------------------------------------------------


void __fastcall TForm1::Button20Click(TObject *Sender)
{
Edit53->Text=Edit155->Text;
Edit54->Text=Edit158->Text;
Edit64->Text=Edit156->Text;
Edit65->Text=Edit157->Text;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button21Click(TObject *Sender)
{
Edit2->Text=Edit153->Text;
Edit3->Text=Edit154->Text;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void TForm1::ReadMemo2stovp(TMemo *Memo,int km, double *DD,double *Dl)
{         //      1<=k<=km
AnsiString Ds22,Ds11;
//AnsiString Mas[KM],Mas1[KM],Mas2[KM];
AnsiString *Mas, *Mas1, *Mas2;
  Mas  = new AnsiString[KM];
  Mas1 = new AnsiString[KM];
  Mas2 = new AnsiString[KM];

  for (int i=0;i<=km-1;i++)
{
Mas[i]=(Memo->Lines->Strings[i]);  //зчитуємо у масив з Memo5
//Mas[i]=List->Strings[i];// зчитуємо у масив рядки файла
Mas1[i]="";
for (int k=1; k<=(Mas[i].Length());k++)
{
Ds11=Mas[i][k]; //допоміжна змінна типу AnsiString
if (Ds11!=("\t"))
if (Ds11!=(" "))
Mas1[i]=Mas1[i]+Ds11; //у масив Mas1 заносяться значення першого стовпця
else break;
else break;
}
DD[i]=atof(Mas1[i].c_str());//перший стовбець переводиться із тексту в числа
Mas2[i]="";
for ( int j=Mas1[i].Length()+2; j<=Mas[i].Length();j++ )
{
Ds22=Mas[i][j];//допоміжна змінна типу AnsiString
if (Ds22==("\t")) j++;
if (Ds22==(" ")) j++;
else
Mas2[i]=Mas2[i]+Ds22;//у масив Mas2 заносяться значення другого стовпця
}
Dl[i]=atof(Mas2[i].c_str());//*1e-8;//другий стовбець переводиться із тексту в числа
}
for (int k=km; k>=1;k--) //  Перенумерація елементів масивів
{
DD[k]=DD[k-1];
Dl[k]=Dl[k-1];
}
delete Mas, Mas1, Mas2;
}
//---------------------------------------------------------------------------
void TForm1::ReadMemo1stovp(TMemo *Memo,int km, double *DD)
{
AnsiString Ds11;
//AnsiString Mas[100],Mas1[100];
AnsiString *Mas, *Mas1;
  Mas  = new AnsiString[KM];
  Mas1 = new AnsiString[KM];

  for (int i=0;i<=km;i++)
{
Mas[i]=(Memo->Lines->Strings[i]);  //зчитуємо у масив з Memo5
//Mas[i]=List->Strings[i];// зчитуємо у масив рядки файла
Mas1[i]="";
for (int k=1; k<=(Mas[i].Length());k++)
{
Ds11=Mas[i][k]; //допоміжна змінна типу AnsiString
if (Ds11!=("\t"))
if (Ds11!=(" "))
Mas1[i]=Mas1[i]+Ds11; //у масив Mas1 заносяться значення першого стовпця
else break;
else break;
}
DD[i]=atof(Mas1[i].c_str());//перший стовбець переводиться із тексту в числа
}
/*for (int k=km; k>=1;k--) //  Перенумерація елементів масивів
{
DD[k]=DD[k-1];
}  */
delete Mas, Mas1;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

/*void __fastcall TForm1::Button22Click(TObject *Sender)
{

Form2->Edit2->Text = Form1->Edit69->Text;

Form2->Visible = true;
//Form2->Edit2->Text = Form1->Edit69->Text;
Form1->Button1->Height = 30;
Form1->Chart11->Height = 300;
Form1->Chart11->LeftAxis->Logarithmic=true;
Label35->Caption=Mu0;
}                        */
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------


void __fastcall TForm1::Button26Click(TObject *Sender)  //логарифмічна шкала
{
Chart3->LeftAxis->Automatic = False ;
Chart6->LeftAxis->Automatic = False ;
Chart11->LeftAxis->Automatic = False ;
Chart3->LeftAxis->Minimum = 0.00001 ;
Chart6->LeftAxis->Minimum = 0.00001 ;
Chart11->LeftAxis->Minimum = 0.00001 ;
Chart3->LeftAxis->Logarithmic=true;
Chart6->LeftAxis->Logarithmic=true;
Chart11->LeftAxis->Logarithmic=true;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button27Click(TObject *Sender)  // лінійна шкала
{
Chart3->LeftAxis->Logarithmic=false;
Chart6->LeftAxis->Logarithmic=false;
Chart11->LeftAxis->Logarithmic=false;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void __fastcall TForm1::Button22Click(TObject *Sender)
{
Memo7->Clear();
km_rozv=StrToInt(Edit84->Text);
ReadMemo2stovp(Memo2,km_rozv+1, nn_m,DFi);   //       Зчитуємо профіль з Memo2
for (int k=1; k<=km_rozv+1;k++) Memo7->Lines->Add(FloatToStr(nn_m[k])+'\t'+FloatToStr(DFi[k]));
}
//---------------------------------------------------------------------------



void __fastcall TForm1::FormDestroy(TObject *Sender)
{   // По закритті форми - чистимо пам'ять
    this->clearArrays();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::CheckBox54Click(TObject *Sender)  //Зміна кроку експ.КДВ
{
Edit213->Text=Edit202->Text;
Edit214->Text=Edit203->Text;
Edit215->Text=Edit204->Text;
}
//---------------------------------------------------------------------------


void __fastcall TForm1::RadioButton39Click(TObject *Sender)
{
MessageBox(0,"Тільки (444) і (888)","!!!", MB_OK + MB_ICONEXCLAMATION);
}
//---------------------------------------------------------------------------

void __fastcall TForm1::RadioButton40Click(TObject *Sender)
{
MessageBox(0,"Тільки (444)","!!!", MB_OK + MB_ICONEXCLAMATION);
}
//---------------------------------------------------------------------------

void __fastcall TForm1::RadioButton44Click(TObject *Sender)
{
MessageBox(0,"Тільки (444) і (888)","!!!", MB_OK + MB_ICONEXCLAMATION);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------


void __fastcall TForm1::Button1Click(TObject *Sender)
{
double  koefCKV1, koefCKV2, koefCKV3;
if (CheckBox42->Checked==true)  koefCKV1=StrToFloat(Edit400->Text);
if (CheckBox43->Checked==true)  koefCKV2=koefCKV1/StrToFloat(Edit392->Text);
if (CheckBox44->Checked==true)  koefCKV3=koefCKV1/StrToFloat(Edit393->Text);

//    Edit400->Text=FloatToStr(koefCKV1);
if (CheckBox43->Checked==true)    Edit401->Text=FloatToStr(koefCKV2);
if (CheckBox44->Checked==true)    Edit402->Text=FloatToStr(koefCKV3);
}
//---------------------------------------------------------------------------


