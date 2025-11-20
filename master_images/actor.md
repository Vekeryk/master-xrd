```plantUML
@startuml
actor Користувач

rectangle "Система аналізу КДВ" {
    component "Базове ПЗ\nC++ Builder" as CPP #LightBlue

    storage "Файловий обмін\nКДВ ⇄ Параметри" as Files #LightGray

    component "ML Модуль\nPyTorch" as ML #LightYellow
}

Користувач --> CPP : Запуск аналізу
CPP -right-> Files : Запис КДВ
Files -down-> ML : Читання
ML -up-> Files : Запис параметрів
Files -left-> CPP : Читання
CPP --> Користувач : Результати

@enduml
```
