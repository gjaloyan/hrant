---
topic: source code analysis
category: profession
created: 2026-04-07 19:56
updated: 2026-04-07 19:56
keywords: SAST, static analysis, code security, vulnerability detection, OWASP, false positives, CI/CD integration, code quality, security testing, automated analysis
source: //duckduckgo.com/l/?uddg=https%3A%2F%2Fowasp.org%2Fwww%2Dcommunity%2FSource_Code_Analysis_Tools&amp;rut=8090f53fad015d18d4f62ff8cdf63eced83b10ad3c5f581c6c695cd79ba928d3; //duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2FComeOnOliver%2Fclaude%2Dcode%2Danalysis&amp;rut=1b6530689ef48e5b87739e402f708022c7e44a4799ea88b1013818ac2311bc53; //duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.aikido.dev%2Fblog%2Fcode%2Danalysis%2Dtools&amp;rut=fb47856a7a2da56f780a235372b392bf754f6f18ee0283cd52c8256d883b6994
confidence: partial
access_count: 25
---

# source code analysis

## Что это
Автоматизированный анализ исходного кода для выявления уязвимостей безопасности, багов и проблем качества без выполнения программы. Также известен как Static Application Security Testing (SAST).

## Ключевые параметры
- **Типы анализа**: статический (SAST), динамический (DAST), интерактивный (IAST)
- **Охват языков**: зависит от инструмента (Java, C/C++, Python, JavaScript, Go и др.)
- **Точность**: варьируется от 20% до 80% в зависимости от инструмента
- **False positive rate**: может достигать 30-50% у базовых инструментов
- **Время сканирования**: от минут до часов в зависимости от размера кодовой базы

## Практические заметки
- Интегрировать в CI/CD pipeline на ранних этапах разработки
- Комбинировать несколько инструментов для лучшего покрытия
- Настраивать правила под специфику проекта для снижения ложных срабатываний
- Использовать инкрементальный анализ для больших проектов
- Приоритизировать критические уязвимости (OWASP Top 10)
- Обучать команду интерпретации результатов

## Частые ошибки
- **Игнорирование контекста**: инструменты не понимают бизнес-логику — требуется ручная верификация
- **Перегрузка алертами**: слишком много предупреждений без приоритизации парализует команду
- **Запуск только перед релизом**: анализ должен быть непрерывным процессом
- **Полагаться только на автоматизацию**: code review человеком остается критичным
- **Неправильная конфигурация**: дефолтные настройки редко оптимальны для конкретного проекта
- **Отсутствие baseline**: нужно установить начальную метрику для отслеживания прогресса

## Связанные темы
- [[SAST]]
- [[OWASP Top 10]]
- [[CI/CD security]]
- [[code review]]
- [[vulnerability management]]
- [[static analysis]]
- [[security testing]]
