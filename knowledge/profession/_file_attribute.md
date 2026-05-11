---
topic: __file__ attribute
category: profession
created: 2026-04-08 14:26
updated: 2026-04-08 14:26
keywords: __file__, Python module attribute, pathname, module path, global namespace, extension modules, C modules, dynamic file paths, module location, special variables
source: //duckduckgo.com/l/?uddg=https%3A%2F%2Fpytutorial.com%2Fpython%2D__file__%2Dattribute%2Dguide%2Dexamples%2F&amp;rut=3831ea69f053d51b270ffac6a5d5fce4de83f00b7bc7793fe2846c21d826244f; //duckduckgo.com/l/?uddg=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F9271464%2Fwhat%2Ddoes%2Dthe%2Dfile%2Dvariable%2Dmean%2Ddo&amp;rut=7ac0f927acbef2f34f46528d33ece8e5daed58a0ed997bd3af088911602715e8; //duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.geeksforgeeks.org%2Fpython%2F__file__%2Da%2Dspecial%2Dvariable%2Din%2Dpython%2F&amp;rut=50b9b4d3f2e2fbe7fe0b205ed7e18da498d563524e1e95a0ee147985cb98dc18
confidence: partial
access_count: 42
---

# __file__ attribute

## Что это
Специальная переменная-атрибут модуля в Python, содержащая путь к файлу текущего модуля. Представляет собой строку (String) с pathname файла, из которого был загружен модуль.

## Ключевые параметры
- **Тип данных**: строка (String)
- **Область видимости**: глобальное пространство имён модуля
- **Доступность**: 
  - Присутствует для модулей, загруженных из .py файлов
  - Отсутствует для C-модулей, статически слинкованных в интерпретатор
  - Для динамически загруженных extension-модулей содержит путь к shared library файлу

## Практические заметки
- Используется для определения местоположения модуля в файловой системе
- Позволяет строить динамические пути к файлам относительно текущего модуля
- Доступен через `__file__` в глобальном namespace модуля
- Полезен для загрузки ресурсов, находящихся рядом с модулем

## Частые ошибки
- **AttributeError при обращении**: `__file__` отсутствует в интерактивной сессии Python и для статически слинкованных C-модулей
- **Относительные vs абсолютные пути**: значение может быть как относительным, так и абсолютным путём в зависимости от способа импорта
- **Использование в __main__**: при запуске скрипта напрямую `__file__` может содержать относительный путь

## Связанные темы
- [[__name__ attribute]]
- [[module import]]
- [[os.path]]
- [[pathlib]]
- [[sys.modules]]
- [[__main__ module]]
