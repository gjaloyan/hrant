---
topic: recursive_reasoning
category: profession
created: 2026-04-13 11:44
updated: 2026-04-13 11:44
keywords: recursive reasoning, tiny networks, TRM, HRM, ARC-AGI, 7M parameters, small models, puzzle solving, hierarchical reasoning, parameter efficiency
source: https://arxiv.org/abs/2510.04871; https://medium.com/data-science-in-your-pocket/less-is-more-recursive-reasoning-with-tiny-networks-paper-explained-a4573708376d; https://github.com/SamsungSAILMontreal/TinyRecursiveModels
confidence: partial
access_count: 88
---

# recursive_reasoning

## Что это

Подход к решению сложных задач через рекурсивное применение малых нейросетей вместо больших языковых моделей. Tiny Recursive Model (TRM) — реализация с одной крошечной сетью из 2 слоёв, превосходящая LLM на задачах типа ARC-AGI, Sudoku, Maze. Hierarchical Reasoning Model (HRM) — предшественник с двумя сетями, рекурсирующими на разных частотах.

## Ключевые параметры

- **TRM**: 7M параметров, 2 слоя
- **HRM**: 27M параметров, две сети
- **Данные для обучения**: ~1000 примеров
- **Точность TRM на ARC-AGI-1**: 45%
- **Точность TRM на ARC-AGI-2**: 8%
- **Соотношение параметров**: <0.01% от размера сравниваемых LLM

## Практические заметки

- TRM показывает значительно лучшую генерализацию чем HRM при меньшем размере
- Превосходит большинство LLM (DeepSeek R1, o3-mini, Gemini 2.5 Pro) на задачах-головоломках
- Биологически инспирированный подход — рекурсия на разных уровнях абстракции
- Эффективен для hard reasoning tasks при минимальных вычислительных ресурсах
- Обучается на малых датасетах (~1000 примеров)

## Частые ошибки

- HRM может быть субоптимальным — TRM проще и эффективнее
- Подход пока недостаточно изучен, механизмы работы требуют дополнительного исследования
- Не стоит полагаться исключительно на массивные foundation models для сложных задач

## Связанные темы

- [[arc_agi]]
- [[small_language_models]]
- [[reasoning_models]]
- [[hierarchical_models]]
- [[parameter_efficiency]]
- [[puzzle_solving]]
