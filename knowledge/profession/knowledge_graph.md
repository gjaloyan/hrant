---
topic: knowledge_graph
category: profession
created: 2026-04-14 09:59
updated: 2026-04-14 09:59
keywords: knowledge graph, graph database, RDF, triples, ontology, semantic enrichment, entity relationships, GraphRAG, Neo4j, reasoning and inference
source: https://en.wikipedia.org/wiki/Knowledge_graph; https://www.geeksforgeeks.org/data-analysis/what-is-a-knowledge-graph/; https://neo4j.com/blog/knowledge-graph/how-to-build-knowledge-graph/
confidence: partial
access_count: 80
---

# knowledge_graph

## Что это
Граф знаний (Knowledge Graph) — структурированная база данных, использующая графовую модель для представления сущностей (entities) и связей между ними. Данные организованы в формате троек (subject, predicate, object), например: (Париж, isCapitalOf, Франция). Основан на Resource Description Framework (RDF).

## Ключевые параметры
**Базовые компоненты:**
- Nodes (узлы) — сущности с атрибутами (человек, компания, продукт)
- Edges (рёбра) — связи между узлами с метками типа "employed by", "located in"
- Properties/Labels — характеристики узлов и рёбер
- Triples — формат (субъект, предикат, объект)

**Технологические элементы:**
- Ontology — формальная схема концепций и отношений в домене (опциональна, но полезна для сложных доменов)
- Semantic enrichment — обогащение через NLP для понимания контекста
- Reasoning and inference — вывод новых знаний из существующих связей

## Практические заметки
**7 шагов построения:**
1. Определить use case (поиск, рекомендации, fraud detection)
2. Создать модель данных (ontology)
3. Определить источники данных
4. Загрузить данные (data ingestion)
5. Обогатить семантически
6. Настроить запросы и визуализацию
7. Итеративно улучшать

**Реальные применения:**
- Enterprise Search + GenAI — GraphRAG для точных ответов на основе структурированных знаний
- Fraud Detection — выявление подозрительных паттернов в финансовых транзакциях
- Master Data Management — единое представление клиентов/продуктов из разных систем
- Supply Chain — визуализация потоков товаров и логистики
- NLP — улучшение семантического поиска через связывание сущностей в тексте

**Примеры использования:**
- Google Knowledge Graph — релевантные результаты поиска через понимание связей
- Recommendation Systems — предложения на основе графа связей

## Частые ошибки
- Построение без чёткого use case — граф становится неуправляемым
- Игнорирование ontology в сложных доменах — теряется формальная семантика и возможность reasoning
- Смешивание "Apple" (фрукт) и "Apple" (компания) — требуется disambiguation через семантическое обогащение
- Отсутствие де-дупликации сущностей при интеграции из разных источников

## Причинно-следственные связи
- Графовая структура данных → возможность inference (вывода неявных знаний)
- Использование троек (RDF) → стандартизация представления знаний
- Semantic enrichment через NLP → disambiguation сущностей и понимание контекста
- Ontology как схема → формальное reasoning в сложных доменах
- Связывание разрозненных источников → единое представление Master Data
- Визуализация графа → упрощение аналитики и поиска паттернов
- Интеграция с LLM (GraphRAG) → точные, объяснимые ответы AI-систем

## Связанные темы
- [[graph_database]]
- [[neo4j]]
- [[rdf]]
- [[ontology]]
- [[semantic_web]]
- [[nlp]]
- [[entity_disambiguation]]
- [[graphrag]]
- [[cypher]]
- [[master_data_management]]
- [[recommendation_systems]]
- [[fraud_detection]]
