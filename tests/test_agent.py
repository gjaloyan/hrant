"""Тесты агентского цикла с замоканным router и web-поиском."""
import json
from unittest.mock import patch

from backend.agent import Agent
from backend.finetune import store as finetune_store
from backend.llm import TaskType


class FakeRouter:
    """Мок DualModelRouter: возвращает заранее заданные ответы по TaskType.

    classification_queue позволяет последовательно возвращать разные ответы
    для двух LLM-вызовов подряд (сперва intent-классификатор, затем
    preference-экстрактор — оба используют TaskType.CLASSIFICATION).
    """

    def __init__(
        self,
        analyze_json: dict,
        solve_text: str,
        verify_json: dict,
        *,
        chat_text: str = "",
        intent: str = "task",
        classification_queue: list[dict] | None = None,
    ):
        self.analyze_json = analyze_json
        self.solve_text = solve_text
        self.verify_json = verify_json
        self.chat_text = chat_text
        self.intent = intent
        self.classification_queue = list(classification_queue or [])
        self.calls: list[TaskType] = []
        # last system prompts per TaskType — для проверки identity preamble
        self.last_system: dict[TaskType, str] = {}

    def call(self, task_type: TaskType, system: str, user: str, **kw) -> str:
        self.calls.append(task_type)
        self.last_system[task_type] = system
        if task_type == TaskType.VERIFICATION:
            return json.dumps(self.verify_json)
        if task_type == TaskType.COMPLEX_SOLVING:
            return self.solve_text
        if task_type == TaskType.QUICK_ANSWER:
            return self.chat_text
        if task_type == TaskType.NOTE_CREATION:
            return "# fake note\nfake body"
        return ""

    def call_with_tools(self, task_type: TaskType, system: str, user: str,
                        tools=None, execute_tool=None, **kw) -> str:
        """В тестах не имитируем настоящий tool-loop — просто возвращаем
        тот же ответ, что и обычный call(). Если тесту нужна проверка
        конкретного поведения tool-use, см. tests/test_tools.py."""
        return self.call(task_type, system, user, **kw)

    def call_json(self, task_type: TaskType, system: str, user: str, **kw) -> dict:
        self.calls.append(task_type)
        self.last_system[task_type] = system
        if task_type == TaskType.TASK_ANALYSIS:
            return self.analyze_json
        if task_type == TaskType.VERIFICATION:
            return self.verify_json
        if task_type == TaskType.CLASSIFICATION:
            if self.classification_queue:
                return self.classification_queue.pop(0)
            return {"intent": self.intent, "reason": "test"}
        return {}


def test_agent_uses_existing_note(tmp_kb):
    tmp_kb.save_note(
        topic="RS-485",
        body="## Что это\nдифференциальная шина 12Мбит/с макс 1200м",
        category="profession",
        keywords=["rs485", "serial"],
        source="test",
    )

    fake = FakeRouter(
        analyze_json={"required_topics": ["RS-485"], "plan": ["ответить"], "confidence": 90},
        solve_text="RS-485 — дифференциальная шина. [RS-485]",
        verify_json={
            "confidence": 95,
            "verified_claims": ["дифференциальная шина"],
            "unverified_claims": [],
            "contradictions": [],
            "notes_used": ["RS-485"],
        },
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        with patch("backend.agent.learn_topic") as mock_learn:
            agent = Agent()
            res = agent.run("расскажи про RS-485")
            mock_learn.assert_not_called()

    assert res.verification.confidence == 95
    assert "RS-485" in res.used_topics
    assert not res.learned_topics

    # Маршрутизация: должны быть вызваны TASK_ANALYSIS → COMPLEX_SOLVING → VERIFICATION
    assert TaskType.TASK_ANALYSIS in fake.calls
    assert TaskType.COMPLEX_SOLVING in fake.calls
    assert TaskType.VERIFICATION in fake.calls

    # confidence 95 ≥ 85 + verified + source_notes → должно попасть в finetune queue
    assert finetune_store().count() == 1
    pair = finetune_store().list_all()[0]
    assert "RS-485" in pair.metadata.source_notes
    assert pair.metadata.confidence == 95


def test_agent_learns_new_topic(tmp_kb):
    fake = FakeRouter(
        analyze_json={"required_topics": ["DS18B20"], "plan": ["изучить", "ответить"], "confidence": 70},
        solve_text="DS18B20 — цифровой датчик температуры. [DS18B20]",
        verify_json={
            "confidence": 80,
            "verified_claims": ["цифровой датчик"],
            "unverified_claims": [],
            "contradictions": [],
            "notes_used": ["DS18B20"],
        },
    )

    def fake_learn(topic, **kw):
        return tmp_kb.save_note(
            topic=topic,
            body=f"# {topic}\nцифровой датчик температуры",
            category="profession",
            keywords=[topic.lower()],
            source="fake_web",
            confidence="partial",
        )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic", side_effect=fake_learn):
        agent = Agent()
        res = agent.run("как подключить DS18B20?")

    assert "DS18B20" in res.learned_topics
    assert tmp_kb.get_note("DS18B20") is not None


def test_agent_greeting_takes_chat_shortcut(tmp_kb):
    """«hello» должно обрабатываться коротко: без analyze/learn/verify."""
    fake = FakeRouter(
        analyze_json={"required_topics": [], "plan": [], "confidence": 0},
        solve_text="should not be called",
        verify_json={"confidence": 0},
        chat_text="Привет! Чем могу помочь?",
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic") as mock_learn:
        agent = Agent()
        res = agent.run("hello")
        mock_learn.assert_not_called()

    assert res.is_chat is True
    assert res.answer == "Привет! Чем могу помочь?"
    assert res.used_topics == []
    assert res.learned_topics == []
    assert res.verification.confidence == 100

    # Критично: полный цикл не запускался.
    assert TaskType.TASK_ANALYSIS not in fake.calls
    assert TaskType.COMPLEX_SOLVING not in fake.calls
    assert TaskType.VERIFICATION not in fake.calls
    # И finetune queue не должен расти от болтовни.
    assert finetune_store().count() == 0


def test_agent_chat_fallback_when_api_down(tmp_kb):
    """When API is down, chat should return a warm fallback instead of error."""
    from backend.llm import LLMError

    class DownRouter:
        def call(self, *a, **kw):
            raise LLMError("529 overloaded")
        def call_json(self, task_type, *a, **kw):
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "chat"}
            raise LLMError("529 overloaded")
        def call_with_tools(self, *a, **kw):
            raise LLMError("529 overloaded")

    with patch("backend.agent.router", return_value=DownRouter()), \
         patch("backend.verifier.router", return_value=DownRouter()):
        agent = Agent()
        res = agent.run("привет")

    assert res.is_chat is True
    assert res.verification.confidence == 100
    # Chat path now surfaces the actual LLM error short instead of a hardcoded
    # warm fallback — easier to debug what's wrong than to be reassured.
    assert res.answer.startswith("⚠")
    assert "529" in res.answer or "overloaded" in res.answer
    # Should NOT be the old generic "Ошибка LLM" prefix
    assert "Ошибка LLM" not in res.answer


def test_agent_uses_llm_classifier_for_ambiguous_input(tmp_kb):
    """Сообщения, не попавшие в regex, должны классифицироваться LLM-ом."""
    fake = FakeRouter(
        analyze_json={"required_topics": [], "plan": [], "confidence": 0},
        solve_text="should not be called",
        verify_json={"confidence": 0},
        chat_text="У меня всё отлично, спасибо!",
        intent="chat",  # LLM-классификатор говорит «это болтовня»
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic") as mock_learn:
        agent = Agent()
        # Фраза не из regex — значит пойдёт в LLM-классификатор
        res = agent.run("а у тебя как настроение сегодня?")
        mock_learn.assert_not_called()

    assert res.is_chat is True
    assert TaskType.CLASSIFICATION in fake.calls
    assert TaskType.TASK_ANALYSIS not in fake.calls


def test_agent_preference_saves_to_user_md_not_knowledge(tmp_kb):
    """«отвечай на русском» должно попасть в user.md, а не в заметки."""
    from backend.identity import IDENTITY
    fake = FakeRouter(
        analyze_json={"required_topics": [], "plan": [], "confidence": 0},
        solve_text="should not be called",
        verify_json={"confidence": 0},
        classification_queue=[
            # 1-й CLASSIFICATION call — intent-классификатор
            {"intent": "preference", "reason": "user sets language"},
            # 2-й — preference-экстрактор
            {
                "category": "language",
                "fact": "Пользователь предпочитает русский язык",
                "acknowledgment": "Понял, буду отвечать по-русски.",
            },
        ],
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic") as mock_learn:
        agent = Agent()
        res = agent.run("answer me in russian always, remember it")
        mock_learn.assert_not_called()

    # Ответ — это is_chat-подобный короткий ack, без футера верификации.
    assert res.is_chat is True
    assert "по-русски" in res.answer.lower() or "русск" in res.answer.lower()
    assert res.used_topics == []
    assert res.learned_topics == []

    # Критично: полный цикл не запускался.
    assert TaskType.TASK_ANALYSIS not in fake.calls
    assert TaskType.COMPLEX_SOLVING not in fake.calls
    assert TaskType.VERIFICATION not in fake.calls

    # И никакой темы «русский язык» в базе знаний быть не должно.
    assert tmp_kb.get_note("русский язык") is None
    assert tmp_kb.get_note("русский") is None

    # Зато факт должен быть в user.md в разделе «Язык общения».
    user_md = IDENTITY.user_profile()
    assert "Пользователь предпочитает русский язык" in user_md
    assert "## Язык общения" in user_md
    # И не должно остаться плейсхолдера в этой секции.
    lang_section = user_md.split("## Язык общения", 1)[1].split("##", 1)[0]
    assert "(пока не указано)" not in lang_section

    # finetune queue тоже не должен расти от предпочтений.
    assert finetune_store().count() == 0


def test_solver_system_prompt_includes_identity_preamble(tmp_kb):
    """Полный цикл должен передавать SOUL/IDENTITY/USER PROFILE в system."""
    from backend.identity import IDENTITY
    IDENTITY.add_user_fact("Пользователь предпочитает краткие ответы", category="style")

    tmp_kb.save_note(
        topic="Ohm's law",
        body="U = I * R",
        category="fundamentals",
        keywords=["ohm"],
        source="test",
    )
    fake = FakeRouter(
        analyze_json={"required_topics": ["Ohm's law"], "plan": ["ответить"], "confidence": 90},
        solve_text="U = I * R [Ohm's law]",
        verify_json={
            "confidence": 95,
            "verified_claims": ["U = I * R"],
            "unverified_claims": [],
            "contradictions": [],
            "notes_used": ["Ohm's law"],
        },
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("what is Ohm's law?")

    solver_system = fake.last_system.get(TaskType.COMPLEX_SOLVING, "")
    assert "# SOUL" in solver_system
    assert "# IDENTITY" in solver_system
    assert "# USER PROFILE" in solver_system
    assert "Пользователь предпочитает краткие ответы" in solver_system
    # Solver base prompt должен присутствовать тоже
    assert "KNOWLEDGE PRIORITY" in solver_system


def test_solver_injects_matching_skill_block(tmp_kb):
    """Если запрос содержит триггер скилла — его блок попадает в system."""
    tmp_kb.save_note(
        topic="Ohm's law",
        body="U = I * R",
        category="fundamentals",
        keywords=["ohm"],
        source="test",
    )
    fake = FakeRouter(
        analyze_json={"required_topics": ["Ohm's law"], "plan": ["ответить"], "confidence": 90},
        solve_text="765 [calc]",
        verify_json={
            "confidence": 95,
            "verified_claims": [],
            "unverified_claims": [],
            "contradictions": [],
            "notes_used": [],
        },
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        # «посчитай» — триггер calc-скилла
        agent.run("посчитай 4500 * 0.17")

    solver_system = fake.last_system.get(TaskType.COMPLEX_SOLVING, "")
    assert "AVAILABLE SKILLS" in solver_system
    assert "SKILL: calc" in solver_system  # активный блок скилла
    assert "calc(expression" in solver_system or "calc(" in solver_system or "calc" in solver_system
