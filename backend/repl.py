"""Interactive REPL for the agent.

Used by both:
  -  (root-level shim re-exports main from here)
  -  (backend.cli dispatcher imports backend.repl.main)

The legacy invocation
    python cli.py                    # interactive chat
    python cli.py 'твой вопрос'       # one-shot query
is preserved by the thin shim at the repo root.
"""
from __future__ import annotations
import sys

from backend.agent import Agent
from backend.commands import parse
from backend.config import CONFIG
from backend.core_memory import CORE
from backend.finetune import store as finetune_store
from backend.finetune_curator import FinetuneDataCurator
from backend.knowledge_manager import KM
from backend.llm import router as _router_fn
from backend.model_versions import VERSIONS
from backend.models import AgentAnswer
from backend.note_creator import learn_topic
from backend.project_mode import PROJECTS


HELP = """\
Команды (на русском или английском):

  Память:
    запомни <факт>                  — добавить в core memory
    забудь про <текст>              — удалить из core memory
    покажи core memory              — показать core
    что ты знаешь?                  — список всех тем
    что ты знаешь о <тема>?         — показать заметку
    удали знания о <тема>           — удалить заметку
    изучи <тема>                    — быстрое исследование
    изучи глубоко <тема>            — подробное исследование
    краткая заметка <текст>         — без исследования

  Проекты:
    начать проект <имя>
    завершить проект
    контекст проекта <текст>
    решили <что> потому что <почему>
    проблема: <проблема> → <решение>

  Fine-tuning:
    finetune status                  — статистика очереди + готовность
    finetune review                  — список curated-примеров
    finetune start                   — запустить полный пайплайн (local_full / local_cpu)
    finetune export-cloud            — пакет для запуска на арендованной GPU (cloud_finetune)
    finetune import-gguf <путь> [--tag vN] — импорт gguf обратно в Ollama
    finetune compare                 — сравнить старую и новую модель
    finetune switch <tag>            — переключить версию
    finetune rollback                — откат
    finetune export                  — экспорт сырого jsonl
    model versions                   — список версий
    learn this to model              — добавить последний Q&A в очередь
    неправильно, правильно: <текст>  — записать correction
    mode                             — показать текущий режим работы

  Прочее:
    статус                          — общая статистика
    пробелы / gaps                  — темы, которых не было в базе в момент вопроса
    whoami / capabilities           — мои инструменты, навыки, исходный код
    help / помощь                   — эта справка
    exit / quit                     — выход

Всё остальное обрабатывается как обычный вопрос — агент запускает полный цикл.
"""


def _progress(event: str, msg: str) -> None:
    markers = {
        "core": "💾",
        "think": "🧠",
        "strategy": "📊",
        "analyze": "🧠",
        "chat": "💬",
        "preference": "🎯",
        "preference_saved": "📝",
        "found": "✓",
        "learning": "📖",
        "learned": "📚",
        "solve": "✍️ ",
        "skill": "⚡",
        "tool": "🔧",
        "tool_error": "🔧⚠️",
        "verify": "🔍",
        "experience": "💡",
        "cleanup": "🧹",
        "error": "⚠️ ",
    }
    print(f"  {markers.get(event, '·')} {msg}", flush=True)


def _print_answer(res: AgentAnswer) -> None:
    # Лёгкий вывод для small-talk: без рамки и без футера с верификацией.
    if res.is_chat:
        print(f"\n{res.answer}\n")
        return

    print("\n" + "=" * 60)
    print(res.answer)
    print("=" * 60)
    vr = res.verification
    badge = "🟢" if vr.confidence >= 90 else ("🟡" if vr.confidence >= 70 else "🔴")
    print(f"{badge} уверенность: {vr.confidence}%")
    if res.used_topics:
        print(f"📂 использованы темы: {', '.join(res.used_topics)}")
    if res.learned_topics:
        print(f"📚 изучено нового: {', '.join(res.learned_topics)}")
    if vr.unverified_claims:
        print(f"⚠️  не подтверждено: {len(vr.unverified_claims)} утверждений")
        for c in vr.unverified_claims[:5]:
            print(f"    · {c}")
    if vr.contradictions:
        print(f"❌ противоречия: {len(vr.contradictions)}")
        for c in vr.contradictions[:5]:
            print(f"    · {c}")
    print()


def handle_command(text: str, agent: Agent, last: dict) -> bool:
    """Возвращает True если команда обработана (без запуска агента)."""
    cmd = parse(text)
    k = cmd.kind

    if k == "help":
        print(HELP); return True

    if k == "status":
        topics = KM.list_topics()
        cats: dict[str, int] = {}
        for t in topics:
            cats[t.category] = cats.get(t.category, 0) + 1
        ft = finetune_store()
        cur = VERSIONS.current()
        print(f"📊 статус:")
        print(f"   🎛  mode: {CONFIG.mode} (training: {CONFIG.training_location})")
        print(f"   тем всего: {len(topics)}")
        for c, n in cats.items():
            print(f"   · {c}: {n}")
        print(f"   core memory: ~{CORE.tokens()} токенов / {CORE.max_tokens}")
        print(f"   finetune queue: {ft.count()} пар (готовность: {'да' if ft.ready() else 'нет'})")
        print(f"   finetune version: {cur.tag if cur else '—'} ({cur.model_id if cur else '—'})")
        print(f"   текущий проект: {PROJECTS.current or '—'}")
        print(f"   🧠 Model A: {CONFIG.model_a.get('model')}")
        print(f"   📖 Model B: {CONFIG.model_b.get('model')}")
        try:
            rs = _router_fn().stats()
            budget = rs.get("budget_usd", 0) or 0
            print(
                f"   today: A={rs['api_calls_today']} (${rs['api_cost_today']:.3f}"
                f"{f'/${budget}' if budget else ''}) / B={rs['model_b_calls_today']}"
            )
            print(f"   total: A={rs['total_a_calls']} / B={rs['total_b_calls']}")
            if rs.get("last_reason"):
                print(f"   last routing: {rs['last_reason']}")
        except Exception as e:
            print(f"   router: ошибка ({e})")
        return True

    if k == "show_core":
        print("\n" + CORE.read() + "\n"); return True

    if k == "gaps":
        gaps = KM.hot_gaps(threshold=1)
        if not gaps:
            print("(пробелов нет — все запрошенные темы были в базе)")
            return True
        open_ = [g for g in gaps if not g["has_note_now"]]
        closed = [g for g in gaps if g["has_note_now"]]
        print(f"\n📍 Пробелы знаний (всего {len(gaps)}):")
        if open_:
            print(f"\n  ⚠️  ещё не изучено ({len(open_)}):")
            for g in open_:
                print(f"    · {g['topic']}  ×{g['count']}  ({g['last']})")
        if closed:
            print(f"\n  ✓ закрыто on-demand ({len(closed)}):")
            for g in closed[:10]:
                print(f"    · {g['topic']}  ×{g['count']}")
        print()
        return True

    if k == "whoami":
        from backend.agent import _capabilities_block
        print()
        print(_capabilities_block())
        print()
        return True

    if k == "graph":
        from backend.knowledge_graph import GRAPH, reindex_all_notes
        s = GRAPH.stats()
        print(f"Knowledge Graph: {s['entities']} entities, {s['edges']} edges, {s['notes_indexed']} notes")
        if cmd.arg == "reindex":
            print("Reindexing all notes...")
            r = reindex_all_notes()
            print(f"  indexed {r['notes']} notes, {r['triples']} triples, {r['errors']} errors")
            s = GRAPH.stats()
            print(f"  now: {s['entities']} entities, {s['edges']} edges")
        return True

    if k == "remember":
        print(CORE.add_fact(cmd.arg)); return True

    if k == "forget":
        print(CORE.remove_fact(cmd.arg)); return True

    if k == "list":
        topics = KM.list_topics()
        if not topics:
            print("(база знаний пуста)"); return True
        by_cat: dict[str, list] = {}
        for t in topics:
            by_cat.setdefault(t.category, []).append(t)
        for cat, items in by_cat.items():
            print(f"\n[{cat}]")
            for t in sorted(items, key=lambda x: -x.access_count):
                print(f"  · {t.topic}  (×{t.access_count})")
        return True

    if k == "show":
        note = KM.get_note(cmd.arg)
        if not note:
            print(f"(нет заметки о '{cmd.arg}')"); return True
        print(f"\n--- {note.frontmatter.topic} ---")
        print(f"категория: {note.frontmatter.category}")
        print(f"источник: {note.frontmatter.source}")
        print(f"уверенность: {note.frontmatter.confidence}")
        print(f"обращений: {note.frontmatter.access_count}\n")
        print(note.body); print()
        return True

    if k == "delete":
        ok = KM.delete_note(cmd.arg)
        print("✓ удалено" if ok else "✗ не найдено")
        return True

    if k in ("learn", "learn_deep"):
        depth = "deep" if k == "learn_deep" else "quick"
        print(f"📖 изучаю '{cmd.arg}' ({depth})...")
        try:
            note = learn_topic(cmd.arg, depth=depth, project=PROJECTS.current)
            print(f"✓ создана заметка: {note.path}")
        except Exception as e:
            print(f"✗ ошибка: {e}")
        return True

    if k == "quick_note":
        note = KM.save_note(
            topic=cmd.arg[:40],
            body=cmd.arg,
            category="personal",
            keywords=[cmd.arg.split()[0].lower()] if cmd.arg else [],
            source="user_quick_note",
            confidence="verified",
        )
        print(f"✓ сохранено: {note.frontmatter.topic}")
        return True

    if k == "start_project":
        print(PROJECTS.start(cmd.arg)); return True
    if k == "end_project":
        print(PROJECTS.end()); return True
    if k == "project_ctx":
        print(PROJECTS.add_context(cmd.arg)); return True
    if k == "decision":
        print(PROJECTS.add_decision(cmd.arg, cmd.arg2)); return True
    if k == "issue":
        print(PROJECTS.add_issue(cmd.arg, cmd.arg2)); return True

    if k == "ft_status":
        ft = finetune_store()
        examples = ft.list_all()
        curated = FinetuneDataCurator().curate(examples)
        print(f"📊 finetune:")
        print(f"   всего: {len(examples)}")
        print(f"   после курации: {len(curated)}")
        print(f"   порог готовности: {ft.min_required}")
        print(f"   готов к обучению: {'✓' if ft.ready() else '✗'}")
        for cat, n in ft.count_by_category().items():
            print(f"   · {cat}: {n}")
        return True

    if k == "ft_review":
        examples = finetune_store().list_all()
        curator = FinetuneDataCurator()
        for sp in curator.score_all(examples):
            tag = "✓" if sp.score >= curator.MIN_SCORE else "·"
            print(f"  {tag} [{sp.pair.id}] {sp.pair.metadata.category:14} score={sp.score:.2f}")
            print(f"     Q: {sp.pair.user_text()[:80]}")
        return True

    if k == "ft_export":
        out = finetune_store().export_jsonl()
        print(out or "(пусто)")
        return True

    if k == "ft_versions":
        state = VERSIONS.list()
        print(f"current: {state.current}")
        for v in state.versions:
            mark = "→" if v.tag == state.current else " "
            print(f"  {mark} {v.tag}: {v.model_id}  ({v.examples_count} ex, {v.created})")
        return True

    if k == "ft_switch":
        print(VERSIONS.switch(cmd.arg))
        return True

    if k == "ft_rollback":
        print(VERSIONS.rollback())
        return True

    if k == "ft_start":
        from backend.finetune_pipeline import FineTunePipeline

        def _ftprog(stage: str, msg: str) -> None:
            print(f"  · [{stage}] {msg}")

        pipe = FineTunePipeline(progress=_ftprog)
        try:
            name = pipe.run_full_pipeline()
            print(f"✓ модель {name} готова")
        except Exception as e:
            print(f"✗ {e}")
        return True

    if k == "ft_export_cloud":
        from backend.finetune_pipeline import FineTunePipeline

        def _ftprog(stage: str, msg: str) -> None:
            print(f"  · [{stage}] {msg}")

        pipe = FineTunePipeline(progress=_ftprog)
        try:
            pkg = pipe.export_for_cloud()
            print(f"✓ пакет готов: {pkg}")
            print(f"  файлы: {', '.join(sorted(p.name for p in pkg.iterdir()))}")
            print(f"  дальше: залей папку на GPU, запусти train_script.py")
            print(f"  потом:  finetune import-gguf <путь_к_unsloth.Q4_K_M.gguf>")
        except Exception as e:
            print(f"✗ {e}")
        return True

    if k == "ft_import_gguf":
        from backend.finetune_pipeline import FineTunePipeline

        def _ftprog(stage: str, msg: str) -> None:
            print(f"  · [{stage}] {msg}")

        pipe = FineTunePipeline(progress=_ftprog)
        try:
            name = pipe.import_gguf(cmd.arg, tag=cmd.arg2 or None)
            print(f"✓ зарегистрирована: {name}")
        except Exception as e:
            print(f"✗ {e}")
        return True

    if k == "show_mode":
        print(f"🎛  режим работы: {CONFIG.mode}")
        print(f"    finetune enabled: {CONFIG.finetune_enabled}")
        print(f"    training location: {CONFIG.training_location}")
        print(f"    Model A: {CONFIG.model_a.get('model')}")
        print(f"    Model B: {(CONFIG.model_b or {}).get('model') if CONFIG.model_b else '— (выключен)'}")
        print()
        print("    Доступные режимы: local_full | cloud_finetune | local_cpu | claude_only")
        print("    Чтобы сменить — отредактируй 'mode:' в config.yaml и перезапусти.")
        return True

    if k == "ft_compare":
        from backend.model_evaluator import ModelEvaluator

        state = VERSIONS.list()
        if len(state.versions) < 2:
            print("✗ нужно минимум 2 версии модели")
            return True
        old = state.versions[-2].model_id
        new = state.versions[-1].model_id
        print(f"сравниваю {old} vs {new}...")
        res = ModelEvaluator().compare(old, new)
        print(f"  old: {res.old_score:.3f}")
        print(f"  new: {res.new_score:.3f}")
        print(f"  improvement: {res.improvement:+.3f}")
        print(f"  upgrade recommended: {'да' if res.should_upgrade else 'нет'}")
        return True

    if k == "ft_learn_this":
        if "result" not in last:
            print("(ещё не было ответов)")
            return True
        if "task" not in last:
            print("(нет последнего вопроса)")
            return True
        res: AgentAnswer = last["result"]
        finetune_store().add(
            question=last["task"],
            answer=res.answer,
            source_notes=res.used_topics,
            confidence=max(res.verification.confidence, 100),
            project=res.project,
            verified=True,
        )
        print("✓ Q&A добавлено в finetune queue")
        return True

    if k == "ft_correct":
        if "result" not in last or "task" not in last:
            print("(нет последнего ответа для коррекции)")
            return True
        corrected = cmd.arg
        finetune_store().add_correction(
            question=last["task"],
            wrong_answer=last["result"].answer,
            corrected_answer=corrected,
            project=last["result"].project,
            source_notes=last["result"].used_topics,
        )
        print("✓ correction записан в finetune queue")
        return True

    if k == "verify_last":
        if "result" not in last:
            print("(ещё не было ответов)"); return True
        _print_answer(last["result"])
        return True

    if k == "confidence":
        if "result" not in last:
            print("(ещё не было ответов)"); return True
        vr = last["result"].verification
        print(f"уверенность: {vr.confidence}%")
        print(f"  подтверждено: {len(vr.verified_claims)}")
        print(f"  не подтверждено: {len(vr.unverified_claims)}")
        print(f"  противоречия: {len(vr.contradictions)}")
        return True

    return False


def main() -> None:
    agent = Agent(progress=_progress)
    last: dict = {}

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        if not handle_command(task, agent, last):
            res = agent.run(task, project=PROJECTS.current)
            last["task"] = task
            last["result"] = res
            _print_answer(res)
        return

    print("🧠 Self-Learning Agent — CLI MVP")
    print("   help — справка   |   exit — выход")
    if PROJECTS.current:
        print(f"   активный проект: {PROJECTS.current}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nпока.")
            return
        if not text:
            continue
        if text.lower() in ("exit", "quit", "выход"):
            return
        if handle_command(text, agent, last):
            continue
        res = agent.run(text, project=PROJECTS.current)
        last["task"] = text
        last["result"] = res
        _print_answer(res)


if __name__ == "__main__":
    main()
