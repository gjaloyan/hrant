"""Разбор естественно-языковых команд управления памятью и проектами."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedCommand:
    kind: str              # remember, forget, learn, list, show, delete, show_core,
                           # start_project, end_project, project_ctx, decision, issue,
                           # verify_last, finetune_status, quick_note, none
    arg: str = ""
    arg2: str = ""


PATTERNS: list[tuple[str, re.Pattern]] = [
    ("remember",        re.compile(r"^\s*(?:запомни(?:\s+навсегда)?|remember(?:\s+forever)?)[:\s]+(.+)$", re.I)),
    ("forget",          re.compile(r"^\s*(?:забудь(?:\s+про)?|forget(?:\s+about)?)[:\s]+(.+)$", re.I)),
    ("show_core",       re.compile(r"^\s*(?:покажи\s+core\s*memory|show\s+core\s*memory|core)\s*$", re.I)),
    ("list",            re.compile(r"^\s*(?:что\s+ты\s+знаешь\??|what\s+do\s+you\s+know\??|list\s+topics)\s*$", re.I)),
    ("show",            re.compile(r"^\s*(?:что\s+ты\s+знаешь\s+о|what\s+do\s+you\s+know\s+about)\s+(.+?)\??\s*$", re.I)),
    ("delete",          re.compile(r"^\s*(?:удали\s+(?:знания\s+о\s+)?|delete\s+knowledge\s+about\s+)(.+)$", re.I)),
    ("learn_deep",      re.compile(r"^\s*(?:изучи\s+глубоко|learn\s+deeply)[:\s]+(.+)$", re.I)),
    ("learn",           re.compile(r"^\s*(?:изучи|learn\s+about)[:\s]+(.+)$", re.I)),
    ("quick_note",      re.compile(r"^\s*(?:краткая\s+заметка|quick\s+note)[:\s]+(.+)$", re.I)),
    ("start_project",   re.compile(r"^\s*(?:начать\s+проект|start\s+project)[:\s]+(.+)$", re.I)),
    ("end_project",     re.compile(r"^\s*(?:завершить\s+проект|end\s+project)\s*$", re.I)),
    ("project_ctx",     re.compile(r"^\s*(?:контекст\s+проекта|project\s+context)[:\s]+(.+)$", re.I)),
    ("decision",        re.compile(r"^\s*(?:решили|we\s+decided)\s+(.+?)\s+(?:потому\s+что|because)\s+(.+)$", re.I)),
    ("issue",           re.compile(r"^\s*(?:проблема|issue)[:\s]+(.+?)\s*(?:→|->|\s+fix[:\s])\s*(.+)$", re.I)),
    ("verify_last",     re.compile(r"^\s*(?:проверь(?:\s+ответ)?|verify(?:\s+your\s+answer)?)\s*$", re.I)),
    ("confidence",      re.compile(r"^\s*(?:насколько\s+ты\s+уверен\??|how\s+confident\s+are\s+you\??)\s*$", re.I)),

    # fine-tune
    ("ft_switch",       re.compile(r"^\s*finetune\s+switch\s+(\S+)\s*$", re.I)),
    ("ft_rollback",     re.compile(r"^\s*finetune\s+rollback\s*$", re.I)),
    ("ft_compare",      re.compile(r"^\s*finetune\s+compare\s*$", re.I)),
    ("ft_review",       re.compile(r"^\s*finetune\s+review\s*$", re.I)),
    ("ft_start",        re.compile(r"^\s*finetune\s+start\s*$", re.I)),
    ("ft_status",       re.compile(r"^\s*finetune\s+status\s*$", re.I)),
    ("ft_export_cloud", re.compile(r"^\s*finetune\s+export-cloud\s*$", re.I)),
    ("ft_import_gguf",  re.compile(r"^\s*finetune\s+import-gguf\s+(\S+)(?:\s+--tag\s+(\S+))?\s*$", re.I)),
    ("ft_export",       re.compile(r"^\s*(?:export\s+finetune(?:\s+data)?|finetune\s+export)\s*$", re.I)),
    ("ft_versions",     re.compile(r"^\s*(?:model\s+versions|finetune\s+versions)\s*$", re.I)),
    ("ft_learn_this",   re.compile(r"^\s*(?:learn\s+this\s+to\s+model|\u0437\u0430\u043f\u043e\u043c\u043d\u0438\s+\u0432\s+\u043c\u043e\u0434\u0435\u043b\u044c)\s*$", re.I)),
    ("show_mode",       re.compile(r"^\s*(?:mode|\u0440\u0435\u0436\u0438\u043c)\s*$", re.I)),
    # correction: "неправильно, правильно: <текст>" или "correction: <текст>"
    ("ft_correct",      re.compile(r"^\s*(?:\u043d\u0435\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u043e[,\s]+\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u043e|correction|wrong[,\s]+correct)[:\s]+(.+)$", re.I | re.S)),

    ("status",          re.compile(r"^\s*(?:статус|status)\s*$", re.I)),
    ("gaps",            re.compile(r"^\s*(?:пробелы|gaps)\s*$", re.I)),
    ("whoami",          re.compile(r"^\s*(?:whoami|кто\s*я|что\s*я\s*умею|capabilities)\s*$", re.I)),
    ("graph",           re.compile(r"^\s*(?:graph|граф)(?:\s+(reindex))?\s*$", re.I)),
    ("help",            re.compile(r"^\s*(?:help|помощь|\?)\s*$", re.I)),
]


def parse(text: str) -> ParsedCommand:
    t = text.strip()
    for kind, pat in PATTERNS:
        m = pat.match(t)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                return ParsedCommand(kind=kind, arg=groups[0].strip(), arg2=groups[1].strip())
            if len(groups) == 1:
                return ParsedCommand(kind=kind, arg=groups[0].strip())
            return ParsedCommand(kind=kind)
    return ParsedCommand(kind="none")
