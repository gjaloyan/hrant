from backend.commands import parse


def test_remember():
    c = parse("запомни навсегда: пользователь — инженер КИПиА")
    assert c.kind == "remember"
    assert "КИПиА" in c.arg


def test_forget():
    c = parse("забудь про старый проект")
    assert c.kind == "forget"


def test_learn_quick():
    c = parse("изучи: MAX485")
    assert c.kind == "learn"
    assert c.arg == "MAX485"


def test_learn_deep():
    c = parse("изучи глубоко: Modbus RTU")
    assert c.kind == "learn_deep"
    assert "Modbus" in c.arg


def test_show():
    c = parse("что ты знаешь о RS-485?")
    assert c.kind == "show"
    assert c.arg == "RS-485"


def test_list():
    assert parse("что ты знаешь?").kind == "list"


def test_start_project():
    c = parse("начать проект: котельная №3")
    assert c.kind == "start_project"
    assert "котельная" in c.arg


def test_decision():
    c = parse("решили взять MAX485 потому что дешёвый")
    assert c.kind == "decision"
    assert c.arg == "взять MAX485"
    assert c.arg2 == "дешёвый"


def test_issue():
    c = parse("проблема: дребезг → поставить конденсатор 100нФ")
    assert c.kind == "issue"


def test_none():
    assert parse("какой ток через резистор?").kind == "none"
