import os

from ragpick.envfile import load_dotenv


def test_env_found_in_parent_folder_and_real_env_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("RAGPICK_TEST_A=from_file\nRAGPICK_TEST_B='quoted'\n", encoding="utf-8")
    child = tmp_path / "project" / "sub"
    child.mkdir(parents=True)
    monkeypatch.delenv("RAGPICK_TEST_B", raising=False)
    monkeypatch.setenv("RAGPICK_TEST_A", "from_env")
    load_dotenv(child)
    assert os.environ["RAGPICK_TEST_A"] == "from_env"
    assert os.environ["RAGPICK_TEST_B"] == "quoted"
    monkeypatch.delenv("RAGPICK_TEST_B")
