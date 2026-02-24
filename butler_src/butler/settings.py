import os

from dynaconf import Dynaconf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_SETTINGS_PATH = os.path.join(CURRENT_DIR, "config.toml")
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "nlp_butler")
USER_CONFIG_FILE = os.path.join(USER_CONFIG_DIR, "config.toml")

settings = Dynaconf(
    envvar_prefix="NLP_BUTLER",
    settings_files=[PACKAGE_SETTINGS_PATH, USER_CONFIG_FILE, "config.local.toml"],
    load_dotenv=True,
    merge_enabled=True,
    root_path=CURRENT_DIR,
)
