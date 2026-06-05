import os
from pathlib import Path

from dynaconf import Dynaconf  # type: ignore

# settings = Dynaconf(
#     envvar_prefix="DYNACONF",
#     settings_files=["settings.toml", "dataset.toml", ".secrets.toml"],
# )

# # `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# # `settings_files` = Load these files in the order.

base_dir = os.path.dirname(os.path.abspath(__file__))

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        os.path.join(base_dir, "user.toml"),
        os.path.join(base_dir, "settings.toml"),
        os.path.join(base_dir, "dataset.toml"),
        os.path.join(base_dir, "models.toml"),
        os.path.join(base_dir, ".secrets.toml"),
    ],
)

for disaster in ["fire", "flood", "landslide"]:
    settings[disaster].results_path = str(
        Path(settings.default.user_path)
        / settings.default.checkpoint_path
        / settings[disaster].results_path
    )
    settings[disaster].data_path = str(
        Path(settings.default.user_path) / settings.default.data_path
    )
    settings[disaster].checkpoint_path = str(
        Path(settings.default.user_path) / settings.default.checkpoint_path
    )


for disaster in ["flood"]:
    settings[disaster].dataloader.train_test_split_file = str(
        Path(settings[disaster].data_path) / settings[disaster].dataloader.train_test_split_file
    )
    settings[disaster].dataloader.windows_file = str(
        Path(settings[disaster].data_path) / settings[disaster].dataloader.windows_file
    )
