import os
from dotenv import load_dotenv
from pathlib import Path


class DotEnvNotFound(Exception):
    def __str__(self):
        return ".env file not found"


class DotEnvFinder:
    _default_env_dirs = ["docker/app_dev"]

    def __init__(
        self,
        # project_root: Path = Path("/home/devspace/project"),
        project_root: Path,
        expected_dotenv_paths: list[Path] = None,
    ):
        self._project_root = project_root
        if expected_dotenv_paths is None:
            self._expected_dot_env_paths = [
                self._project_root / Path(env_dir) / ".env"
                for env_dir in self._default_env_dirs
            ]

    @property
    def expected_dot_env_paths(self) -> list[Path]:
        return self._expected_dot_env_paths

    @property
    def available_dot_env_files(self) -> list[Path]:
        return [path for path in self.expected_dot_env_paths if path.exists()]

    def get_dot_env(self) -> Path:
        if len(self.available_dot_env_files) == 0:
            raise DotEnvNotFound
        if len(self.available_dot_env_files) > 1:
            print(
                "Warning: Multiple .env files found:\n"
                f"{self.available_dot_env_files}\n"
                f"Using .env file at {self.available_dot_env_files[0]}"
            )
        return self.available_dot_env_files[0]


class EnvValueRetriever:
    def __init__(self, dot_env_path: Path):
        self._dot_env_path = dot_env_path

    def get(self, env_var: str) -> str:
        load_dotenv(self._dot_env_path)
        return os.getenv(env_var)

    def all(self) -> dict[str, str]:
        load_dotenv(self._dot_env_path)
        return dict(os.environ.items())


class JBDockerProjectConfig:
    def __init__(
        self, dot_env_path: str | Path
    ):
        self._env = EnvValueRetriever(dot_env_path=dot_env_path)
        # self._work_path_env_vars = work_path_env_vars

    @property
    def env(self):
        return self._env

    @property
    def work_paths(self) -> list[Path]:
        work_path_env_vars = [
            key
            for key, value in self._env.all().items()
            if key.startswith(self._env.get("WORKDIR_VARNAME_PREFIX"))
        ]

        return [
            Path(self._env.get(env_var))
            for env_var in work_path_env_vars
        ]
        # return [
        #     Path(self._env.get("CONTAINER_SRC_DIR")),
        #     Path(self._env.get("CONTAINER_DATA_DIR"))
        # ]

        # below method sometimes worked, but not always:
        # return [
        #     Path(entry) for entry in json.loads(self._env.get("WORK_PATHS"))
        # ]

    @property
    def ssh_interpreter_name(self) -> str:
        return (
            f"{self._env.get('JETBRAINS_USER')}@"
            f"{self._env.get('CONTAINER_HOST')}:"
            f"{self._env.get('CONTAINER_SSH_PORT')} "
            f"{self._env.get('JETBRAINS_SSH_ACCESS_TYPE')}"
        )

        # return self._env.get("JETBRAINS_SSH_INTERPRETER_NAME")
        # return (
        #     f"jetbrains@localhost:${self._env.get('CONTAINER_SSH_PORT')} key"
        # )

    @property
    def deployment_xml_path(self) -> Path:
        return (
            Path(self._env.get("CONTAINER_PROJECT_ROOT"))
            / ".idea"
            / "deployment.xml"
        )


# PROJECT_CONFIG = JBDockerProjectConfig(
#     dot_env_path=DotEnvFinder().get_dot_env()
# )
