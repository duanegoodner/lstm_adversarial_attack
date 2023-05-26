import subprocess

import simple_access_manager as sam
import xml_data_container as xdc
from project_config import JBDockerProjectConfig, DotEnvFinder
from pathlib import Path


class DeploymentRiskFactorsBuilder:
    def __init__(
            self,
            project_config: JBDockerProjectConfig,
            # xml: xdc.XMLDataContainer
    ):
        self._project_config = project_config
        self._xml = xdc.XMLDataContainer.from_file(project_config.deployment_xml_path)

    # @classmethod
    # def from_file(cls, deployment_xml_path: str | Path):
    #     return cls(xml=xdc.XMLDataContainer.from_file(deployment_xml_path))

    _exactly_one_element = [
        "component",
        "serverData",
        "paths",
        "serverdata",
        "mapping",
    ]

    @property
    def _required_settings(self): return [
        xdc.ElementSettings(
            tag="component",
            key_val_pairs={
                "autoUpload": None,
                "remoteFilesAllowedToDisappearOnAutoupload": "false",
            },
        ),
        xdc.ElementSettings(
            tag="paths",
            key_val_pairs={
                "name": (
                    "jetbrains@localhost:"
                    f"{self._project_config.env.get('CONTAINER_SSH_PORT')} key"
                )
            },
        ),
        xdc.ElementSettings(
            tag="mapping",
            key_val_pairs={
                "deploy": self._project_config.env.get("CONTAINER_PROJECT_DIR"),
                "local": "$PROJECT_DIR$",
            },
        ),
        xdc.ElementSettings(
            tag="excludedPath",
            key_val_pairs={
                "local": "true",
                "path": "$PROJECT_DIR$"
            },
        ),
    ]

    _prohibited_settings = [
        xdc.ElementSettings(
            tag="component", key_val_pairs={"autoUpload": "Always"}
        ),
        xdc.ElementSettings(
            tag="option", key_val_pairs={"name": "myAutoUpload"}
        )
        # Option element has a value attribute, but have not seen it take
        # on anything other than "ALWAYS" when name="myAutoUpload", so just
        # prohibit any option element with name="myAutoUpload"
    ]

    @property
    def _xml_requirements(self):
        exactly_one = [
            xdc.ExactlyOne(ref=item, xml=self._xml)
            for item in self._exactly_one_element
        ]

        required = [
            xdc.RequiredSetting(ref=item, xml=self._xml)
            for item in self._required_settings
        ]

        prohibited = [
            xdc.ProhibitedSetting(ref=item, xml=self._xml)
            for item in self._prohibited_settings
        ]

        return exactly_one + required + prohibited

    def build(self):
        return [
            item.to_config_risk_factor() for item in self._xml_requirements
        ]


class JetBrainsAccessController(sam.AccessController):
    def __init__(self, username: str, work_group: str):
        self._username = username
        self._work_group = work_group

    def grant_access(self):
        cmd = ["sudo", "usermod", "-a", "-G", self._work_group, self._username]
        subprocess.run(cmd)

    def revoke_access(self):
        cmd = [
            "sudo",
            "gpasswd",
            "-d",
            self._username,
            self._work_group,
        ]
        subprocess.run(cmd)


class JetbrainsAccessManagerBuilder:
    def __init__(
        self,
        project_config: JBDockerProjectConfig,
        # username: str = PROJECT_CONFIG.env.get("JETBRAINS_USER"),
        # root_access_paths: list[Path] = PROJECT_CONFIG.work_paths,
        # deployment_xml_path: str = PROJECT_CONFIG.deployment_xml_path,
    ):
        self._project_config = project_config
        self._deployment_xml_path = project_config.deployment_xml_path
        self._username = project_config.env.get("JETBRAINS_USER")
        self._root_access_paths = project_config.work_paths

    @classmethod
    def from_project_root(cls, project_root: Path):
        dot_env_path = DotEnvFinder(project_root=project_root).get_dot_env()
        project_config = JBDockerProjectConfig(
            dot_env_path=dot_env_path)
        return cls(project_config=project_config)

    @property
    def risk_factors(self):
        return DeploymentRiskFactorsBuilder(
            project_config=self._project_config
            # xml=xdc.XMLDataContainer.from_file(
            #     file_path=self._deployment_xml_path
            # )
        ).build()

    def build(self):
        return sam.SimpleAccessManager(
            username=self._username,
            root_access_paths=self._root_access_paths,
            condition_checker=sam.ConfigConditionsChecker(self.risk_factors),
            access_controller=JetBrainsAccessController(
                username=self._username,
                work_group=self._project_config.env.get("WORK_GROUP"),
            ),
        )


