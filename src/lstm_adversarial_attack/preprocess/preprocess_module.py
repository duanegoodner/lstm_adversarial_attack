import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, Callable

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


@dataclass
class PreprocessModuleSettings(ABC):
    """
    Ensure all module's settings classes have output_dir attribute
    """

    output_dir: Path

    def to_json(self, path: Path, **kwargs):
        output_dict = rio.convert_posix_paths_to_strings(data=self.__dict__)
        with path.open(mode="w") as out_file:
            json.dump(obj=output_dict, fp=out_file)


@dataclass
class PreprocessResourceRefs(ABC):
    def __post_init__(self):
        assert all(
            [
                (type(value) == PosixPath)
                for key, value in self.__dict__.items()
            ]
        )

    # TODO consider an ABC that provides this method or just a standalone funct
    def to_json(self, path: Path, **kwargs):
        output_dict = rio.convert_posix_paths_to_strings(data=self.__dict__)
        with path.open(mode="w") as out_file:
            json.dump(obj=output_dict, fp=out_file)


class PreprocessModule(ABC):
    """
    Base class for all modules used by Preprocessor.
    """

    def __init__(
        self,
        name: str,
        settings: PreprocessModuleSettings,
        incoming_resource_refs: dataclass,
    ):
        """
        :param name: identifying name for the module
        :param settings: contains settings used by implemented module
        :param incoming_resource_refs: contains paths of files to be imported
        """
        self.name = name
        self.settings = settings
        self.incoming_resource_refs = incoming_resource_refs
        self._resource_exporter = rio.ResourceExporter()
        self.resource_importer = rio.ResourceImporter()
        self.exported_resources = {}

    def __call__(
        self, *args, **kwargs
    ) -> dict[str, pr.ExportedPreprocessResource]:
        """
        Runs concrete class' .process() method
        :return: dictionary with references to all files exported by module
        """
        self.process()
        self.export_resource_new(
            key="resource_refs",
            resource=self.incoming_resource_refs,
            path=self.settings.output_dir / "resource_refs.json",
            exporter=self.incoming_resource_refs.to_json
        )
        self.export_resource_new(
            key="settings",
            resource=self.settings,
            path=self.settings.output_dir / "settings.json",
            exporter=self.settings.to_json,
        )
        return self.exported_resources

    def import_pickle_to_df(self, path: Path) -> pd.DataFrame:
        """
        Convenience funct for type-hinting when import pickle to df
        """
        return self.resource_importer.import_pickle_to_df(path=path)

    def import_pickle_to_object(self, path: Path) -> object:
        """
        Imports pickle with output type-hinting just a generic object.
        """
        return self.resource_importer.import_pickle_to_object(path=path)

    def import_pickle_to_list(self, path: Path) -> list:
        """
        Type-hinting convenience funct for importing pickle to list.
        """
        return self.resource_importer.import_pickle_to_list(path=path)

    def export_resource(self, key: str, resource: object, path: Path):
        """
        Exports python object to pickle file & updates self.exported_resources
        :param key: identifier for entry in self.exported_resources
        :param resource: python object to be saved as pickle
        :param path: file path where pickle will be saved.
        """
        assert key not in self.exported_resources
        assert path not in [
            item.path for item in self.exported_resources.values()
        ]
        self._resource_exporter.export(resource=resource, path=path)
        exported_resource = pr.ExportedPreprocessResource(
            path=path, data_type=type(resource).__name__
        )
        self.exported_resources[key] = exported_resource

    def export_resource_new(
        self,
        key: str,
        resource: object,
        path: Path,
        exporter: Callable,
        exporter_kwargs: dict[str, Any] = None,
    ):
        if exporter_kwargs is None:
            exporter_kwargs = {}
        assert key not in self.exported_resources
        assert path not in [
            item.path for item in self.exported_resources.values()
        ]
        exporter(resource=resource, path=path, **exporter_kwargs)
        self.exported_resources[key] = pr.ExportedPreprocessResource(
            path=path, data_type=type(resource).__name__
        )

    @abstractmethod
    def process(self):
        pass
