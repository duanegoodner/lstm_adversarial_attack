import pandas as pd
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


class PreprocessModule(ABC):
    def __init__(
        self,
        settings: dataclass,
        incoming_resource_refs: dataclass,
        exported_resources: dict[str, pr.ExportedPreprocessResource] = None,
    ):
        self.settings = settings
        self.incoming_resource_refs = incoming_resource_refs
        self._resource_exporter = rio.ResourceExporter()
        self.resource_importer = rio.ResourceImporter()
        if exported_resources is None:
            exported_resources = {}
        self.exported_resources = exported_resources

    def __call__(
        self, *args, **kwargs
    ) -> dict[str, pr.ExportedPreprocessResource]:
        self.process()
        return self.exported_resources

    def import_csv(self, path) -> pd.DataFrame:
        return self.resource_importer.import_csv(path=path)

    def import_pickle_to_df(self, path: Path) -> pd.DataFrame:
        return self.resource_importer.import_pickle_to_df(path=path)

    def import_pickle_to_object(self, path: Path) -> object:
        return self.resource_importer.import_pickle_to_object(path=path)

    def import_pickle_to_list(self, path: Path) -> list:
        return self.resource_importer.import_pickle_to_list(path=path)

    def export_resource(self, key: str, resource: object, path: Path):
        assert key not in self.exported_resources
        assert path not in [
            item.path for item in self.exported_resources.values()
        ]
        self._resource_exporter.export(resource=resource, path=path)
        exported_resource = pr.ExportedPreprocessResource(
            path=path, data_type=type(resource).__name__
        )
        self.exported_resources[key] = exported_resource

    @abstractmethod
    def process(self):
        pass
