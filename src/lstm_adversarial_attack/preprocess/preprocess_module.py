import pandas as pd
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


class PreprocessModule(ABC):
    """
    Base class for all modules used by Preprocessor.
    """
    def __init__(
        self,
        name: str,
        settings: dataclass,
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
        return self.exported_resources

    def import_csv(self, path: Path) -> pd.DataFrame:
        """
        Convenience function to aid IDE type-hinting of csv imported to df
        """
        # TODO Can we remove this method and just use pd.read_csv?
        return self.resource_importer.import_csv(path=path)

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

    @abstractmethod
    def process(self):
        pass
