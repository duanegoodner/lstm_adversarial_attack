import pathlib
from abc import ABC, abstractmethod
from pathlib import Path

from dataclasses import dataclass

import lstm_adversarial_attack.resource_io as rio
from typing import Any


@dataclass
class ProvenanceInfo:
    category_name: str
    output_dir: Path
    new_items: dict[str, Any]
    previous_info: dict[str, Any] | Path


class DataProvenanceBuilder:
    def __init__(
        self,
        pipeline_component: object,
        category_name: str,
        output_dir: Path,
        new_items: dict[str, Any] = None,
        previous_info: dict[str, Any] | Path = None,
    ):
        self.pipeline_component = pipeline_component
        self.category_name = category_name
        if new_items is None:
            new_items = {}
        self.new_items = new_items
        if previous_info is None:
            previous_info = {}
        if type(previous_info) == pathlib.PosixPath:
            previous_info = rio.ResourceImporter().import_pickle_to_object(
                path=previous_info
            )
        self.data = previous_info
        self.data[self.category_name] = {}
        self.output_dir = output_dir

    def add_item(self, key: str, value: Any):
        self.data[self.category_name][key] = value

    def build(self) -> dict[str, Any]:
        for key, value in self.new_items.items():
            self.add_item(key=key, value=value)
        if self.output_dir:
            rio.ResourceExporter().export(
                resource=self.data, path=self.output_dir / "provenance.pickle"
            )
        return self.data


class HasDataProvenance(ABC):
    @property
    @abstractmethod
    def provenance_info(self) -> ProvenanceInfo:
        pass

    @property
    def data_provenance(self) -> dict[str, Any]:
        builder = DataProvenanceBuilder(
            pipeline_component=self,
            category_name=self.provenance_info.category_name,
            output_dir=self.provenance_info.output_dir,
            new_items=self.provenance_info.new_items,
            previous_info=self.provenance_info.previous_info,
        )
        return builder.build()

    def write_provenance(self) -> Path:
        output_path = self.provenance_info.output_dir / "provenance.pickle"
        rio.ResourceExporter().export(
            resource=self.data_provenance, path=output_path
        )
        return output_path
