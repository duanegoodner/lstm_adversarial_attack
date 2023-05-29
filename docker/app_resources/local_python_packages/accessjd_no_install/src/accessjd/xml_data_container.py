from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import simple_access_manager as sam
from typing import Callable
import defusedxml.ElementTree as det   # defusedxml for safe parsing
import xml.etree.ElementTree as et     # xml for typehints & post-parse ops


@dataclass
class ElementSettings:
    tag: str
    key_val_pairs: dict[str, Union[str, None]]


class XMLDataContainer:
    def __init__(self, element_tree: et.ElementTree):
        self._element_tree = element_tree

    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        return cls(det.parse(file_path))

    @property
    def root(self) -> et.Element:
        return self._element_tree.getroot()

    def num_occurrences_of_tag(self, tag: str) -> int:
        num_occurrences = 0
        for _ in list(self.root.iter(tag)):
            num_occurrences += 1
        return num_occurrences

    @staticmethod
    def has_settings(element: et.Element, key_val_pairs: dict):
        return all(
            [
                element.attrib.get(key) == val
                for key, val in key_val_pairs.items()
            ]
        )

    def has_element_with_settings(self, target: ElementSettings) -> bool:
        elements_with_tag = list(self.root.iter(target.tag))
        return any(
            [
                self.has_settings(element, target.key_val_pairs)
                for element in elements_with_tag
            ]
        )


@dataclass
class XMLRequirement(ABC):
    ref: str | ElementSettings
    xml: XMLDataContainer

    @property
    @abstractmethod
    def safety_check(self) -> Callable[..., bool]:
        ...

    @property
    @abstractmethod
    def error_msg(self) -> str:
        ...

    def to_config_risk_factor(self) -> sam.ConfigRiskFactor:
        return sam.ConfigRiskFactor(
            safety_check=self.safety_check,
            safety_check_kwargs={},
            risk_message=self.error_msg,
        )


class ExactlyOne(XMLRequirement):
    def __init__(self, ref: str, xml: XMLDataContainer):
        super().__init__(ref=ref, xml=xml)

    @property
    def safety_check(self) -> Callable[..., bool]:
        num_occurrences = self.xml.num_occurrences_of_tag(self.ref)

        def safety_check_funct():
            return num_occurrences == 1

        return safety_check_funct

    @property
    def error_msg(self) -> str:
        return (
            f"XML container must have exactly one element of type {self.ref}"
        )


class RequiredSetting(XMLRequirement):
    def __init__(self, ref: ElementSettings, xml: XMLDataContainer):
        super().__init__(ref=ref, xml=xml)

    @property
    def safety_check(self) -> Callable[..., bool]:
        def safety_check_funct() -> bool:
            return self.xml.has_element_with_settings(target=self.ref)

        return safety_check_funct

    @property
    def error_msg(self) -> str:
        return (
            f"XML container must have {self.ref.tag} element with"
            f" {self.ref.key_val_pairs}"
        )


class ProhibitedSetting(XMLRequirement):
    def __init__(self, ref: ElementSettings, xml: XMLDataContainer):
        super().__init__(ref=ref, xml=xml)

    @property
    def safety_check(self) -> Callable[..., bool]:
        def safety_check_funct() -> bool:
            return not self.xml.has_element_with_settings(target=self.ref)
        return safety_check_funct

    @property
    def error_msg(self):
        return f"{self.ref.tag} with {self.ref.key_val_pairs} is prohibited"

