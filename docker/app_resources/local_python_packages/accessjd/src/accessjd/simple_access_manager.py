import grp
import pygetfacl
import pwd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any


@dataclass
class ConfigRiskFactor:
    safety_check: Callable[..., bool]
    safety_check_kwargs: dict[str, Any]
    risk_message: str

    def __call__(self):
        return self.safety_check(**self.safety_check_kwargs)


@dataclass
class ConditionsCheckResult:
    all_safe: bool
    bad_condition_messages: list[str]


class PathAccessChecker:
    def __init__(self, path: str | Path, username: str):
        self._path = Path(path)
        self._username = username
        self._uid = pwd.getpwnam(username).pw_uid

    @property
    def _acl_info(self) -> pygetfacl.ACLData:
        return pygetfacl.getfacl(self._path)

    @property
    # https://stackoverflow.com/a/9324811
    def users_groups(self) -> list[str]:
        groups = [
            g.gr_name for g in grp.getgrall() if self._username in g.gr_mem
        ]
        gid = pwd.getpwnam(self._username).pw_gid
        groups.append(grp.getgrgid(gid).gr_name)
        return groups

    @property
    def effective_permissions(self) -> pygetfacl.EffectivePermissions:
        return self._acl_info.effective_permissions

    @property
    def path_special_groups(self) -> list[str]:
        return [
            key
            for key, val in self.effective_permissions.special_groups.items()
        ]

    def user_is_owning_user(self):
        return self._acl_info.owning_user == self._username

    def user_is_in_owning_group(self):
        return self._acl_info.owning_group in self.users_groups

    def user_is_special_user(self):
        if self._acl_info.special_users and (
            self._username in self._acl_info.special_users
        ):
            return True
        return False

    def user_is_in_special_group(self):
        return any(
            [
                users_group in self.path_special_groups
                for users_group in self.users_groups
            ]
        )

    def user_is_other(self):
        return not (
            self.user_is_owning_user()
            or self.user_is_in_owning_group()
            or self.user_is_special_user()
            or self.user_is_in_special_group()
        )

    def user_has_access_as_owner(self, access_type: str) -> bool:
        return self.user_is_owning_user() and getattr(
            self.effective_permissions.user, access_type
        )

    def user_has_access_as_member_of_owning_group(
        self, access_type: str
    ) -> bool:
        return self.user_is_in_owning_group() and getattr(
            self.effective_permissions.group, access_type
        )

    def user_has_access_as_special_user(self, access_type: str) -> bool:
        return self.user_is_special_user() and getattr(
            self.effective_permissions.special_users[self._username],
            access_type,
        )

    def special_groups_with_access(self, access_type: str) -> list[str]:
        return [
            group
            for group in self.path_special_groups
            if getattr(
                self.effective_permissions.special_groups[group], access_type
            )
        ]

    def user_has_access_as_member_of_special_group(
        self, access_type: str
    ) -> bool:
        return any(
            [
                group
                in self.special_groups_with_access(access_type=access_type)
                for group in self.users_groups
            ]
        )

    def user_has_access_as_other(self, access_type: str):
        return self.user_is_other() and getattr(
            self.effective_permissions.other, access_type
        )

    def user_has_access(self, access_type: str):
        assert len(access_type) == 1 and access_type in "rwx"
        return (
            self.user_has_access_as_owner(access_type=access_type)
            or self.user_has_access_as_member_of_owning_group(
                access_type=access_type
            )
            or self.user_has_access_as_special_user(access_type=access_type)
            or self.user_has_access_as_member_of_special_group(
                access_type=access_type
            )
            or self.user_has_access_as_other(access_type=access_type)
        )


class ConfigConditionsChecker:
    def __init__(self, risk_factors: list[ConfigRiskFactor]):
        self._risk_factors = risk_factors

    @property
    def risk_factors(self) -> list[ConfigRiskFactor]:
        return self._risk_factors

    def check_conditions(self) -> ConditionsCheckResult:
        safety_check_results = []
        bad_condition_messages = []
        for risk_factor in self.risk_factors:
            condition_result = risk_factor()
            safety_check_results.append(condition_result)
            if not condition_result:
                bad_condition_messages.append(risk_factor.risk_message)

        return ConditionsCheckResult(
            all_safe=all(safety_check_results),
            bad_condition_messages=bad_condition_messages,
        )


class AccessController(ABC):
    @abstractmethod
    def grant_access(self):
        ...

    @abstractmethod
    def revoke_access(self):
        ...


class SafeStateSetter(ABC):
    @abstractmethod
    def set_to_safe_state(self):
        ...


class SimpleAccessManager:
    def __init__(
        self,
        username: str,
        root_access_paths: list[Path],
        condition_checker: ConfigConditionsChecker,
        access_controller: AccessController,
        safe_state_setter: SafeStateSetter
    ):
        self._user = username
        self._root_access_paths = root_access_paths
        self._condition_checker = condition_checker
        self._access_controller = access_controller
        self._safe_state_setter = safe_state_setter

    @property
    def user(self) -> str:
        return self._user

    @property
    def root_access_paths(self) -> list[Path]:
        return self._root_access_paths

    @property
    # each key is a root accessjd path, vals are descendants of that path
    def access_paths(self) -> dict[Path, list[Path]]:
        return {
            path: sorted(path.glob("**/*")) for path in self._root_access_paths
        }

    def _user_has_access(self, path: Path, access_type: str):
        return PathAccessChecker(
            path=path, username=self._user
        ).user_has_access(access_type=access_type)

    def _has_any_access_to_tree(self, root_path: Path, access_type: str):
        return self._user_has_access(
            path=root_path, access_type=access_type
        ) or any(
            [
                self._user_has_access(path=sub_path, access_type=access_type)
                for sub_path in root_path.glob("*/**")
            ]
        )

    def check_conditions(self) -> ConditionsCheckResult:
        return self._condition_checker.check_conditions()

    def is_safe_to_grant_access(self) -> bool:
        return self.check_conditions().all_safe

    def _has_full_access_to_tree(self, root_path: Path, access_type: str):
        return self._user_has_access(
            path=root_path, access_type=access_type
        ) and all(
            [
                self._user_has_access(path=sub_path, access_type=access_type)
                for sub_path in root_path.glob("*/**")
            ]
        )

    def any_root_access_dict(self, access_type: str) -> dict[Path, bool]:
        return {
            path: self._has_any_access_to_tree(
                root_path=path, access_type=access_type
            )
            for path in self._root_access_paths
        }

    def full_root_access_dict(self, access_type: str) -> dict[Path, bool]:
        return {
            path: self._has_full_access_to_tree(
                root_path=path, access_type=access_type
            )
            for path in self._root_access_paths
        }

    def has_any_access(self, access_type: str) -> bool:
        return any(
            [
                val
                for key, val in self.any_root_access_dict(
                    access_type=access_type
                ).items()
            ]
        )

    def has_full_access(self, access_type: str) -> bool:
        return all(
            [
                val
                for key, val in self.any_root_access_dict(
                    access_type=access_type
                ).items()
            ]
        )

    def grant_access(self):
        self._access_controller.grant_access()

    @abstractmethod
    def revoke_access(self):
        self._access_controller.revoke_access()
        self._safe_state_setter.set_to_safe_state()
        # if self.has_any_access("w"):
        #     trees_with_write_access = [
        #         root_path
        #         for root_path, access_status in self.any_root_access_dict(
        #             access_type="w"
        #         ).items()
        #         if access_status
        #     ]
        #     print(
        #         f"Warning: {self._user} still has write access to some items"
        #         "under the following root paths:\n"
        #         f"{trees_with_write_access}"
        #     )

    # def set_safe_access_state(self):
    #     self._safe_state_setter.set_to_safe_state()
