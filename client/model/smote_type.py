from enum import Enum


class SmoteType(Enum):
    SMOTE = ["smote", "SMOTE"]
    SMOTENC = ["smotenc", "SMOTENC"]

    @classmethod
    def match_str(cls, to_match: str) -> str | None:
        for sm_type in cls:
            if to_match in sm_type.value:
                return sm_type.value[0]
        return None
