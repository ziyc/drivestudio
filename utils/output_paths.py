import os
import re
import time
from typing import Iterable


def _slugify(value) -> str:
    text = str(value)
    text = text.replace(os.sep, "-")
    text = text.replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._@-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or "na"


def timestamp_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def make_tag(key: str, value) -> str:
    return f"{_slugify(key)}-{_slugify(value)}"


def join_tags(tags: Iterable[str]) -> str:
    return "_".join([tag for tag in tags if tag])


def build_auto_output_dir(output_root: str, command: str, *tags: str) -> str:
    dirname = join_tags([timestamp_now(), *[_slugify(tag) for tag in tags]])
    return os.path.join(output_root, command, dirname)


def count_cameras(camera_list) -> int:
    try:
        return len(camera_list)
    except TypeError:
        return 0
