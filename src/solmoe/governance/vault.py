from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict


def _sha256_of_path(path: str) -> str:
    h = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    h.update(fh.read())
    else:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


@dataclass
class RegistryRecord:
    dataset_hash: str
    reward_spec_hash: str
    simulator_hash: str
    meta: Dict[str, str]


class PolicyVault:
    """
    Policy/model registry with hashes of dataset snapshot, reward spec, simulator version;
    reliability and drift reports can be attached.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def register(self, name: str, dataset_path: str, reward_spec_path: str, simulator_path: str, meta: Dict[str, str]) -> str:
        rec = RegistryRecord(
            dataset_hash=_sha256_of_path(dataset_path),
            reward_spec_hash=_sha256_of_path(reward_spec_path),
            simulator_hash=_sha256_of_path(simulator_path),
            meta=meta,
        )
        outp = os.path.join(self.root, f"{name}.json")
        with open(outp, "w") as f:
            json.dump(rec.__dict__, f, indent=2)
        return outp

    def load(self, name: str) -> Dict:
        p = os.path.join(self.root, f"{name}.json")
        with open(p, "r") as f:
            return json.load(f)

