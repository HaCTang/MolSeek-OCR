import multiprocessing
from typing import Any

import numpy as np
import rdkit
from SmilesPE.pretokenizer import atomwise_tokenizer
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(
    smiles: Any,
    ignore_chiral: bool = False,
    ignore_cistrans: bool = False,
    replace_rgroup: bool = True,
) -> tuple[str, bool]:
    if type(smiles) is not str or smiles == "":
        return "", False
    if ignore_cistrans:
        smiles = smiles.replace("/", "").replace("\\", "")
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == "[" and token[-1] == "]":
                symbol = token[1:-1]
                if symbol[0] == "R" and symbol[1:].isdigit():
                    tokens[j] = f"[{symbol[1:]}*]"
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = "*"
        smiles = "".join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except Exception:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def convert_smiles_to_canonsmiles(
    smiles_list: list[str],
    ignore_chiral: bool = False,
    ignore_cistrans: bool = False,
    replace_rgroup: bool = True,
    num_workers: int = 16,
) -> tuple[list[str], float]:
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(
            canonicalize_smiles,
            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
            chunksize=128,
        )
    canon_smiles, success = zip(*results)
    return list(canon_smiles), float(np.mean(success))


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        return float(DataStructs.FingerprintSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def compute_tanimoto_similarities(gold_smiles: list[str], pred_smiles: list[str], num_workers: int = 16) -> list[float]:
    with multiprocessing.Pool(num_workers) as pool:
        similarities = pool.starmap(
            tanimoto_similarity,
            [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)],
        )
    return similarities


class SmilesEvaluator:
    def __init__(self, gold_smiles: list[str], num_workers: int = 16, tanimoto: bool = False):
        self.gold_smiles = gold_smiles
        self.num_workers = num_workers
        self.tanimoto = tanimoto
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(
            gold_smiles, ignore_cistrans=True, num_workers=num_workers
        )
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(
            gold_smiles, ignore_chiral=True, ignore_cistrans=True, num_workers=num_workers
        )
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)

    @staticmethod
    def _replace_empty(smiles_list: list[str]) -> list[str]:
        return [
            smiles if smiles is not None and isinstance(smiles, str) and smiles != "" else "<empty>"
            for smiles in smiles_list
        ]

    def evaluate(self, pred_smiles: list[str]) -> dict[str, float]:
        results: dict[str, float] = {}
        if self.tanimoto:
            results["tanimoto"] = float(
                np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles, self.num_workers))
            )
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(
            pred_smiles, ignore_cistrans=True, num_workers=self.num_workers
        )
        results["canon_smiles"] = float(
            np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        )
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(
            pred_smiles, ignore_chiral=True, ignore_cistrans=True, num_workers=self.num_workers
        )
        results["graph"] = float(np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral)))
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if "@" in g])
        results["chiral"] = float(np.mean(chiral[:, 0] == chiral[:, 1])) if len(chiral) > 0 else -1.0
        results["accuracy"] = results["canon_smiles"]
        return results


def evaluate_smiles_predictions(
    gold_smiles: list[str],
    pred_smiles: list[str],
    num_workers: int = 16,
    tanimoto: bool = False,
) -> dict[str, float]:
    evaluator = SmilesEvaluator(gold_smiles, num_workers=num_workers, tanimoto=tanimoto)
    return evaluator.evaluate(pred_smiles)
