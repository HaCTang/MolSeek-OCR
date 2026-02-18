import os
import random
import io
import string
import sys
import argparse
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset


DEFAULT_INSTRUCTION = "<image>\n Give me the SMILES of the molecule. "
SMILES_CANDIDATE_COLUMNS = ("SMILES", "smiles", "canonical_smiles")
ID_CANDIDATE_COLUMNS = ("Unnamed: 0", "id", "idx", "index", "image_id", "pubchem_cid")

_WORKER_RENDERER = None


def _ensure_list(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _find_smiles_column(df: pd.DataFrame) -> str:
    for column in SMILES_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(
        f"No SMILES column found. Tried columns: {SMILES_CANDIDATE_COLUMNS}, got: {list(df.columns)}"
    )


def _read_and_merge_csvs(
    csv_files: Union[str, Sequence[str]],
    keep_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    files = _ensure_list(csv_files)
    if not files:
        raise ValueError("csv_files is empty.")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        frame = pd.read_csv(file_path)
        frame["source_csv"] = file_path
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    smiles_col = _find_smiles_column(merged)
    merged = merged[merged[smiles_col].notna()].copy()
    merged[smiles_col] = merged[smiles_col].astype(str).str.strip()
    merged = merged[merged[smiles_col] != ""].copy()

    if keep_columns is not None:
        keep = set(keep_columns)
        keep.add(smiles_col)
        keep.add("source_csv")
        existing = [c for c in merged.columns if c in keep]
        merged = merged[existing]

    merged = merged.reset_index(drop=True)
    return merged


def discover_training_csvs(
    training_data_root: str,
    include_keywords: Optional[Sequence[str]] = None,
) -> List[str]:
    if not os.path.isdir(training_data_root):
        raise FileNotFoundError(f"Directory not found: {training_data_root}")

    keywords = [k.lower() for k in include_keywords] if include_keywords else None
    csv_files: List[str] = []

    for root, _, files in os.walk(training_data_root):
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(root, name)
            if keywords:
                low = path.lower()
                if not any(k in low for k in keywords):
                    continue
            csv_files.append(path)

    csv_files.sort()
    return csv_files


@dataclass
class MoleculeStyleConfig:
    mol_augment: bool = True
    default_option: bool = False
    include_condensed: bool = True
    comment_prob: float = 0.25
    color_prob: float = 0.30
    blur_prob: float = 0.20
    noise_prob: float = 0.20
    allow_text_fallback: bool = False
    # Supported styles: "molscribe_default", "chemdraw_like"
    render_style: str = "molscribe_default"

    def __post_init__(self) -> None:
        allowed = {"molscribe_default", "chemdraw_like"}
        if self.render_style not in allowed:
            raise ValueError(
                f"Unsupported render_style={self.render_style}. "
                f"Expected one of {sorted(allowed)}"
            )


class MoleculeStyleRenderer:
    """
    Self-contained molecule renderer with style randomization.
    Backend priority:
    1) RDKit (real molecular rendering)
    2) PIL text fallback (always available)
    """

    def __init__(self, style_config: MoleculeStyleConfig):
        self.style_config = style_config
        self.backend = "none"
        self._rdkit = None
        self._indigo = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore
            from rdkit.Chem.Draw import PrepareAndDrawMolecule  # type: ignore

            self._rdkit = {
                "Chem": Chem,
                "AllChem": AllChem,
                "rdMolDraw2D": rdMolDraw2D,
                "PrepareAndDrawMolecule": PrepareAndDrawMolecule,
            }
            self.backend = "rdkit"
            return
        except Exception:
            self._rdkit = None

        self._init_indigo_backend()
        if self.backend != "indigo" and self.style_config.allow_text_fallback:
            self.backend = "pil_text"

    def _init_indigo_backend(self) -> None:
        indigo_base = os.path.join(os.path.dirname(__file__), "MolScribe", "molscribe")
        if not os.path.isdir(indigo_base):
            return
        if indigo_base not in sys.path:
            sys.path.insert(0, indigo_base)

        try:
            from indigo import Indigo  # type: ignore
            from indigo.renderer import IndigoRenderer  # type: ignore
            from constants import RGROUP_SYMBOLS, SUBSTITUTIONS, COLORS  # type: ignore

            self._indigo = {
                "Indigo": Indigo,
                "IndigoRenderer": IndigoRenderer,
                "RGROUP_SYMBOLS": RGROUP_SYMBOLS,
                "SUBSTITUTIONS": SUBSTITUTIONS,
                "COLORS": COLORS,
            }
            self.backend = "indigo"
        except Exception:
            self._indigo = None

    @staticmethod
    def _line_wrap_smiles(smiles: str, width: int = 38) -> List[str]:
        if len(smiles) <= width:
            return [smiles]
        lines: List[str] = []
        for i in range(0, len(smiles), width):
            lines.append(smiles[i : i + width])
        return lines

    def _render_with_pil_text(self, smiles: str) -> Tuple[Image.Image, str]:
        width = random.randint(720, 1200)
        height = random.randint(360, 900)
        background = (255, 255, 255)
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)

        font_size = random.randint(18, 34)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        lines = self._line_wrap_smiles(smiles)
        x = random.randint(20, 50)
        y = random.randint(20, 60)
        line_gap = random.randint(6, 12)

        for line in lines:
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y += font_size + line_gap
            if y >= height - (font_size + 10):
                break

        if random.random() < self.style_config.comment_prob:
            # Simulate random annotations in chemical drawings.
            draw.text(
                (random.randint(10, max(10, width - 120)), random.randint(8, 32)),
                f"{random.randint(1, 20)}{random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
                fill=(0, 0, 0),
                font=font,
            )

        if random.random() < 0.3:
            angle = random.uniform(-5.0, 5.0)
            image = image.rotate(angle, expand=True, fillcolor=background)

        if random.random() < self.style_config.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        if random.random() < self.style_config.noise_prob:
            arr = np.array(image, dtype=np.int16)
            noise = np.random.normal(0, random.uniform(2.0, 8.0), arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")

        return image, smiles

    def _add_explicit_hydrogen(self, indigo, mol) -> object:
        atoms = []
        for atom in mol.iterateAtoms():
            try:
                hs = atom.countImplicitHydrogens()
                if hs > 0:
                    atoms.append((atom, hs))
            except Exception:
                continue
        if atoms and random.random() < 0.2:
            atom, hs = random.choice(atoms)
            for _ in range(hs):
                h = mol.addAtom("H")
                h.addBond(atom, 1)
        return mol

    def _add_rgroup(self, mol, smiles: str) -> object:
        assert self._indigo is not None
        rgroup_symbols = self._indigo["RGROUP_SYMBOLS"]
        atoms = []
        for atom in mol.iterateAtoms():
            try:
                hs = atom.countImplicitHydrogens()
                if hs > 0:
                    atoms.append(atom)
            except Exception:
                continue
        if atoms and "*" not in smiles and random.random() < 0.5:
            atom = random.choice(atoms)
            symbol = random.choice(rgroup_symbols)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
        return mol

    @staticmethod
    def _gen_rand_condensed() -> str:
        symbols = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br"]
        parts: List[str] = []
        for i in range(4):
            if i >= 1 and random.random() < 0.7:
                break
            token = random.choice(symbols)
            if random.random() < 0.25:
                token = f"({token}{random.randint(2, 9)})"
            if random.random() < 0.3:
                token += str(random.randint(2, 9))
            parts.append(token)
        return "".join(parts) if parts else "CH3"

    def _add_rand_condensed(self, mol) -> object:
        atoms = []
        for atom in mol.iterateAtoms():
            try:
                hs = atom.countImplicitHydrogens()
                if hs > 0:
                    atoms.append(atom)
            except Exception:
                continue
        if atoms and random.random() < 0.5:
            atom = random.choice(atoms)
            symbol = self._gen_rand_condensed()
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
        return mol

    def _add_functional_group(self, indigo, mol) -> object:
        assert self._indigo is not None
        substitutions = [sub for sub in self._indigo["SUBSTITUTIONS"]]
        random.shuffle(substitutions)
        if random.random() > 0.8:
            return mol
        for sub in substitutions[:20]:
            try:
                query = indigo.loadSmarts(sub.smarts)
                matcher = indigo.substructureMatcher(mol)
                matched_atoms_ids = set()
                for match in matcher.iterateMatches(query):
                    if random.random() >= sub.probability:
                        continue
                    atoms = []
                    atoms_ids = set()
                    for item in query.iterateAtoms():
                        atom = match.mapAtom(item)
                        atoms.append(atom)
                        atoms_ids.add(atom.index())
                    if matched_atoms_ids.intersection(atoms_ids):
                        continue
                    abbrv = random.choice(sub.abbrvs)
                    superatom = mol.addAtom(abbrv)
                    for atom in atoms:
                        for nei in atom.iterateNeighbors():
                            if nei.index() not in atoms_ids and nei.symbol() != "H":
                                superatom.addBond(nei, nei.bond().bondOrder())
                    for atom_id in atoms_ids:
                        mol.getAtom(atom_id).remove()
                    matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
            except Exception:
                continue
        return mol

    def _set_indigo_style_options(self, indigo, mol) -> None:
        assert self._indigo is not None
        style = self.style_config.render_style
        indigo.setOption("render-output-format", "png")
        indigo.setOption("render-background-color", "1,1,1")
        indigo.setOption("render-stereo-style", "none")
        indigo.setOption("render-label-mode", "hetero")
        indigo.setOption("render-font-family", "Arial")

        if style == "chemdraw_like":
            # Clean black-and-white style similar to ChemDraw exports.
            indigo.setOption("render-relative-thickness", random.uniform(0.8, 1.2))
            indigo.setOption("render-bond-line-width", random.uniform(1.0, 1.8))
            indigo.setOption("render-font-family", random.choice(["Arial", "Helvetica"]))
            indigo.setOption("render-label-mode", "hetero")
            # ChemDraw-like display should keep hetero-atom hydrogens visible (e.g. N-H).
            indigo.setOption("render-implicit-hydrogens-visible", True)
            return

        # style == "molscribe_default"
        thickness = random.uniform(0.5, 2.0)
        indigo.setOption("render-relative-thickness", thickness)
        indigo.setOption("render-bond-line-width", random.uniform(1.0, max(1.2, 4.0 - thickness)))
        if random.random() < 0.5:
            indigo.setOption("render-font-family", random.choice(["Arial", "Times", "Courier", "Helvetica"]))
        indigo.setOption("render-label-mode", random.choice(["hetero", "terminal-hetero"]))
        indigo.setOption("render-implicit-hydrogens-visible", random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption("render-stereo-style", "old")
        if random.random() < 0.2:
            indigo.setOption("render-atom-ids-visible", True)

        if random.random() < self.style_config.comment_prob:
            indigo.setOption(
                "render-comment",
                str(random.randint(1, 20)) + random.choice(string.ascii_letters),
            )
            indigo.setOption("render-comment-font-size", random.randint(40, 60))
            indigo.setOption("render-comment-alignment", random.choice([0, 0.5, 1]))
            indigo.setOption("render-comment-position", random.choice(["top", "bottom"]))
            indigo.setOption("render-comment-offset", random.randint(2, 30))

        if random.random() < self.style_config.color_prob:
            indigo.setOption("render-coloring", True)
            if random.random() < 0.7:
                indigo.setOption("render-base-color", random.choice(list(self._indigo["COLORS"].values())))
            if random.random() < 0.5:
                indigo.setOption("render-highlight-color-enabled", True)
                indigo.setOption("render-highlight-color", random.choice(list(self._indigo["COLORS"].values())))
            if random.random() < 0.5:
                indigo.setOption("render-highlight-thickness-enabled", True)
            for atom in mol.iterateAtoms():
                if random.random() < 0.1:
                    atom.highlight()

    def _render_with_indigo(self, smiles: str) -> Tuple[Image.Image, str]:
        assert self._indigo is not None
        style = self.style_config.render_style
        Indigo = self._indigo["Indigo"]
        IndigoRenderer = self._indigo["IndigoRenderer"]
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        mol = indigo.loadMolecule(smiles)

        if self.style_config.mol_augment and style == "molscribe_default":
            if random.random() < 0.8:
                mol.dearomatize()
            else:
                mol.aromatize()
            smiles = mol.canonicalSmiles()
            mol = self._add_explicit_hydrogen(indigo, mol)
            mol = self._add_rgroup(mol, smiles)
            if self.style_config.include_condensed:
                mol = self._add_rand_condensed(mol)
            mol = self._add_functional_group(indigo, mol)

        self._set_indigo_style_options(indigo, mol)
        buf = renderer.renderToBuffer(mol)
        image_bgr = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Indigo failed to decode rendered image for: {smiles}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb, mode="RGB")

        if style == "molscribe_default":
            if random.random() < self.style_config.blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.6)))
            if random.random() < self.style_config.noise_prob:
                arr = np.array(image, dtype=np.int16)
                noise = np.random.normal(0, random.uniform(1.0, 6.0), arr.shape).astype(np.int16)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(arr, mode="RGB")
        elif style == "chemdraw_like":
            if random.random() < 0.08:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.08, 0.2)))

        try:
            rendered_smiles = mol.smiles()
            if " " in rendered_smiles:
                rendered_smiles = rendered_smiles.split(" ")[0]
        except Exception:
            rendered_smiles = smiles
        return image, rendered_smiles

    def _render_with_rdkit(self, smiles: str) -> Tuple[Image.Image, str]:
        assert self._rdkit is not None
        style = self.style_config.render_style
        Chem = self._rdkit["Chem"]
        AllChem = self._rdkit["AllChem"]
        rdMolDraw2D = self._rdkit["rdMolDraw2D"]
        PrepareAndDrawMolecule = self._rdkit["PrepareAndDrawMolecule"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for RDKit: {smiles}")

        if style == "chemdraw_like":
            # Force hetero atom hydrogens to explicit labels for cleaner ChemDraw-like readability.
            rw = Chem.RWMol(mol)
            changed = False
            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() not in (7, 8, 15, 16):
                    continue
                implicit_h = atom.GetNumImplicitHs()
                if implicit_h > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + implicit_h)
                    atom.SetNoImplicit(True)
                    changed = True
            if changed:
                candidate = rw.GetMol()
                try:
                    Chem.SanitizeMol(candidate)
                    mol = candidate
                except Exception:
                    pass

        if self.style_config.mol_augment and style == "molscribe_default" and random.random() < 0.15:
            mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        if style == "chemdraw_like":
            width = random.choice([640, 768, 896])
            height = random.choice([480, 576, 640])
        else:
            width = random.choice([512, 640, 768, 896, 1024])
            height = random.choice([384, 512, 640, 768])
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()

        if style == "chemdraw_like":
            if hasattr(opts, "bondLineWidth"):
                opts.bondLineWidth = random.uniform(1.2, 1.8)
            if hasattr(opts, "addAtomIndices"):
                opts.addAtomIndices = False
            if hasattr(opts, "fixedBondLength"):
                opts.fixedBondLength = random.uniform(28.0, 34.0)
            if hasattr(opts, "baseFontSize"):
                opts.baseFontSize = random.uniform(0.55, 0.75)
            if hasattr(opts, "useBWAtomPalette"):
                opts.useBWAtomPalette()
            if hasattr(opts, "rotate"):
                opts.rotate = random.uniform(-4.0, 4.0)
        elif not self.style_config.default_option:
            # These style switches are inspired by MolScribe/Indigo dynamic rendering.
            if hasattr(opts, "bondLineWidth"):
                opts.bondLineWidth = random.uniform(1.0, 2.8)
            if hasattr(opts, "addAtomIndices"):
                opts.addAtomIndices = random.random() < 0.2
            if hasattr(opts, "rotate"):
                opts.rotate = random.uniform(-15.0, 15.0)
            if random.random() < self.style_config.color_prob and hasattr(opts, "setSymbolColour"):
                opts.setSymbolColour((random.random(), random.random(), random.random()))
            if random.random() < 0.3 and hasattr(opts, "useBWAtomPalette"):
                opts.useBWAtomPalette()
            if hasattr(opts, "fixedBondLength") and random.random() < 0.5:
                opts.fixedBondLength = random.uniform(20.0, 38.0)
            if hasattr(opts, "baseFontSize"):
                opts.baseFontSize = random.uniform(0.45, 0.9)

        if style == "chemdraw_like":
            kekulize = True
        else:
            kekulize = bool(random.getrandbits(1))
        PrepareAndDrawMolecule(drawer, mol, kekulize=kekulize)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        if style == "molscribe_default" and random.random() < self.style_config.comment_prob:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", random.randint(20, 30))
            except Exception:
                font = ImageFont.load_default()
            draw.text(
                (random.randint(8, 20), random.randint(8, 24)),
                f"{random.randint(1, 20)}{random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
                fill=(0, 0, 0),
                font=font,
            )
        if style == "molscribe_default":
            if random.random() < self.style_config.blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.6)))
            if random.random() < self.style_config.noise_prob:
                arr = np.array(image, dtype=np.int16)
                noise = np.random.normal(0, random.uniform(1.0, 6.0), arr.shape).astype(np.int16)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(arr, mode="RGB")
        elif style == "chemdraw_like":
            if random.random() < 0.08:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.08, 0.2)))

        try:
            rendered_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=False)
        except Exception:
            rendered_smiles = smiles
        return image, rendered_smiles

    def render(self, smiles: str) -> Tuple[Image.Image, str]:
        smiles = (smiles or "").strip()
        if not smiles:
            raise ValueError("SMILES is empty.")
        if self.backend == "rdkit":
            return self._render_with_rdkit(smiles)
        if self.backend == "indigo":
            return self._render_with_indigo(smiles)
        if self.backend == "pil_text":
            return self._render_with_pil_text(smiles)
        raise RuntimeError(
            "No molecule rendering backend available. "
            "Install RDKit or configure local Indigo libs."
        )


class ChemConversationDataset(Dataset):
    """
    Dataset for DeepSeek-OCR2 style finetuning:
    each sample -> {"messages": [{"role": "<|User|>", ...}, {"role": "<|Assistant|>", ...}]}
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        instruction: str = DEFAULT_INSTRUCTION,
        style_config: Optional[MoleculeStyleConfig] = None,
        use_rendered_smiles_as_label: bool = True,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.smiles_col = _find_smiles_column(self.df)
        self.instruction = instruction
        self.renderer = MoleculeStyleRenderer(style_config or MoleculeStyleConfig())
        self.use_rendered_smiles_as_label = use_rendered_smiles_as_label

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        input_smiles = str(row[self.smiles_col])
        image, rendered_smiles = self.renderer.render(input_smiles)
        target_smiles = rendered_smiles if self.use_rendered_smiles_as_label else input_smiles

        return {
            "messages": [
                {
                    "role": "<|User|>",
                    "content": self.instruction,
                    "images": [image],
                },
                {
                    "role": "<|Assistant|>",
                    "content": target_smiles,
                },
            ],
            "meta": {
                "idx": idx,
                "input_smiles": input_smiles,
                "target_smiles": target_smiles,
                "source_csv": row.get("source_csv", ""),
            },
        }


def build_chem_conversation_dataset(
    csv_files: Union[str, Sequence[str]],
    *,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    instruction: str = DEFAULT_INSTRUCTION,
    style_config: Optional[MoleculeStyleConfig] = None,
    keep_columns: Optional[Sequence[str]] = None,
    use_rendered_smiles_as_label: bool = True,
) -> ChemConversationDataset:
    df = _read_and_merge_csvs(csv_files, keep_columns=keep_columns)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    return ChemConversationDataset(
        dataframe=df,
        instruction=instruction,
        style_config=style_config,
        use_rendered_smiles_as_label=use_rendered_smiles_as_label,
    )


def build_dataset_from_training_data_dir(
    training_data_root: str,
    *,
    include_keywords: Optional[Sequence[str]] = ("pubchem", "uspto_mol"),
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    instruction: str = DEFAULT_INSTRUCTION,
    style_config: Optional[MoleculeStyleConfig] = None,
    keep_columns: Optional[Sequence[str]] = None,
    use_rendered_smiles_as_label: bool = True,
) -> ChemConversationDataset:
    csv_files = discover_training_csvs(
        training_data_root=training_data_root,
        include_keywords=include_keywords,
    )
    if not csv_files:
        raise ValueError(f"No CSV files found under: {training_data_root}")

    return build_chem_conversation_dataset(
        csv_files=csv_files,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        instruction=instruction,
        style_config=style_config,
        keep_columns=keep_columns,
        use_rendered_smiles_as_label=use_rendered_smiles_as_label,
    )


def to_conversation_list(dataset: Dataset, limit: Optional[int] = None) -> List[Dict]:
    if limit is None:
        limit = len(dataset)
    return [dataset[i] for i in range(min(limit, len(dataset)))]


def make_style_config(
    style: str = "molscribe_default",
    mol_augment: bool = True,
    include_condensed: bool = True,
) -> MoleculeStyleConfig:
    """
    Quick preset helper.
    - molscribe_default: random augmented style close to MolScribe
    - chemdraw_like: cleaner black-and-white style
    """
    return MoleculeStyleConfig(
        render_style=style,
        mol_augment=mol_augment,
        include_condensed=include_condensed,
    )


def _sanitize_filename(text: str) -> str:
    safe_chars: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "sample"


def _resolve_sample_id(row: pd.Series, fallback_idx: int) -> str:
    for col in ID_CANDIDATE_COLUMNS:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            if raw != "":
                return _sanitize_filename(raw)
    return str(fallback_idx)


def _safe_read_csv(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _init_worker_renderer(style_config_dict: Dict) -> None:
    global _WORKER_RENDERER
    _WORKER_RENDERER = MoleculeStyleRenderer(MoleculeStyleConfig(**style_config_dict))


def _render_image_task(task: Tuple[str, str, str, bool]) -> Dict[str, str]:
    global _WORKER_RENDERER
    sample_id, smiles, img_path, overwrite = task
    if (not overwrite) and os.path.isfile(img_path):
        return {
            "id": sample_id,
            "smiles": smiles,
            "image_path": img_path,
            "status": "skipped_exists",
        }
    try:
        if _WORKER_RENDERER is None:
            raise RuntimeError("Worker renderer is not initialized.")
        image, rendered_smiles = _WORKER_RENDERER.render(smiles)
        image.save(img_path)
        return {
            "id": sample_id,
            "smiles": rendered_smiles,
            "image_path": img_path,
            "status": "ok",
        }
    except Exception:
        return {
            "id": sample_id,
            "smiles": smiles,
            "image_path": img_path,
            "status": "failed",
        }


def export_images_for_csv(
    csv_path: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    style_config: Optional[MoleculeStyleConfig] = None,
    num_workers: int = 1,
) -> Dict[str, Union[str, int]]:
    """
    Export molecule images for one CSV.
    Default output:
    - /path/to/train_200k.csv -> /path/to/train_200k/*.png
    """
    style_cfg = style_config or MoleculeStyleConfig()
    style_suffix = style_cfg.render_style

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = _safe_read_csv(csv_path)
    if frame.empty:
        if output_dir is None:
            base_dir = os.path.join(
                os.path.dirname(csv_path),
                os.path.splitext(os.path.basename(csv_path))[0],
            )
            out_dir = f"{base_dir}_{style_suffix}"
        else:
            out_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)
        return {
            "csv_path": csv_path,
            "output_dir": out_dir,
            "total_rows": 0,
            "saved_images": 0,
            "failed_images": 0,
        }

    smiles_col = _find_smiles_column(frame)
    frame = frame[frame[smiles_col].notna()].copy()
    frame[smiles_col] = frame[smiles_col].astype(str).str.strip()
    frame = frame[frame[smiles_col] != ""].reset_index(drop=True)

    if max_samples is not None:
        frame = frame.iloc[:max_samples].copy()

    if output_dir is None:
        base_dir = os.path.join(
            os.path.dirname(csv_path),
            os.path.splitext(os.path.basename(csv_path))[0],
        )
        out_dir = f"{base_dir}_{style_suffix}"
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    worker_count = max(1, int(num_workers))
    records: List[Dict[str, str]] = []
    tasks: List[Tuple[str, str, str, bool]] = []
    for i, row in frame.iterrows():
        smiles = str(row[smiles_col]).strip()
        sample_id = _resolve_sample_id(row, fallback_idx=i)
        img_path = os.path.join(out_dir, f"{sample_id}.png")
        tasks.append((sample_id, smiles, img_path, overwrite))

    if worker_count == 1:
        _init_worker_renderer(asdict(style_cfg))
        for task in tasks:
            records.append(_render_image_task(task))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker_renderer,
            initargs=(asdict(style_cfg),),
        ) as executor:
            for record in executor.map(_render_image_task, tasks, chunksize=16):
                records.append(record)

    saved = sum(1 for record in records if record["status"] == "ok")
    failed = sum(1 for record in records if record["status"] == "failed")

    map_path = os.path.join(out_dir, "image_index.csv")
    pd.DataFrame(records).to_csv(map_path, index=False)
    return {
        "csv_path": csv_path,
        "output_dir": out_dir,
        "total_rows": int(len(frame)),
        "saved_images": int(saved),
        "failed_images": int(failed),
    }


def export_images_from_training_data(
    training_data_root: str,
    include_keywords: Optional[Sequence[str]] = None,
    max_samples_per_csv: Optional[int] = None,
    overwrite: bool = False,
    style_config: Optional[MoleculeStyleConfig] = None,
    num_workers: int = 1,
) -> List[Dict[str, Union[str, int]]]:
    csv_files = discover_training_csvs(
        training_data_root=training_data_root,
        include_keywords=include_keywords,
    )
    reports: List[Dict[str, Union[str, int]]] = []
    for csv_path in csv_files:
        report = export_images_for_csv(
            csv_path=csv_path,
            output_dir=None,
            max_samples=max_samples_per_csv,
            overwrite=overwrite,
            style_config=style_config,
            num_workers=num_workers,
        )
        reports.append(report)
        print(
            f"[export] {os.path.basename(csv_path)} -> "
            f"{report['saved_images']} saved, {report['failed_images']} failed "
            f"(rows={report['total_rows']})"
        )
    return reports


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Generate molecule images from CSV.")
    parser.add_argument(
        "--style",
        type=str,
        default="molscribe_default",
        choices=["molscribe_default", "chemdraw_like"],
        help="Rendering style switch.",
    )
    parser.add_argument(
        "--mol_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable molecule-level augmentation (explicit H, R-group, etc.).",
    )
    parser.add_argument(
        "--include_condensed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable random condensed-group augmentation when mol_augment is enabled.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "training_data", "pubchem", "train_200k.csv"
        ),
        help="Input CSV file path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of molecules to export. Default: all rows in CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images if set.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview one dataset sample before export.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for image generation. 1 means single process.",
    )
    args = parser.parse_args()

    style_cfg = make_style_config(
        args.style,
        mol_augment=args.mol_augment,
        include_condensed=args.include_condensed,
    )

    if args.preview:
        ds = build_chem_conversation_dataset(
            csv_files=[args.csv],
            max_samples=1,
            shuffle=False,
            style_config=style_cfg,
        )
        sample = ds[0]
        print(f"Renderer backend: {ds.renderer.backend}")
        print(f"Dataset length: {len(ds)}")
        print(sample["messages"][0]["content"])
        print(sample["messages"][1]["content"][:120])

    report = export_images_for_csv(
        csv_path=args.csv,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        style_config=style_cfg,
        num_workers=args.workers,
    )
    print(report)
