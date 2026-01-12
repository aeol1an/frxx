from typing import Optional
from pathlib import Path

from ..core import data

def validatePath(path: str | Path, mustBeDir: bool = False) -> Path | None:
	p = Path(path)
	
	if not p.exists():
		return None
	
	if mustBeDir and not p.is_dir():
		return None
	
	return p.resolve()


def globPath(pattern: str | Path) -> list[Path]:
	p = Path(pattern)
	parent = p.parent if p.parent != Path() else Path(".")
	globPattern = p.name
	
	verifiedDir = validatePath(parent, mustBeDir=True)
	
	if verifiedDir is None:
		return []
	
	return sorted(match.resolve() for match in verifiedDir.glob(globPattern))

def getCfrad(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"cfrad.*.nc")