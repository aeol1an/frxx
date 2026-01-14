from typing import Optional
from pathlib import Path
import platform

from ..core import frxxData

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

def getPlatform() -> str:
	system = platform.system()
	if system == "Windows":
		return "win"
	elif system == "Darwin":
		return "macos"
	else:
		return "linux"

def pathToJson(path: str | Path) -> dict:
	path = Path(path)
	currentPlatform = getPlatform()
	
	if path.is_absolute():
		if currentPlatform == "win":
			drive = path.drive
			parts = list(path.parts[1:])
			prefix = {
				"win": drive,
				"macos": f"/Volumes/{drive[0]}",
				"linux": f"/mnt/{drive[0].lower()}"
			}
		elif currentPlatform == "macos":
			# Path looks like /Volumes/DriveName/folder/file or /Users/...
			if path.parts[1] == "Volumes":
				driveName = path.parts[2]
				parts = list(path.parts[3:])
			else:
				# Main drive (Macintosh HD)
				driveName = "Macintosh HD"
				parts = list(path.parts[1:])
			
			prefix = {
				"win": f"{driveName[0].upper()}:",
				"macos": f"/Volumes/{driveName}",
				"linux": f"/mnt/{driveName.lower().replace(' ', '_')}"
			}
		else:  # linux
			# Path looks like /mnt/drivename/folder/file or /home/...
			if path.parts[1] == "mnt":
				driveName = path.parts[2]
				parts = list(path.parts[3:])
			else:
				driveName = "root"
				parts = list(path.parts[1:])
			
			prefix = {
				"win": f"{driveName[0].upper()}:",
				"macos": f"/Volumes/{driveName}",
				"linux": f"/mnt/{driveName}"
			}
	else:
		parts = list(path.parts)
		if parts and parts[0] == ".":
			parts = parts[1:]
		
		prefix = {
			"win": ".",
			"macos": ".",
			"linux": "."
		}
	
	return {
		"prefix": prefix,
		"path": parts
	}


def jsonToPath(pathJson: dict) -> Path:
	currentPlatform = getPlatform()
	prefix = pathJson["prefix"][currentPlatform]
	
	return Path(prefix, *pathJson["path"])


def getCfrad(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"cfrad.*.nc")

