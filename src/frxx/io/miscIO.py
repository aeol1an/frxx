from typing import Optional
from pathlib import Path
import platform

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

def get_platform() -> str:
	"""Get normalized platform name."""
	system = platform.system()
	if system == "Windows":
		return "win"
	elif system == "Darwin":
		return "macos"
	else:
		return "linux"

def path_to_json(path: str | Path) -> dict:
    """
    Convert a Path to a cross-platform JSON representation.
    """
    path = Path(path)
    current_platform = get_platform()
    
    if path.is_absolute():
        if current_platform == "win":
            drive = path.drive
            parts = list(path.parts[1:])
            prefix = {
                "win": drive,
                "macos": f"/Volumes/{drive[0]}",
                "linux": f"/mnt/{drive[0].lower()}"
            }
        elif current_platform == "macos":
            # Path looks like /Volumes/DriveName/folder/file or /Users/...
            if path.parts[1] == "Volumes":
                drive_name = path.parts[2]
                parts = list(path.parts[3:])
            else:
                # Main drive (Macintosh HD)
                drive_name = "Macintosh HD"
                parts = list(path.parts[1:])
            
            prefix = {
                "win": f"{drive_name[0].upper()}:",
                "macos": f"/Volumes/{drive_name}",
                "linux": f"/mnt/{drive_name.lower().replace(' ', '_')}"
            }
        else:  # linux
            # Path looks like /mnt/drivename/folder/file or /home/...
            if path.parts[1] == "mnt":
                drive_name = path.parts[2]
                parts = list(path.parts[3:])
            else:
                drive_name = "root"
                parts = list(path.parts[1:])
            
            prefix = {
                "win": f"{drive_name[0].upper()}:",
                "macos": f"/Volumes/{drive_name}",
                "linux": f"/mnt/{drive_name}"
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


def json_to_path(path_json: dict) -> Path:
    """
    Convert a JSON path representation to a Path for the current OS.
    """
    current_platform = get_platform()
    prefix = path_json["prefix"][current_platform]
    
    return Path(prefix, *path_json["path"])

def getCfrad(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"cfrad.*.nc")

