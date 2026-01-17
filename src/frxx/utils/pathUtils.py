from typing import Optional
from pathlib import Path
import platform

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
	
	# Determine if path is a directory or file
	# If path exists, check directly; otherwise, assume it's a file if it has a suffix
	if path.exists():
		isDir = path.is_dir()
	else:
		isDir = not path.suffix
	
	if isDir:
		name = ""
		dirPath = path
	else:
		name = path.name
		dirPath = path.parent
	
	if dirPath.is_absolute():
		if currentPlatform == "win":
			drive = dirPath.drive
			dirParts = list(dirPath.parts[1:])
			prefix = {
				"win": drive,
				"macos": f"/Volumes/{drive[0]}",
				"linux": f"/mnt/{drive[0].lower()}"
			}
		elif currentPlatform == "macos":
			if dirPath.parts[1] == "Volumes":
				driveName = dirPath.parts[2]
				dirParts = list(dirPath.parts[3:])
			else:
				driveName = "Macintosh HD"
				dirParts = list(dirPath.parts[1:])
			
			prefix = {
				"win": f"{driveName[0].upper()}:",
				"macos": f"/Volumes/{driveName}",
				"linux": f"/mnt/{driveName.lower().replace(' ', '_')}"
			}
		else:  # linux
			if dirPath.parts[1] == "mnt":
				driveName = dirPath.parts[2]
				dirParts = list(dirPath.parts[3:])
			else:
				driveName = "root"
				dirParts = list(dirPath.parts[1:])
			
			prefix = {
				"win": f"{driveName[0].upper()}:",
				"macos": f"/Volumes/{driveName}",
				"linux": f"/mnt/{driveName}"
			}
	else:
		dirParts = list(dirPath.parts)
		if dirParts and dirParts[0] == ".":
			dirParts = dirParts[1:]
		
		prefix = {
			"win": ".",
			"macos": ".",
			"linux": "."
		}
	
	return {
		"prefix": prefix,
		"dir": dirParts,
		"name": name
	}

def jsonToPath(pathJson: dict) -> Path:
	currentPlatform = getPlatform()
	prefix = pathJson["prefix"][currentPlatform]
	if pathJson["name"]:
		return Path(prefix, *pathJson["dir"], pathJson["name"])
	else:
		return Path(prefix, *pathJson["dir"])

def editPathJsonPrefix(pathJson: dict, platform: str, prefix: str) -> dict:
	if platform not in ('win', 'macos', 'linux'):
		raise ValueError("Platform must be 'win', 'macos', or 'linux'.")
	_ = Path(prefix)  # validate prefix is a valid path string
	
	return {
		"prefix": {**pathJson["prefix"], platform: prefix},
		"dir": pathJson["dir"],
		"name": pathJson["name"]
	}

def editPathJsonDir(pathJson: dict, newDirPath: str | Path) -> dict:
	if not validatePath(newDirPath, mustBeDir=True):
		raise ValueError(f"Invalid directory path: {newDirPath}")
	
	newDirJson = pathToJson(Path(newDirPath))
	
	return {
		"prefix": newDirJson["prefix"],
		"dir": newDirJson["dir"],
		"name": pathJson["name"]
	}

def editPathJsonName(pathJson: dict, newName: str) -> dict:
	return {
		"prefix": pathJson["prefix"],
		"dir": pathJson["dir"],
		"name": newName
	}

def pathJsonEqual(pathJson1: dict, pathJson2: dict, prefixReqdEqual: bool = True) -> bool:
	if prefixReqdEqual:
		for platform in ("win", "macos", "linux"):
			if not (pathJson1["prefix"][platform] == pathJson2["prefix"][platform]):
				return False
	if not (jsonToPath(pathJson1).resolve() == jsonToPath(pathJson2).resolve()):
		return False
	return True

def getCfrad(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"cfrad.*.nc")

