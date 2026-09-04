from typing import Optional
from pathlib import Path
from os.path import relpath
import platform

def normPath(path: str | Path, basePath: str | Path | None = None) -> Path:
	if basePath is None:
		basePath = Path.cwd()
	basePath = Path(basePath).resolve()
	fullPath = (basePath / path).resolve()
	if not fullPath.exists():
		raise FileNotFoundError(f"Path does not exist: {fullPath}")
	return Path(relpath(fullPath, basePath))

def validatePath(path: str | Path, mustBeDir: bool = False) -> Path | None:
	p = Path(path)
	
	if not p.exists():
		return None
	
	if mustBeDir and not p.is_dir():
		return None
	
	return normPath(p)


def globPath(pattern: str | Path) -> list[Path]:
	p = Path(pattern)
	parent = p.parent if p.parent != Path() else Path(".")
	globPattern = p.name
	
	verifiedDir = validatePath(parent, mustBeDir=True)
	
	if verifiedDir is None:
		return []
	
	return sorted(normPath(match) for match in verifiedDir.glob(globPattern))

def getPlatform() -> str:
	system = platform.system()
	if system == "Windows":
		return "win"
	elif system == "Darwin":
		return "macos"
	else:
		return "linux"

def resolveIsDir(path: Path, rawPath: str, override: bool | None = None) -> bool:
	"""
	Determine whether a path refers to a directory.
	Priority: explicit override > filesystem check > trailing separator > suffix guess.
	"""
	if override is not None:
		return override
	
	if path.exists():
		return path.is_dir()
	
	# Trailing slash is an authoritative signal from the user
	if rawPath.endswith(("/", "\\")):
		return True
	
	# Last-resort guess: extension-less names are more often directories,
	# but this is genuinely ambiguous (Makefile, README, my.project.folder, etc.)
	# Caller should pass `override` if they know.
	return not path.suffix


def pathToJson(path: str | Path, isDir: bool | None = None) -> dict:
	rawPath = str(path)
	path = Path(path)
	currentPlatform = getPlatform()
	
	isDir = resolveIsDir(path, rawPath, isDir)
	
	if isDir:
		name = ""
		dirPath = path
	else:
		name = path.name
		dirPath = path.parent
	
	if dirPath.is_absolute():
		if currentPlatform == "win":
			drive = dirPath.drive  # "C:", "\\wsl$\Ubuntu", "\\server\share", or ""
			dirParts = list(dirPath.parts[1:])
			
			if drive.startswith("\\\\"):
				# UNC path — parse \\host\share
				uncBits = drive.lstrip("\\").split("\\")
				host  = uncBits[0] if len(uncBits) > 0 else ""
				share = uncBits[1] if len(uncBits) > 1 else ""
				
				if host.lower() in ("wsl$", "wsl.localhost"):
					# WSL: share is the distro name, and the path inside is
					# that distro's root filesystem.
					prefix = {
						"win":   f"\\\\{host}\\{share}\\",
						"macos": "/",
						"linux": "/",
					}
				else:
					# Generic SMB / network share
					safeShare = share or "share"
					prefix = {
						"win":   f"\\\\{host}\\{share}\\",
						"macos": f"/Volumes/{safeShare}",
						"linux": f"/mnt/{safeShare.lower().replace(' ', '_')}",
					}
			else:
				# Regular drive letter
				letter = drive[0] if drive else "C"
				prefix = {
					"win":   f"{letter}:\\",
					"macos": f"/Volumes/{letter}",
					"linux": f"/mnt/{letter.lower()}",
				}
		
		elif currentPlatform == "macos":
			if len(dirPath.parts) > 2 and dirPath.parts[1] == "Volumes":
				driveName = dirPath.parts[2]
				dirParts = list(dirPath.parts[3:])
				prefix = {
					"win":   f"{driveName[0].upper()}:\\",
					"macos": f"/Volumes/{driveName}",
					"linux": f"/mnt/{driveName.lower().replace(' ', '_')}",
				}
			else:
				dirParts = list(dirPath.parts[1:])
				prefix = {
					"win":   "C:\\",
					"macos": "/Volumes/Macintosh HD",
					"linux": "/",
				}
		
		else:  # linux
			if len(dirPath.parts) > 2 and dirPath.parts[1] == "mnt":
				driveName = dirPath.parts[2]
				dirParts = list(dirPath.parts[3:])
				prefix = {
					"win":   f"{driveName[0].upper()}:\\",
					"macos": f"/Volumes/{driveName}",
					"linux": f"/mnt/{driveName}",
				}
			else:
				dirParts = list(dirPath.parts[1:])
				prefix = {
					"win":   "C:\\",
					"macos": "/Volumes/Macintosh HD",
					"linux": "/",
				}
	
	else:
		dirParts = list(dirPath.parts)
		if dirParts and dirParts[0] == ".":
			dirParts = dirParts[1:]
		prefix = {"win": ".", "macos": ".", "linux": "."}
	
	return {
		"prefix": prefix,
		"dir": dirParts,
		"name": name,
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

def getMoments(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"cfrad.*.nc")

def getIQ(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"frxxIQ.*.nc")

def getSpectra(path: Optional[str | Path]= None) -> list[Path]:
	if path is None:
		path = Path(".")

	return globPath(Path(path)/"frxxSP.*.nc")

