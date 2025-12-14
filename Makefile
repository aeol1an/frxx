PACKAGE=frxx

# The default target runs when you just type 'make'
.PHONY: all
all: build

# Build the package
# 1. Runs dot_clean to merge/remove macOS attribute files
# 2. Runs the python build command

UNAME := $(shell uname)

.PHONY: build
build:
ifeq ($(UNAME), Darwin)
	@echo "Building package (macOS workaround)..."
	dot_clean .
	rm -rf /tmp/${PACKAGE}-build
	mkdir -p /tmp/${PACKAGE}-build
	rsync -a --exclude='.*' --exclude='*.egg-info' --exclude='dist' --exclude='build' --exclude='__pycache__' . /tmp/${PACKAGE}-build/
	cd /tmp/${PACKAGE}-build && python -m build
	mkdir -p dist
	cp /tmp/${PACKAGE}-build/dist/* dist/
	find dist -name '._*' -delete
	rm -rf ./build /tmp/${PACKAGE}-build
	dot_clean .
	@echo "Build complete."
else
	@echo "Building package..."
	python -m build
	rm -rf ./build
	@echo "Build complete."
endif

# Clean up artifacts
# Only run this when you specifically want to wipe the slate clean
.PHONY: clean
clean:
	@echo "Removing build artifacts..."
ifeq ($(UNAME), Darwin)
	dot_clean .
	rm -rf /tmp/${PACKAGE}-build
endif
	rm -rf ./dist ./build
	find . -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +