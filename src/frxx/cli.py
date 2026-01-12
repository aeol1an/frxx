import argparse

def main() -> None:
	parser = argparse.ArgumentParser(
		description = "CLI/GUI for frxx.\n"
		"Running with no arguments looks for cfradial files (cfrad.*.nc)"
		" in the working directory, or a valid frxx_cases directory.\n"
		"GUI uses requires the frxx-view package."
		"A frxx_cases directory (will) enables spectral processing within the GUI."
	)
    
	parser.add_argument(
		"-i",
		"--init",
		type=str,
		nargs="?",
		const=".",
		help="Create empty frxx_cases directory. "
		"Basically just does \"mkdir frxx_cases\" in CWD or passed directory."
	)

	parser.add_argument(
		
	)

	args = parser.parse_args()



if __name__ == "__main__":
	main()