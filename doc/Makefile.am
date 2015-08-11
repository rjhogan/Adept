dist_pkgdata_DATA = adept_documentation.pdf
EXTRA_DIST = adept_documentation.tex

adept_documentation.pdf: adept_documentation.tex
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex

documentation: adept_documentation.pdf

.PHONY: documentation
