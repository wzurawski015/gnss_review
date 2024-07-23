# projekt/makefile
# project/makefile

all:
	pdflatex main_document.tex
	bibtex main_document
	pdflatex main_document.tex
	pdflatex main_document.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out
