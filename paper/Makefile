.PHONY: all clean

all: main.pdf
	cp -f main.pdf PiT-Mem_PVLDB.pdf

main.pdf: main.tex pvldb.sty acmart.cls generated_metrics.tex generated_tables.tex references.bib
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
