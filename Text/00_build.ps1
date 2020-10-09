# latexmk -lualatex -f -interaction=nonstopmode THESIS.tex

$name = "THESIS"
lualatex $name -f -interaction=nonstopmode -file-line-error
biber $name
makeglossaries $name
lualatex $name -f -interaction=nonstopmode -file-line-error
lualatex $name -f -interaction=nonstopmode -file-line-error

# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx