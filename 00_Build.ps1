# latexmk -lualatex -f -interaction=nonstopmode THESIS.tex

$name = "THESIS"
lualatex -interaction=nonstopmode -file-line-error $name
biber $name
makeglossaries $name
lualatex -interaction=nonstopmode -file-line-error $name
lualatex -interaction=nonstopmode -file-line-error $name
lualatex -interaction=nonstopmode -file-line-error $name

# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx