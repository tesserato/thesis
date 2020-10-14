# latexmk -lualatex -f -interaction=nonstopmode THESIS.tex

$name = "SoundRepresentation"
lualatex --shell-escape $name -f -interaction=nonstopmode -file-line-error
biber $name
# makeglossaries $name
lualatex --shell-escape $name -f -interaction=nonstopmode -file-line-error
lualatex --shell-escape $name -f -interaction=nonstopmode -file-line-error

# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx