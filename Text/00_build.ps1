# latexmk -lualatex -f -interaction=nonstopmode THESIS.tex

$name = "THESIS"
lualatex.exe $name -f -interaction=nonstopmode -file-line-error
biber.exe $name
makeglossaries.exe $name
lualatex.exe $name -f -interaction=nonstopmode -file-line-error
lualatex.exe $name -f -interaction=nonstopmode -file-line-error

# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx