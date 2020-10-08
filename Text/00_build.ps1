latexmk -lualatex -f -interaction=nonstopmode PTBR.tex

# $name = "Envelope"
# lualatex.exe $name -interaction=nonstopmode -f
# biber $name -interaction=nonstopmode
# lualatex.exe $name -interaction=nonstopmode
# lualatex.exe $name -interaction=nonstopmode

# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx