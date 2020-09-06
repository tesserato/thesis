# $name = "PulsesEnvelope"
# pdflatex $name
# biber $name
# pdflatex $name
# pdflatex $name

pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx