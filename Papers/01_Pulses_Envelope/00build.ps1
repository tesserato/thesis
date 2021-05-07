$name = "00ElsevierPulsesEnvelope"
pdflatex $name
bibtex $name
pdflatex $name
pdflatex $name
pdflatex $name
# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx