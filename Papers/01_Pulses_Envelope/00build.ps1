# $name = "00ElsevierPulsesEnvelope"
$name = "Response to Reviewers"
pdflatex $name
bibtex $name
pdflatex $name
pdflatex $name
pdflatex $name
# pandoc PulsesEnvelope.tex --bibliography=bibli.bib -o PulsesEnvelope.docx