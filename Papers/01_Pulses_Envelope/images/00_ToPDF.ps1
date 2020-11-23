$items = Get-ChildItem | Where-Object {$_.Extension -in ".svg"} #-Encoding UTF8

# $root = Get-Location

foreach ($item in $items) {
  $item.Name
  inkscape --export-type="pdf" --export-pdf-version="1.5" $item.Name
  
  # inkscape --batch-process --export-type="pdf" --export-pdf-version="1.5" -g --actions="EditSelectAll;SelectionUnGroup;EditSelectAll;SelectionUnGroup;EditSelectAll;SelectionUnGroup" $item.Name
  # $item.Name | inkscape --pipe --export-filename=($item.BaseName + ".pdf")
}