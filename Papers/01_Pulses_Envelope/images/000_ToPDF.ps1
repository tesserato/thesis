$items = Get-ChildItem -Recurse | Where-Object {$_.Extension -in ".svg"} #-Encoding UTF8
$ErrorActionPreference = "Stop"

foreach ($item in $items) {
  # Test-Path $item.Name.replace($item.Extension,".pdf")

  # $pdf = $item.Name.replace($item.Extension,".pdf")
  # $pdf
  # if (-Not (Test-Path $pdf)){
  $item.Name
  $interm = ("___" + $item.Name)
  scour -i $item.Name -o $interm --enable-viewboxing --enable-id-stripping --enable-comment-stripping --shorten-ids --indent=none
  inkscape --export-type="pdf" --export-pdf-version="1.5" $interm -o $item.Name
  Remove-Item $interm
  Remove-Item $item.Name
  # }
  
  # inkscape --batch-process --export-type="pdf" --export-pdf-version="1.5" -g --actions="EditSelectAll;SelectionUnGroup;EditSelectAll;SelectionUnGroup;EditSelectAll;SelectionUnGroup" $item.Name
  # $item.Name | inkscape --pipe --export-filename=($item.BaseName + ".pdf")
}