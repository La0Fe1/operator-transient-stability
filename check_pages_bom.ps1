$ErrorActionPreference = 'Stop'
$src = 'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce_post.docx'
$pdf = 'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce_word.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($src, $false, $true)
    $pages = $doc.ComputeStatistics(2)  # wdStatisticPages
    Write-Output ("PAGES: " + $pages)
    # also words
    $words = $doc.ComputeStatistics(0)  # wdStatisticWords
    Write-Output ("WORDS: " + $words)
    $doc.SaveAs([ref]$pdf, [ref]17)  # wdFormatPDF
    Write-Output "PDF exported"
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
