$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName Microsoft.VisualBasic

$desktop = 'C:\Users\佬肥\Desktop'
$files = @(
 'A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_revised.pdf',
 'A Physics-Encoded Neural Operator with Conformal Trajectory Bands_Liu_Wang_Liu_Zeng_revised.docx',
 'A Physics-Informed Neural Operator with Conformal Certification for Fast Transient Stability Assessment_Liu_Wang_Liu_Zeng.docx',
 'A Physics-Informed Neural Operator with Conformal Certification for Fast Transient Stability Assessment_Liu_Wang_Liu_Zeng.pdf',
 '面向快速暂态稳定评估的物理信息神经算子与共形认证_刘振宇_王佳乐_刘家永_曾志诚.pdf',
 '算子 aigc.pdf',
 '算子 aigc2.0.pdf',
 '算子8turn AIGC.pdf',
 '算子 picture2.pdf',
 '算子picture2_矢量图源文件.zip',
 '论文完整审稿报告_算子picture2.docx',
 '_pdf_dump.txt',
 '_docx_dump.txt',
 '_mpce_guidelines.txt',
 '_eval_report1.txt',
 '_springer_mpce.html'
)
$dirs = @(
 '算子picture2_矢量图源文件'
)

foreach ($f in $files) {
    $p = Join-Path $desktop $f
    if (Test-Path $p) {
        try {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p, 'OnlyErrorDialogs', 'SendToRecycleBin')
            Write-Output ("RECYCLED: " + $f)
        } catch {
            Write-Output ("FAILED: " + $f + " -> " + $_.Exception.Message)
        }
    } else {
        Write-Output ("MISSING: " + $f)
    }
}
foreach ($d in $dirs) {
    $p = Join-Path $desktop $d
    if (Test-Path $p) {
        try {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($p, 'OnlyErrorDialogs', 'SendToRecycleBin')
            Write-Output ("RECYCLED DIR: " + $d)
        } catch {
            Write-Output ("FAILED DIR: " + $d + " -> " + $_.Exception.Message)
        }
    } else {
        Write-Output ("MISSING DIR: " + $d)
    }
}
Write-Output 'DONE'
