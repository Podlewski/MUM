$FilePath = "kmeans.txt"
$datasets = 4 

"Losowanie centrow" >> $FilePath

For ($i = 1; $i -le $datasets; $i++)
{
    "`nData set $i" >> $FilePath

    $array = @(1, 2, 5, 10, 20)
    For ($j = 0; $j -lt $array.length; $j++)
    {
        $var = $array[$j]
        "$var`n" | Out-File $FilePath -Append -NoNewline
        python ./main.py -d $i -a 2 --fixed -p $array[$j] 300 None >> $FilePath
        "`n" | Out-File $FilePath -Append -NoNewline
    }
}

"`nMaksymalne iteracje" >> $FilePath

For ($i = 1; $i -lt $datasets; $i++)
{
    "`nData set $i" >> $FilePath

    $array = @(5, 10, 20, 25, 50, 100, 300, 600)
    For ($j = 0; $j -le $array.length; $j++)
    {
        $var = $array[$j]
        "$var`n" | Out-File $FilePath -Append -NoNewline
        python ./main.py -d $i -a 2 --fixed -p 10 $array[$j] None >> $FilePath
        "`n" | Out-File $FilePath -Append -NoNewline
    }
}

"`nInicjalizacja centrow" >> $FilePath

For ($i = 1; $i -le $datasets; $i++)
{
    "`nData set $i" >> $FilePath

    $array = @("None", 0, 42)
    For ($j = 0; $j -lt $array.length; $j++)
    {
        $var = $array[$j]
        "$var`n" | Out-File $FilePath -Append -NoNewline
        python ./main.py -d $i -a 2 --fixed -p 10 300 $array[$j] >> $FilePath
        "`n" | Out-File $FilePath -Append -NoNewline
    }
}
