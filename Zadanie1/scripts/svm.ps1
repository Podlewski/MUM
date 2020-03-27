$FilePath = "svm.txt"

"Zbior treningowy" >> $FilePath

For ($i = 1; $i -le 3; $i++)
{
    "`nData set $i" >> $FilePath
    For ($j = 60; $j -le 90; $j = $j + 5)
    {
        "$j`t" | Out-File $FilePath -Append -NoNewline
        python ./main.py -d $i -c 3 -t $j -a 1 2 1 --accuracy >> $FilePath
    }
}

"`n`nKernel & Regularyzacja" >> $FilePath

"`nData set 1" >> $FilePath
For ($j = 2; $j -le 4; $j++)
{
    "Kernel $j" >> $FilePath

    $array = @("0.1", "0.2", "0.5", "1", "2", "5")
    foreach ($k in $array)
    {
        "$k`t" | Out-File $FilePath -Append -NoNewline
        python ./main.py -d 1 -c 3 -t 75 -a $k $j 1 --accuracy >> $FilePath
    }
}

For ($i = 2; $i -le 3; $i++)
{
    "`nData set $i" >> $FilePath
    For ($j = 1; $j -le 4; $j++)
    {
        "Kernel $j" >> $FilePath

        $array = @("0.1", "0.2", "0.5", "1", "2", "5")
        foreach ($k in $array)
        {
            "$k`t" | Out-File $FilePath -Append -NoNewline
            python ./main.py -d $i -c 3 -t 75 -a $k $j 1 --accuracy >> $FilePath
        }
    }
}

"`n`Gamma" >> $FilePath

For ($i = 2; $i -le 3; $i++)
{
    "`nData set $i" >> $FilePath
    For ($j = 2; $j -le 3; $j++)
    {
        "Kernel $j" >> $FilePath

        $array = @("0.1", "0.2", "0.5", "1", "2", "5")
        foreach ($k in $array)
        {
            "$k`t" | Out-File $FilePath -Append -NoNewline
            python ./main.py -d $i -c 3 -t 75 -a 1 $j 3 $k --accuracy >> $FilePath
        }
    }
}