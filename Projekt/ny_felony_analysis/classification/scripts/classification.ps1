$FilePath = "classification_part_1.txt"

$label = @(3, 4, 5, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    python ./main.py -ln $label[$i] -s --time >> $FilePath
    python ./main.py -ln $label[$i] -s -p --time >> $FilePath
    python ./main.py -ln $label[$i] -s -i --time >> $FilePath
}

$label = @(3, 4, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    python ./main.py -ln $label[$i] -dn 2 5  -s --time >> $FilePath
}

$label = @(5, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    python ./main.py -ln $label[$i] -dn 3 4 -s --time >> $FilePath
}

$label = @(3, 4, 5, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    python ./main.py -ln $label[$i] -dn 8 9 10 -s --time >> $FilePath
}

$label = @(3, 4, 5, 8, 9, 10)
For ($i = 0; $i -lt $label.Length; $i++)
{
    python ./main.py -ln $label[$i] -dn 11 12 13 -s --time >> $FilePath
}