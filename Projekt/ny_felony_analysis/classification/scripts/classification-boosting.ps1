$FilePath = "classification_boosting.txt"

$label = @(3, 4, 5, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    py ./main.py -ln $label[$i] -bt --time >> $FilePath
}
$label = @(3, 4, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    py ./main.py -ln $label[$i] -dn 2 5 -bt --time >> $FilePath
}

$label = @(5, 8, 9, 10, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    py ./main.py -ln $label[$i] -dn 3 4 -bt --time >> $FilePath
}

$label = @(3, 4, 5, 11, 12, 13)
For ($i = 0; $i -lt $label.Length; $i++)
{
    py ./main.py -ln $label[$i] -dn 8 9 10 -bt --time >> $FilePath
}

$label = @(3, 4, 5, 8, 9, 10)
For ($i = 0; $i -lt $label.Length; $i++)
{
    py ./main.py -ln $label[$i] -dn 11 12 13 -bt --time >> $FilePath
}