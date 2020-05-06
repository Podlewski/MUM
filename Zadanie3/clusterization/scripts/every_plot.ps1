$dataset = 2
$algorithm = 2
$columns = 11

For ($i = 1; $i -le $columns; $i++)
{
    For ($j = 1; $j -le $columns; $j++)
    {
        if ($j -ne $i)
        {
            python ./main.py -d $dataset -a $algorithm --fixed -x $i -y $j
        }
    }
}
