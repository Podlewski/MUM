$FilePath = "dbscan.txt"

$algo = 4
$dataset = 3


$fixedEPS = 17
$fixedNumOfSamples = 5

"dbscan d$dataset a$algo eps:" >> $FilePath

For ($i = 1; $i -le 20; $i++)
{
    python ./main.py -d $dataset -a $algo -p $i $fixedNumOfSamples 1 -x 3 -y 4 >> $FilePath
}

"dbscan d$dataset a$algo min_samples:" >> $FilePath

For ($i = 1; $i -le 10; $i++)
{
    python ./main.py -d $dataset -a $algo -p $fixedEPS $i 1 -x 3 -y 4 >> $FilePath
}