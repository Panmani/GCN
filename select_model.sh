n=200
for i in $(seq 0 $n);
do
    echo $i
    python train.py $i
done
