n=50
for i in $(seq 0 $n);
do
    echo $i
    python train.py $i
done
