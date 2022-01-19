#/bin/sh
#compare the results under different various beta values

betas=(0.01 0.1 0.5 0.9 0.99)

for i in "${betas[@]}"
do
    python main.py $i
done