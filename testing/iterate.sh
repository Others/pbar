echo "" > output/$1output.log
for i in 1 2 3 4 5 6 7 8 9 10
do
	echo $i >> output/$1output.log
	python3 ../pbar.py sh $1.sh | grep -E -- 'model training accuracy|testing accuracy' >> output/$1output.log
done
