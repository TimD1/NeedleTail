set -e

tc=100
qpt=100
lens=(100 500 1000 2500 5000 7500 10000)

make clean
make gpu_nw
make batchgen

printf "\n%10s" "qlen\\tlen"
for tlen in ${lens[@]}; do
  printf "│%10s" $tlen
done
printf "│\n"

printf "──────────"
for idc in ${lens[@]}; do
  printf "┼──────────"
done
printf "┤\n"

for qlen in ${lens[@]}; do
  printf "%10s" $qlen
  for tlen in ${lens[@]}; do
    if [[ $qlen -gt $tlen ]]; then
      printf "│██████████"
    else
      ./batchgen.o --tc $tc --qpt $qpt --tl $tlen --ql $qlen --qsp 0 -o batch.txt
      ./gpu_nw.o > /dev/null
      printf "│%10s" $(head -1 GPU_results.txt | cut -d' ' -f2)
    fi
  done
  printf "│\n"
done

printf "\n"
