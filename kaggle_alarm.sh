i1=1
i2=1
while [ "$(i1+i2)" -ne 0 ]
do
  out1=$(kaggle kernels status eit-dual-vs-basic-unreduced-sp500-weekly)
  out2=$(kaggle kernels status eit-dual-vs-basic-unreduced-sp500-daily)
  if [[ "$out1" == *"complete"* ]] && [[ "$i1" -ne 0 ]]; then
    echo $'\a'
    afplay /System/Library/Sounds/Funk.aiff
    say Weekly Done
    echo $out
    i1=0
  fi
  if [[ "$out2" == *"complete"* ]] && [[ "$i2" -ne 0 ]]; then
    echo $'\a'
    afplay /System/Library/Sounds/Funk.aiff
    say Daily Done
    echo $out
    i2=0
  fi
  sleep 10
done