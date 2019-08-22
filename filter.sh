for s in normal 90_ccw 90_cw 180; do
  for c in pr roc; do
    for t in capsnet baseline; do
      python filter_csv.py results/$t/${s}_${c}.csv > results/$t/${s}_${c}_filtered.csv
    done
  done
done
