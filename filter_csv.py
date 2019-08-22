import csv
import sys

if __name__ == "__main__":
    num_entries = 101
    filename = sys.argv[1]
    count= 0
    with open(filename, 'r') as csvfile:
        count = sum(1 for row in csvfile)
    with open(filename, 'r') as csvfile:
        step = (count - 1) / (num_entries-1)
        r = csv.reader(csvfile)
        print(next(r))
        for i,row in enumerate(r):
            if i % step == 0:
                print(row)
