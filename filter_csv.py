import sys

if __name__ == "__main__":
    num_entries = 101
    filename = sys.argv[1]
    count= 0
    with open(filename, 'r') as csvfile:
        count = sum(1 for line in csvfile)
    with open(filename, 'r') as csvfile:
        step = (count - 1) / (num_entries-1)
        sys.stdout.write(next(csvfile))
        for i,line in enumerate(csvfile):
            if i % step == 0:
                sys.stdout.write(line)
