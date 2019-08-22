import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    prec_sum = 0.0
    count = 0

    with open(filename, 'r') as csvfile:
        for i,line in enumerate(csvfile):
            prec, recall = line.split(',')
            try:
                prec_float = float(prec)
                prec_sum += prec_float
                count += 1
            except:
                print('Could not parse to float:', prec)

    print('mAP:', str(prec_sum / float(count)))
