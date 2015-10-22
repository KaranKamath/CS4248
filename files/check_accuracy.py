import sys

fAns = sys.argv[1]
fOutput = sys.argv[2]

with open(fAns, 'r') as fa, open(fOutput, 'r') as fo:
    count = 0
    count_correct = 0

    answers = [ line for line in fa.readlines() if line.strip() != '']
    output = [ line for line in fo.readlines() if line.strip() != '']

    for idx, ans in enumerate(answers):
        count = count + 1
        if ans == output[idx]:
            count_correct = count_correct + 1
    
    print 'Accuracy: ', str(1.0 * count_correct / count) 
