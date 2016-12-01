

files = ['feat_1_5000.csv', 'feat_5001_10000.csv','feat_10001_15000.csv', 'feat_15001_20000.csv',
         'feat_20001_25000.csv', 'feat_25001_30000.csv', 'feat_30001_35000.csv', 'feat_35001_40000.csv',
         'feat_40001_45000.csv', 'feat_45001_50000.csv', 'feat_50001_55000.csv', 'feat_55001_58209.csv']



outFile = open('feat_total.csv', 'w')

i = 0

for file in files:
    inFile = open(file, 'r')
    for line in inFile:
        outFile.write(line)
    inFile.close()
    i += 1
    print(str(i))

outFile.close()





