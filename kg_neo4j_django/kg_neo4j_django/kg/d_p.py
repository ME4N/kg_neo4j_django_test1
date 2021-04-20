
ff = open('data/ner.csv', 'r', encoding='utf-8')
ff_train = open('data/ner/train_data.txt', 'w', encoding='utf-8')
ff_test = open('data/ner/test_data.txt', 'w', encoding='utf-8')
ff_dev = open('data/ner/dev_data.txt', 'w', encoding='utf-8')
num = len(ff.readlines())
print(num)
for i, line in enumerate(ff.readlines()):
    print(line)
    if i < int(num / 10) * 9:
        ff_train.write(line)
        print(line)
    elif int(num / 10) * 9.5 > i >= int(num / 10) * 9:
        ff_test.write(line)
    else:
        ff_dev.write(line)

ff.close()
ff_train.close()
ff_test.close()
ff_dev.close()
