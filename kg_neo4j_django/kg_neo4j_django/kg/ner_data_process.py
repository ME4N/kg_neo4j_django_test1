import pandas as pd
import re


def ner_data_handle():
    ff = open('data/关系抽取222.txt','r',encoding='utf-8')
    ff_out = open("data/ner.csv", 'w', encoding='utf-8')
    ff_out_sentence = open("data/sentence.csv", 'w', encoding='utf-8')
    for line in ff.readlines():
        line = line.strip('\n').split(" ")[1]
        start = -1
        flag = 0
        word_ner = []
        word_list = []
        " </e1>"
        for i in range(len(line)):
            # print(line)
            if line[i] == '<' and line[i + 1] == 'e' and (line[i + 2] == '1' or line[i + 2] == '2') and line[i + 3] == '>':
                start = i + 4
                flag = 1
            if line[i] == '<' and line[i + 1] == '/' and line[i + 2] == 'e' and (line[i + 3] == '1' or line[i + 3] == '2')  and line[
                i + 4] == '>':
                flag = 0

            elif i >= start and flag == 1:
                if i == start:
                    word_ner.append("B")

                else:
                    word_ner.append("I")

                word_list.append(line[i])

            else:
                if line[i] not in '<e12/>':
                    word_list.append(line[i])
                    word_ner.append("O")
        print(word_list)
        print(word_ner)
        for i in range(len(word_list)):
            ff_out.write(word_list[i] + " " + word_ner[i] + '\n')
        ff_out.write("\n")
        ff_out_sentence.write(''.join(word_list) + '\n')

    # #-------------entity_data_show--------------
    # ff = open('data/entity_data_show.')
    # for line in ff.readlines():
    #     line = line.strip('\n')
    #     start = -1
    #     flag = 0
    #     word_ner = []
    #     word_list = []
    #     for i in range(len(line)):
    #         if line[i] == '<' and line[i + 1] == 'B' and line[i + 2] == '-' and line[i + 3] == 'D' and line[
    #             i + 4] == 'L' and line[i + 5] == '>':
    #             start = i + 6
    #             flag = 1
    #         if line[i] == '<' and line[i + 1] == 'E' and line[i + 2] == '-' and line[i + 3] == 'D' and line[
    #             i + 4] == 'L' and line[i + 5] == '>':
    #             flag = 0
    #         elif i >= start and flag == 1:
    #             if i == start:
    #                 word_ner.append("B")
    #             else:
    #                 word_ner.append("I")
    #             word_list.append(line[i])
    #         else:
    #             if line[i] not in '<B-DL>E':
    #                 word_list.append(line[i])
    #                 word_ner.append("O")
    #     print(word_list)
    #     print(word_ner)
    #     for i in range(len(word_list)):
    #         ff_out.write(word_list[i] + " " + word_ner[i] + '\n')
    #     ff_out.write("\n")
    #     ff_out_sentence.write(''.join(word_list) + '\n')


def ner_data_handle2():
    ff = open('data/entity_data_show.','r',encoding='utf-8')
    ff_out = open("data/ner.csv", 'w', encoding='utf-8')
    ff_out_sentence = open("data/sentence.csv", 'w', encoding='utf-8')
    for line in ff.readlines():
        line = line.strip('\n')
        start = -1
        flag = 0
        word_ner = []
        word_list = []
        for i in range(len(line)):
            if line[i] == '<' and line[i + 1] == 'B' and line[i + 2] == '-' and line[i + 3] == 'D' and line[
                i + 4] == 'L' and line[i + 5] == '>':
                start = i + 6
                flag = 1
            if line[i] == '<' and line[i + 1] == 'E' and line[i + 2] == '-' and line[i + 3] == 'D' and line[
                i + 4] == 'L' and line[i + 5] == '>':
                flag = 0
            elif i >= start and flag == 1:
                if i == start:
                    word_ner.append("B")
                else:
                    word_ner.append("I")
                word_list.append(line[i])
            else:
                if line[i] not in '<B-DL>E':
                    word_list.append(line[i])
                    word_ner.append("O")
        print(word_list)
        print(word_ner)
        for i in range(len(word_list)):
            ff_out.write(word_list[i] + " " + word_ner[i] + '\n')
        ff_out.write("\n")
        ff_out_sentence.write(''.join(word_list) + '\n')



def ner_train_test_split():
    ff = open('data/ner.csv','r',encoding='utf-8')
    ff_train = open('data/ner/train_data.txt', 'w',encoding='utf-8')
    ff_test = open('data/ner/test_data.txt', 'w',encoding='utf-8')
    ff_dev=open('data/ner/dev_data.txt','w',encoding='utf-8')
    num = len(ff.readlines())
    for i, line in enumerate(ff.readlines()):
        if i < int(num / 10) * 9:
            ff_train.write(line)
        elif int(num/10)*9.5 > i >= int(num/10) * 9:
            ff_test.write(line)
        else:
            ff_dev.write(line)


if __name__ == '__main__':
    ner_data_handle()
    ner_train_test_split()
