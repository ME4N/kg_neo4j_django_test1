import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    center_word = get_PER_entity(tag_seq, char_seq)
    return center_word


# def get_PER_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     PER = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B':
#             per = char
#             if i + 1 == length:
#                 PER.append(per)
#         if tag == 'I':
#             per += char
#             # if i + 1 == length:
#             PER.append(per)
#         if tag not in ['I', 'B']:
#             continue
#     return PER

def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    org = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i + 1 == length:
                ORG.append(org)
        if tag == 'I':
            org += char
            if i + 1 == length:
                ORG.append(org)
        if tag not in ['I', 'B']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    print("center_word:", ORG)
    return ORG


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i + 1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i + 1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
