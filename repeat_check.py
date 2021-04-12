with open("data/train.src", "r", encoding="utf-8") as f:
    train_src = f.readlines()

with open("data/test.src", "r", encoding="utf-8") as f:
    test_src = f.readlines()

with open("data/train.tgt", "r", encoding="utf-8") as f:
    train_tgt = f.readlines()

with open("data/test.tgt", "r", encoding="utf-8") as f:
    test_tgt = f.readlines()

train = set(list(zip(train_src, train_tgt)))
test = set(list(zip(test_src, test_tgt)))

print(len(set(train_src).intersection(set(test_src))))
print(len(set(train_tgt).intersection(set(test_tgt))))
print(len(train.intersection(test)))
