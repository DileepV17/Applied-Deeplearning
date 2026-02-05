from collections import Counter
import matplotlib.pyplot as plt

txt_path = "data/real/real_train.txt"  # path to your file

labels = []
with open(txt_path, "r") as f:
    for line in f:
        label = int(line.strip().split()[-1])
        labels.append(label)

dist = Counter(labels)
print("Label distribution:", dist)

plt.bar(dist.keys(), dist.values())
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Label Distribution")
plt.savefig("distribution_real_traintxt.png")