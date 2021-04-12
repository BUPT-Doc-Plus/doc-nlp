import online_score
import json
from time import time
from datetime import timedelta
from tqdm import tqdm


if __name__ == '__main__':
    online_score.init()
    results = []
    with open("test.json", "r", encoding="utf-8") as lines:
        for i, line in enumerate(lines):
            sample_start = time()
            result = online_score.run(line.strip())
            duration = timedelta(seconds=time() - sample_start)
            print("========================= Sample duration: {}".format(duration))
            results.append(result)

    with open("results.json", "w", encoding="utf8") as fout:
        json.dump(results, fout, indent=4)