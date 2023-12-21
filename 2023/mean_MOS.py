import os
import shutil
import glob

scores = glob.glob('./MOS/*.txt')


print(f'total {len(scores)} participants.')
score={}
for s in scores:
    f = open(s, 'r', encoding='euc-kr')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i<120:
            if line.split('/')[0] in score.keys():
                score[line.split('/')[0]].append(int(line.split(' ')[-1].replace('\n', '')))
            else:
                score[line.split('/')[0]]=[int(line.split(' ')[-1].replace('\n', ''))]
    f.close()
for name in score.keys():
    s = sum(score[name])/len(score[name])
    print(name, s)
