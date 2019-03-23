import csv

reader_csv = list(csv.reader(open('all_questions_trimmed.csv', 'rt')))

q1_pair = []
for l in reader_csv:
    s = l[0].split('$')[0]
    a = l[1]
    q1_pair.append((s, a))

q2_pair = []
for l in reader_csv:
    s = l[0].split('$')
    if len(s) <= 1:
        continue
    st = s[1]
    a = l[2]
    q2_pair.append((st, a))

q3_pair = []
for l in reader_csv:
    s = l[0].split('$')
    if len(s) <= 2:
        continue
    st = s[2]
    a = l[3]
    q3_pair.append((st, a))

q4_pair = []
for l in reader_csv:
    s = l[0].split('$')
    if len(s) <= 3:
        continue
    st = s[3]
    a = l[4]
    q4_pair.append((st, a))

q5_pair = []
for l in reader_csv:
    s = l[0].split('$')
    if len(s) <= 4:
        continue
    st = s[4]
    a = l[5]
    q5_pair.append((st, a))

with open('q1_pair.csv', 'wt') as f:
    for p in q1_pair:
        f.write('{},{}\n'.format(''.join(p[0].split(',')), p[1]))

with open('q2_pair.csv', 'wt') as f:
    for p in q2_pair:
        f.write('{},{}\n'.format(''.join(p[0].split(',')), p[1]))

with open('q3_pair.csv', 'wt') as f:
    for p in q3_pair:
        f.write('{},{}\n'.format(''.join(p[0].split(',')), p[1]))

with open('q4_pair.csv', 'wt') as f:
    for p in q4_pair:
        f.write('{},{}\n'.format(''.join(p[0].split(',')), p[1]))

with open('q5_pair.csv', 'wt') as f:
    for p in q5_pair:
        f.write('{},{}\n'.format(''.join(p[0].split(',')), p[1]))
