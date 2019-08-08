import sys, re

import pandas as pd
df = pd.read_csv(sys.argv[1])
# print(df)

# print(df.head(10))

n_shot = '0'+str(df.shot[0]) if df.shot[0] < 10 else str(df.shot[0])
hyper = {}
best_novel = {}

df_baseline = df[(df.exp_name == 'Baseline')]
if not df_baseline.empty:
    # print(df_baseline)
    idx = df_baseline.novel.idxmax()
    lr = str(df_baseline.lr[idx])
    ## 3.000000000000001e-07
    lr_base = str(int(float(re.search('([0-9]+)\.*[0-9]*e-[0-9]+', lr).group(1))))
    lr_power = str(int(re.search('[0-9]+e-([0-9]+)', lr).group(1)))
    hyper['baseline'] = {'lr_base': lr_base, 'lr_power': lr_power}
    best_novel['baseline'] = df_baseline.novel.max()
    # print(df_baseline.loc[[idx]])
else:
    idx = df.novel.idxmax()
    lr = str(df.lr[idx])
    ## 3.000000000000001e-07
    lr_base = str(int(float(re.search('([0-9]+)\.*[0-9]*e-[0-9]+', lr).group(1))))
    lr_power = str(int(re.search('[0-9]+e-([0-9]+)', lr).group(1)))
    n_min = str(df.k[idx])
    exp_name_note = str(df.exp_name_note[idx])
    m_support = str(int(re.search('m([0-9]+)n[0-9]+a[0-9]+q[0-9]+', exp_name_note).group(1)))
    n_support = str(int(re.search('m[0-9]+n([0-9]+)a[0-9]+q[0-9]+', exp_name_note).group(1)))
    n_aug = str(int(re.search('m[0-9]+n[0-9]+a([0-9]+)q[0-9]+', exp_name_note).group(1)))
    n_query = str(int(re.search('m[0-9]+n[0-9]+a[0-9]+q([0-9]+)', exp_name_note).group(1)))
    hyper['PN_GAN'] = {'lr_base': lr_base, 'lr_power': lr_power, 'n_min': n_min, 'm_support': m_support, 'n_support': n_support, 'n_aug': n_aug, 'n_query': n_query}
    best_novel['PN_GAN'] = df.novel.max()
    # print(df_baseline.loc[[idx]])


for key in sorted(hyper.keys()):
    print('# [%s] best novel top-5 accuracy is %.2f' % (key, best_novel[key]))

print('# [NOTE] all hyper-parameters are in the order: lr_base, lr_power, m_support, n_aug, n_min, n_query, n_support')
for key in sorted(hyper.keys()):
    print('sh script_final_%s.sh %s' % (sys.argv[2], n_shot) , end = ' ')
    for par in sorted(hyper[key].keys()):
        print(hyper[key][par], end = ' ')
    print()