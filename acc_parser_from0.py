import sys, re
import numpy as np

with open(sys.argv[1], 'r') as fhand:
    n_cluster = ''
    n_shot = ''
    exp_name = ''
    exp_name_note = ''
    hal_type = ''
    acc_list = []
    acc_novel_list = []
    acc_base_list = []
    is_slim = False
    lr_base = '?'
    lr_power = '?'
    lr = lr_base+'e-'+lr_power
    n_min = 0
    
    print('lr,nc,shot,k,exp_name,exp_name_note,hal_type,novel,novel_sd,base,base_sd,all,all_sd')
    
    for line in fhand:
        line = line.strip()
        #print(line)
        if re.search('^WARNING', line):
            if re.search('l2reg1e2_0', line):
                if not len(acc_list) == 0:
                    # compute mean accuracy
                    #print(acc_list)
                    if is_slim:
                        exp_name = exp_name + '_S'
                    print('%s,%s,%s,%d,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
                        (lr, n_cluster, n_shot, n_min, exp_name, exp_name_note, hal_type, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_base_list)*100, np.std(acc_base_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))
                    acc_list = []
                    acc_novel_list = []
                    acc_base_list = []
                lr_base = re.search('_lr([0-9]+)e[0-9]+_ep20', line).group(1)
                lr_power = re.search('_lr[0-9]+e([0-9]+)_ep20', line).group(1)
                lr = lr_base+'e-'+lr_power
            ## Extract nc
            if re.search('nc([0-9]+)', line):
                n_cluster = re.search('nc([0-9]+)', line).group(1)
            else:
                n_cluster = 0
            ## Extract shot
            if re.search('FSL_shot([0-9]+)', line):
                n_shot = re.search('FSL_shot([0-9]+)', line).group(1)
                is_slim = False
                is_PN = False
            elif re.search('FSL_SLIM_shot([0-9]+)', line):
                n_shot = re.search('FSL_SLIM_shot([0-9]+)', line).group(1)
                is_slim = True
                is_PN = False
            elif re.search('FSL_PN_GAN_shot([0-9]+)', line):
                n_shot = re.search('FSL_PN_GAN_shot([0-9]+)', line).group(1)
                is_PN = True
                is_slim = False
            elif re.search('FSL_PN_shot([0-9]+)', line):
                n_shot = re.search('FSL_PN_shot([0-9]+)', line).group(1)
                is_PN = True
                is_slim = False
            ## Extract n_min
            if re.search('nohal', line):
                exp_name = 'Baseline'
                exp_name_note = 'none'
                n_min = int(n_shot)
            elif re.search('basic_hal', line):
                if re.search('FSL_PN_GAN_shot[0-9]+_hal', line):
                    n_min = int(re.search('FSL_PN_GAN_shot[0-9]+_hal([0-9]+)', line).group(1))
                    PN_GAN_setting = re.search('HAL_PN_GAN_(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                    exp_name = 'basic_PN_GAN'
                    exp_name_note = PN_GAN_setting
                elif re.search('FSL_PN_shot[0-9]+_hal', line):
                    n_min = int(re.search('FSL_PN_shot[0-9]+_hal([0-9]+)', line).group(1))
                    PN_setting = re.search('HAL_PN_.*(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                    exp_name = 'basic_PN'
                    exp_name_note = PN_setting
                else:
                    n_min = int(re.search('basic_hal([0-9]+)', line).group(1))
                    exp_name = 'basic'
                    exp_name_note = 'none'
            elif re.search('coarse_hal', line):
                if re.search('FSL_PN_GAN_shot[0-9]+_coarse_hal', line):
                    n_min = int(re.search('FSL_PN_GAN_shot[0-9]+_coarse_hal([0-9]+)', line).group(1))
                    PN_GAN_setting = re.search('HAL_PN_GAN_(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                    exp_name = 'coarse_PN_GAN'
                    exp_name_note = PN_GAN_setting
                elif re.search('FSL_PN_shot[0-9]+_coarse_hal', line):
                    n_min = int(re.search('FSL_PN_shot[0-9]+_coarse_hal([0-9]+)', line).group(1))
                    PN_setting = re.search('HAL_PN_.*(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                    exp_name = 'coarse_PN'
                    exp_name_note = PN_setting
                else:
                    n_min = int(re.search('coarse_hal([0-9]+)', line).group(1))
                    exp_name = 'coarse'
                    exp_name_note = 'none'
            elif re.search('similar[0-9]+_[0-9]+to[0-9]+_hal', line):
                n_min = int(re.search('similar[0-9]+_[0-9]+to[0-9]+_hal([0-9]+)', line).group(1))
                similar_setting = re.search('similar([0-9]+_[0-9]+to[0-9]+)_hal([0-9]+)', line).group(1)
                exp_name = 'similar'
                exp_name_note = similar_setting
            elif re.search('FSL_PN_GAN_shot[0-9]+_hal', line):
                n_min = int(re.search('FSL_PN_GAN_shot[0-9]+_hal([0-9]+)', line).group(1))
                PN_GAN_setting = re.search('HAL_PN_GAN_(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                exp_name = 'PN_GAN'
                exp_name_note = PN_GAN_setting
            elif re.search('FSL_PN_shot[0-9]+_hal', line):
                n_min = int(re.search('FSL_PN_shot[0-9]+_hal([0-9]+)', line).group(1))
                if re.search('HAL_PN_.*m[0-9]+n[0-9]+a[0-9]+q[0-9]+_ite[0-9]+', line):
                    PN_setting = re.search('HAL_PN_.*(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                elif re.search('HAL_PN_T2_m[0-9]+n[0-9]+a[0-9]+q[0-9]+_ite[0-9]+', line):
                    PN_setting = re.search('HAL_PN_T2_(m[0-9]+n[0-9]+a[0-9]+q[0-9]+)_ite[0-9]+', line).group(1)
                exp_name = 'PN'
                exp_name_note = PN_setting
            if re.search('HAL_PN_T2_m[0-9]+n[0-9]+a[0-9]+q[0-9]+_ite[0-9]+', line):
                lambda_code = 0
                hal_type = 'PN_T2'
            elif re.search('HAL_PN_m[0-9]+n[0-9]+a[0-9]+q[0-9]+_ite[0-9]+_T2', line):
                lambda_code = 0
                hal_type = 'PN_T2'
            elif re.search('_T2', line):
                lambda_code = re.search('and([0-9]+)_T2', line).group(1)
                hal_type = 'T2(%s)' % lambda_code
            elif re.search('_T3', line):
                lambda_code = re.search('and([0-9]+)_T3', line).group(1)
                hal_type = 'T3(%s)' % lambda_code
            elif re.search('_T4', line):
                lambda_code = re.search('and([0-9]+)_T4', line).group(1)
                hal_type = 'T4(%s)' % lambda_code
            elif re.search('_T', line):
                lambda_code = re.search('and([0-9]+)_T', line).group(1)
                hal_type = 'T1(%s)' % lambda_code
            else:
                hal_type = 'T0'
        else:
            acc_list.append(float(re.search('top-5 test accuracy: (0\.[0-9]+)', line).group(1)))
            acc_novel_list.append(float(re.search('novel top-5 test accuracy: (0\.[0-9]+)', line).group(1)))
            acc_base_list.append(float(re.search('base top-5 test accuracy: (0\.[0-9]+)', line).group(1)))
    #print(acc_list)
    if is_slim:
        exp_name = exp_name + '_S'
    print('%s,%s,%s,%d,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
        (lr, n_cluster, n_shot, n_min, exp_name, exp_name_note, hal_type, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_base_list)*100, np.std(acc_base_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))


# fhand = open(fileName)
# lst = list()
# for line in fhand:
#     line = line.strip()
#     t = line.split(' ')
#     lst.append(float(t[colIdx]))

# fout = open('ans1.txt', 'w')
# lst.sort()
# for ele in lst[:-1]:
#     fout.write('%s,' % ele)
#     #print '%s,' % ele,; sys.stdout.softspace = False;
# fout.write(str(lst[-1]))
# fout.close()
