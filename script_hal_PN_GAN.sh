for lr_power in 6
do
    for lr_base in 1
    do
        for m_support in $2
        do
            for n_support in $3
            do
                for n_aug in 20 40
                do
                    for n_query in 50
                    do
                        for n_ite_per_epoch in 500
                        do
                            python3 /home/cclin/few_shot_learning/sem_hal/train_hallucinator_PN.py \
                                --result_path /home/cclin/few_shot_learning/sem_hal/awa/results_$1 \
                                --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
                                --hallucinator_name HAL_PN_m${m_support}n${n_support}a${n_aug}q${n_query}_ite${n_ite_per_epoch}_GAN \
                                --learning_rate ${lr_base}e-${lr_power} \
                                --l2scale 1e-2 \
                                --m_support ${m_support} \
                                --n_support ${n_support} \
                                --n_aug ${n_aug} \
                                --n_query ${n_query} \
                                --num_epoch 500 \
                                --n_ite_per_epoch ${n_ite_per_epoch} \
                                --patience 10 \
                                --debug \
                                --z_dim 256 \
                                --fc_dim 256 \
                                --GAN
                        done
                    done
                done
            done
        done
    done
done
