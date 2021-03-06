for num_epoch in 20
do
    for n_shot in $2
    do
        for lr_power in $4
        do
            for lr_base in $3
            do
                for n_min in $7
                do
                    for m_support in $5
                    do
                        for n_support in $9
                        do
                            for n_aug in $6
                            do
                                for n_query in $8
                                do
                                    for n_ite_per_epoch in 500
                                    do
                                        for iteration in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
                                        do
                                            python3 /home/cclin/few_shot_learning/hallucination_by_analogy/train_fsl_infer_PN.py \
                                                --result_path /home/cclin/few_shot_learning/hallucination_by_analogy/_for_awa/results_final_$1 \
                                                --model_name FSL_PN_shot${n_shot}_hal${n_min}_lr${lr_base}e${lr_power}_ep${num_epoch}_l2reg1e2_${iteration} \
                                                --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
                                                --hallucinator_name HAL_PN_fc256_m${m_support}n${n_support}a${n_aug}q${n_query}_ite${n_ite_per_epoch}_GAN_256 \
                                                --data_path None \
                                                --sim_set_path None \
                                                --n_fine_classes 25 \
                                                --n_shot $((n_shot)) \
                                                --n_min ${n_min} \
                                                --n_top 5 \
                                                --bsize 100 \
                                                --learning_rate ${lr_base}e-${lr_power} \
                                                --learning_rate_aux 3e-6 \
                                                --l2scale 1e-2 \
                                                --num_epoch ${num_epoch} \
                                                --num_epoch_per_hal -1 \
                                                --n_iteration_aux 10000 \
                                                --z_dim 256 \
                                                --fc_dim 256 \
                                                --GAN
                                        done
                                        rm -rf /home/cclin/few_shot_learning/hallucination_by_analogy/_for_awa/results_final_$1/HAL_PN_fc256_m${m_support}n${n_support}a${n_aug}q${n_query}_ite${n_ite_per_epoch}_GAN_256/FSL*
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
