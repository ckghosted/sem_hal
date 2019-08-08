for num_epoch in 20
do
    for n_shot in $1
    do
        for lr_power in 5
        do
            for lr_base in 1 2 3 4 5
            do
                for n_min in 10 20 40
                do
                    for m_support in 15
                    do
                        for n_support in 5 10
                        do
                            for n_aug in 20 40
                            do
                                for n_query in 50
                                do
                                    for n_ite_per_epoch in 500
                                    do
                                        for iteration in 0 1 2 3 4 5 6 7 8 9
                                        do
                                            python3 /home/cclin/few_shot_learning/sem_hal/train_fsl_infer_PN.py \
                                                --word_emb_path /data/put_data/cclin/datasets/awa/Animals_with_Attributes2 \
                                                --emb_dim 85 \
                                                --result_path /home/cclin/few_shot_learning/sem_hal/awa/results_cv \
                                                --model_name FSL_PN_shot${n_shot}_hal${n_min}_lr${lr_base}e${lr_power}_ep${num_epoch}_l2reg1e2_${iteration} \
                                                --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
                                                --hallucinator_name HAL_PN_m${m_support}n${n_support}a${n_aug}q${n_query}_ite${n_ite_per_epoch}_VAEGAN2 \
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
                                                --VAEGAN2
                                        done
                                        rm -rf /home/cclin/few_shot_learning/sem_hal/awa/results_cv/HAL_PN_m${m_support}n${n_support}a${n_aug}q${n_query}_ite${n_ite_per_epoch}_VAEGAN2/FSL*
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
