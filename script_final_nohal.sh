for num_epoch in 20
do
    for n_clusters in 50
    do
        for n_shot in $1
        do
            for lr_power in $4
            do
                for lr_base in $3
                do
                    for iteration in 0 1 2 3 4 5 6 7 8 9
                    do
                        python3 /home/cclin/few_shot_learning/sem_hal/train_fsl_infer.py \
                            --result_path /home/cclin/few_shot_learning/sem_hal/awa/results_final \
                            --model_name FSL_shot${n_shot}_nohal_lr${lr_base}e${lr_power}_ep${num_epoch}_l2reg1e2_${iteration} \
                            --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
                            --hallucinator_name baseline \
                            --data_path None \
                            --sim_set_path None \
                            --n_fine_classes 25 \
                            --n_shot $((n_shot)) \
                            --n_min ${n_shot} \
                            --n_top 5 \
                            --bsize 100 \
                            --learning_rate ${lr_base}e-${lr_power} \
                            --l2scale 1e-2 \
                            --fc_dim 256 \
                            --num_epoch ${num_epoch}
                    done
                done
            done
        done
    done
done

