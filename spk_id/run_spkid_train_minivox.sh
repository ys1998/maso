python3 nnet.py --spk2idx minivoxceleb/utt2spk.npy --data_root minivoxceleb/train/ --train_guia minivoxceleb/tr_list.txt \
       --log_freq 50 --batch_size 64 --lr 0.001 --save_path spkid_out/ \
       --model mlp --opt adam --patience 5 --train --lrdec 0.5 \
       --hidden_size 2048 --epoch 100 --sched_mode plateau \
       --fe_cfg ../cfg/PASE.cfg  \
       --seed 2 \
       --num_workers 3
