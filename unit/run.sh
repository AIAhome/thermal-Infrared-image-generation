torchrun --nproc_per_node=8 unit.py \
--model vit \
--n_epochs 300 \
--decay_epoch 200 \
--batch_size 8 \
--lr 0.0001 \