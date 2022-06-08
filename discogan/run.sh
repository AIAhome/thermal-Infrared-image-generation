CUDA_VISIBLE_DEVICES=3 nohup python3 -u discogan.py \
--n_epochs=151  --batch_size=16 --checkpoint_interval=10 --sample_interval=800 \
> discogan_rgb_id.log &