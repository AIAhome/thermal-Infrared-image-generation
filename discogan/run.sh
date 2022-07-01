CUDA_VISIBLE_DEVICES=5 nohup python3 -u discogan.py \
--n_epochs=141  --batch_size=8 --checkpoint_interval=5 --sample_interval=400 --lambda_perceptual=0.05 --b_channels=3 \
> final.log &