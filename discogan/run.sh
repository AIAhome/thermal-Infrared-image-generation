CUDA_VISIBLE_DEVICES=5 nohup python3 -u discogan.py \
--n_epochs=101  --batch_size=8 --checkpoint_interval=10 --sample_interval=400 --lambda_perceptual=0.05 --lambda_adv=2 --lambda_cycle=0.1 --lambda_content=0.1 --b_channels=3 \
> discogan_vgg_3.log &