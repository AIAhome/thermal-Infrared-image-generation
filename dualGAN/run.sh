python3 dualgan.py \
 --n_epochs=50\
 --batch_size=4\
 --data_path="/root/autodl-tmp/FLIR_ADAS_v2"\
 --log_path="/root/thermal-Infrared-image-generation/dualGAN/log"\
 --output_path="/root/autodl-tmp/output/dualGAN"\
 --a_channels=1\
 --b_channels=1\
 --checkpoint_interval=400