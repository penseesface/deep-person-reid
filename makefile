train:
	python scripts/main.py \
		--config-file configs/im_osnet_x0_25_pro.yaml \
		--transforms random_flip color_jitter \
		--root ${HOME}/datasets/

test:
	python scripts/main.py \
		--config-file configs/im_osnet_x0_25_pro.yaml \
		--transforms random_flip color_jitter \
		--root ${HOME}/datasets/ \
		model.load_weights log/osnet_x0_25_pro/model/model.pth.tar-180 \
		test.evaluate True
