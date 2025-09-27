
IMAGE=bankruptcy-predictor:latest
build:
	docker build -t $(IMAGE) .
train: build
	docker run --rm -v $(PWD)/data:/data:ro -v $(PWD)/models:/models $(IMAGE) train --csv /data/american_bankruptcy.csv --outdir /models
serve: build
	docker run --rm -p 7860:7860 -v $(PWD)/models:/models $(IMAGE) serve --model /models/catboost_bankruptcy.cbm --meta /models/metadata.json
evaluate: build
	docker run --rm -v $(PWD)/data:/data -v $(PWD):/out $(IMAGE) evaluate --input /data/$(INPUT) --out /out/$(OUT) $(if $(SHEET),--sheet $(SHEET),) $(if $(THRESH),--threshold $(THRESH),)
