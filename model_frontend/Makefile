
# curl --silent http://<...>.westeurope.azurecontainer.io/swagger.json | jq "." - > ./swagger.json


build:
	docker build -t flask_model_front_end .

.PHONY: run
run:
	docker run \
	  --rm \
	  -d \
	  -p 5000:5000 \
	  --name flask_model_front_end \
	  -e FLASK_SECRET_KEY="$$(uuidgen)" \
	  -e SCORING_ENDPOINT \
	  flask_model_front_end

#	  --rm \
