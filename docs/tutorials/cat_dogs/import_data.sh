#! /bin/bash

# import vars
export $(grep -v '^#' .env | xargs)

# loop through images
cd PetImages
for d in */*.jpg; do
	curl -X POST "$API_BASE_URL/api/projects/$PROJECT_ID/tasks" \
		-H "Authorization: Token $API_TOKEN" \
		-H "Content-Type: application/json" \
		-d "{\"data\": {\"image\": \"$IMAGE_BASE_URL/PetImages/$d\",
		  \"explanation\": \"$IMAGE_BASE_URL/PetImages/explanations/$d\"}}"
done
