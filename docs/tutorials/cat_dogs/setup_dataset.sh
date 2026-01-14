curl -L -o ./dog-and-cat-classification-dataset.zip https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset
unzip dog-and-cat-classification-dataset.zip
rm dog-and-cat-classification-dataset.zip
rm PetImages/Dog/???*.jpg
rm PetImages/Cat/???*.jpg
cp -r PetImages explanations
mv explanations/ PetImages/
