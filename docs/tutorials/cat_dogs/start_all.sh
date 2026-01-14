#! /bin/bash
python ./start_cors_http.py &
label-studio-ml start ./dog_cat_backend &
docker run -it --user root -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest chown -R 1001:root /label-studio/data/; docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
