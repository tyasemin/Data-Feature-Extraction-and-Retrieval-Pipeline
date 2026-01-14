 <h1> Overview </h1>

This project aims to design and implement a robust data feature extraction and retrieval pipeline that supports multimodal data — including text, images, and metadata. The system focuses on efficient storage, retrieval, and visualization with high accuracy, low latency, and strong fault tolerance.
By integrating database optimization, hybrid search, and API-driven architecture, it serves as a scalable foundation for production-ready applications.

 In this project's development Foto Atlas' database is used. The following examples are from their website
https://fotoatlas.net/

## Dataset Analysis


<img width="1784" height="1182" alt="03_top_locations_metadata" src="https://github.com/user-attachments/assets/952b72bb-f319-41fc-843e-c3f0ee7e3c6d" />

<img width="2078" height="1483" alt="01_top_30_tags" src="https://github.com/user-attachments/assets/491b249c-b48f-47bb-9312-a89b089c8193" />

<img width="2082" height="1483" alt="08_tag_categories" src="https://github.com/user-attachments/assets/84d28b6a-ff08-4cc3-a25f-c99a3d5af139" />

<img width="1784" height="882" alt="12_historical_dates_histogram" src="https://github.com/user-attachments/assets/cfb7a9de-048c-40a0-bc32-af7e8342c345" />


## Feature Extraction

In order to find the similarity between the query image and the images from the database, feature extraction module is implemented. (OpenAI CLIP)

## SAM (Segment Anything)

Since the dataset doesn't contain any segmentation annotations, in order to focus more localy while searching (to increase the accuracy) SAM is used to generate the segments.

<img width="2442" height="1483" alt="20k002170_sam_inference" src="https://github.com/user-attachments/assets/d0e3cc3e-7161-4855-a9b7-1e7bd7ca3ea2" />

<img width="2438" height="1483" alt="20k003277_sam_inference" src="https://github.com/user-attachments/assets/6b9af29a-d0ea-4a33-be85-ad6e2b18a423" />

<img width="2985" height="1395" alt="gri_96_r_14_b112_pp9_007_r_sam_inference" src="https://github.com/user-attachments/assets/3e6e7a69-117e-4508-96a3-cc11526b7b80" />

Along with the segments, tag are generated via OpenAI CLIP from the segments

<img width="347" height="423" alt="Screenshot from 2026-01-13 23-43-08" src="https://github.com/user-attachments/assets/f8f49a33-1af7-4f09-a7ad-b7c5a0e867f6" />


## Hybrid Search

The first step for the similarity search is to generate segments via SAM for the images

Then, feature extraction is done by those segments along with the full image feature extraction

Addition to the vectors, also tags are generated from the segments

When an image is searched the module does the similarity search with the full image features which this creates a candidate pool

Then with the segmented features and tags the list is narrowed down and ordered

The suggestions are sent and print to user (max 10)

Examples:

<img width="1053" height="449" alt="obelisk_2" src="https://github.com/user-attachments/assets/44aaf1f0-a9d0-4dac-8f20-663d77b814ec" />

<img width="1057" height="416" alt="ayasofya_2" src="https://github.com/user-attachments/assets/bd5fbdb8-73d6-407d-95e3-902c00754b40" />

# Usage

## Run Manually

In the root directory of the repository run the following for the elasitcsearch and kibana

`docker compose up --build -d`

Then check the services if they are working, it should be accessible on the 0.0.0.0:5602 for the kibana

Build the following image for SAM+CLIP, in the SAM directory of the repository

`docker build -t <image_name>:0.0.1 -f Dockerfile .`

Change the docker image name in the `search_with_sam_segments.sh` file where the docker run is.

Example usage : `./search_with_sam_segments.sh ./selimiyecami.png hybrid 10` means hybrid search with 10 similar results

## Run as Microservice

In order to use the service the elasticsearch and kibana must be up and running.

Build the microservice docker image with the following:

`docker build -f Dockerfile.microservice -t <image_name> .`

Run as microservice with the :

```
docker run -d --name sam_search_api --network host \
  -v $(pwd):/workspace \
  -e ES_HOST=localhost -e ES_PORT=9201 \
  <image_name>
```

Change the docker image name and make sure elasticsearch is running on the port 9201, if its different port, change the port in the above command.

Quick healt check: `curl http://localhost:5000/health`

Send the request to service via:

```
curl -X POST http://localhost:5000/search/hybrid \
  -F "image=@dikilitaş_test.png" \
  -F "top_k=10" \
  -F "visualize=true"
```

Change the image name to the image you want to search
Maksimum limit is 10 for the visualization
Set the visualization option









