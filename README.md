 <h1> Overview </h1>

This project aims to design and implement a robust data feature extraction and retrieval pipeline that supports multimodal data — including text, images, and metadata. The system focuses on efficient storage, retrieval, and visualization with high accuracy, low latency, and strong fault tolerance.
By integrating database optimization, hybrid search, and API-driven architecture, it serves as a scalable foundation for production-ready applications.

 In this project's development Foto Atlas' database is used. The following examples are from their website


## Feature Extraction

In order to find the similarity between the query image and the images from the database, feature extraction module is implemented. (OpenAI CLIP)

## SAM (Segment Anything)

Since the dataset doesn't contain any segmentation annotations, in order to focus more localy while searching (to increase the accuracy) SAM is used to generate the segments.

Segment Examples

| Raw Image | Segment1 | Segment2 | Segment3 | Segment 4 | Segment 5
| :--- | :--- | :---: | :--- | :---: |  :---: |
<img width="389" height="527" alt="sam_seg_ex" src="https://github.com/user-attachments/assets/ef732af1-f9e6-41c6-9929-443a4790a071" /> | <img width="508" height="531" alt="seg_part_1" src="https://github.com/user-attachments/assets/3478b9c4-7197-4bb2-b19d-c0a57933a939" /> |<img width="72" height="529" alt="Screenshot from 2025-12-28 19-17-15" src="https://github.com/user-attachments/assets/90092eea-5256-4b93-92ac-0d135eb9bdcf" /> | <img width="80" height="235" alt="Screenshot from 2025-12-28 19-17-24" src="https://github.com/user-attachments/assets/aa89fb6a-a0fc-4e01-8f46-e8406dd5d9af" /> | <img width="121" height="135" alt="Screenshot from 2025-12-28 19-17-32" src="https://github.com/user-attachments/assets/580682c9-760c-4edb-870e-3634b5afbb1b" /> | <img width="111" height="53" alt="Screenshot from 2025-12-28 19-17-39" src="https://github.com/user-attachments/assets/1963da66-24ef-45fa-89f0-fb87aa275713" />




## Hybrid Search

The first step for the similarity search is to generate segments via SAM for the images

Then, feature extraction is done by those segments along with the full image feature extraction

Addition to the vectors, also tags are generated from the segments

When an image is searched the module does the similarity search with the full image features which this created a candidate pool

Then with the segmented features and tags the list is narrowed down and ordered

The suggestions are sent and print to user (max 10)

Examples:

<img width="1053" height="449" alt="obelisk_2" src="https://github.com/user-attachments/assets/44aaf1f0-a9d0-4dac-8f20-663d77b814ec" />

<img width="1057" height="416" alt="ayasofya_2" src="https://github.com/user-attachments/assets/bd5fbdb8-73d6-407d-95e3-902c00754b40" />


