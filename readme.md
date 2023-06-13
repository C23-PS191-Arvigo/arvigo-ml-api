# Arvigo Machine Learning API
The Arvigo Machine Learning API is a powerful tool that provides several models for various tasks. These models include Human Face Detection, Face Shape Detection, Personality Analysis, Search Engine, and Product Recommendation.

## Table of Models

The API provides the following models:

- Human Face Detection: This model is capable of detecting human faces in images.
- Face Shape Detection: This model can identify and classify different face shapes in images.
- Personality Analysis: This model analyzes text input and provides insights into the personality traits of the author.
- Search Engine: This model allows users to perform searches on a specific dataset or collection of documents.
- Product Recommendation: This model suggests relevant products based on user preferences and historical data.

## Development

### Requirements
- Python 3.x
- Flask
- Other required dependencies (specified in the requirements.txt file)

### How to run with live reload

To enable live reload while running the API server, use the following command:

```
flask --app server --debug run
```
This command will start the API server with live reload functionality, allowing you to make changes to the code and see the updates immediately without manually restarting the server.

## Authentication

Access to the API endpoints requires authentication using an API key. You need to include the API key in the X-API-KEY header of each request to authenticate successfully. Make sure to obtain and use a valid API key for accessing the API endpoints.


## Endpoints

### Human Face Detection

- **URL:** `127.0.0.1:5000/is_human`
- **Method:** POST
- **Body:** `image="base64_string_here"`

> **IMPORTANT**: This endpoint is not used (legacy) because its function was combined to the Face Shape detection endpoint.

In HTTPie, you can use the following command:

```
http POST "127.0.0.1:5000/is_human" image="base64_string_here" "X-API-KEY: XXX"
```

It will return a single boolean value wrapped in JSON:

```json
{
    "result": true
}
```

### Face Shape Detection

- **URL:** `127.0.0.1:5000/face_shape`
- **Method:** POST
- **Body:** `image="base64_string_here"`

In HTTPie, you can use the following command:

```
http POST "127.0.0.1:5000/face_shape" image="base64_string_here" "X-API-KEY: XXX"
```

It will return the shape of the detected face along with its probability, for example:

```json
{
    "is_human": true,
    "probability": 0.48920536041259766,
    "shape": "square"
}
```

### Personality Analysis

- **URL:** `127.0.0.1:5000/detect_personality`
- **Method:** POST
- **Body:** JSON

In HTTPie, you can use the following command:

```bash
http POST http://127.0.0.1:5000/detect_personality \
    Content-Type:application/json \
    EXT1:=4 EXT2:=4 EXT3:=4 EXT4:=4 EXT5:=4 \
    EXT6:=4 EXT7:=4 EXT8:=4 EXT9:=4 EXT10:=4 \
    EST1:=4 EST2:=4 EST3:=4 EST4:=4 EST5:=4 \
    EST6:=4 EST7:=4 EST8:=4 EST9:=4 EST10:=4 \
    AGR1:=4 AGR2:=4 AGR3:=4 AGR4:=4 AGR5:=4 \
    AGR6:=4 AGR7:=4 AGR8:=4 AGR9:=4 AGR10:=4 \
    CSN1:=4 CSN2:=4 CSN3:=4 CSN4:=4 CSN5:=4 \
    CSN6:=4 CSN7:=4 CSN8:=4 CSN9:=4 CSN10:=4 \
    OPN1:=4 OPN2:=4 OPN3:=4 OPN4:=4 OPN5:=4 \
    OPN6:=4 OPN7:=4 OPN8:=4 OPN9:=4 OPN10:=4 \
    "X-API-KEY: XXX"
```

It _could_ return the predicted personality traits, for example:

```json
{
    "percentage_of_agreeable": 19.64,
    "percentage_of_conscientious": 19.64,
    "percentage_of_extraversion": 17.86,
    "percentage_of_neurotic": 23.21,
    "percentage_of_openess": 19.64
}
```

### Search Engine

- **URL:** `127.0.0.1:5000/product_search`
- **Method:** POST
- **Body:** JSON

In Curl, you can use the following command:

```bash
curl -X GET "http://127.0.0.0.1:5000/product_search" -H "Content-Type: application/json" -d '{
  "query": "laptop",
  "num_results": 5
}'
```

It will return a list of product IDs that match the given query, for example:

```json
[
    {
        "brand": "oakley",
        "category": "glasses kacamata",
        "description": "halo bro",
        "id": "2",
        "merchants": "arvigo tech",
        "name": "kacamata kotak",
        "similarity": 0.15706168831168826,
        "tags": "aviator,cat eye,rimless"
    },
    {
        "brand": "ray-ban",
        "category": "makeup",
        "description": "coba lagi",
        "id": "11",
        "merchants": "",
        "name": "arvigo project",
        "similarity": 0.15674603174603174,
        "tags": "aviator,cat eye"
    },
    {
        "brand": "bottega veneta",
        "category": "makeup",
        "description": "halo bro",
        "id": "4",
        "merchants": "arvigo tech,yusuf wibisono store",
        "name": "mascara",
        "similarity": 0.1450892857142857,
        "tags": "foundation"
    },
    {
        "brand": "bottega veneta",
        "category": "makeup",
        "description": "halo bro",
        "id": "7",
        "merchants": "",
        "name": "mascara",
        "similarity": 0.1294642857142857,
        "tags": "foundation"
    },
    {
        "brand": "chanel",
        "category": "makeup",
        "description": "desk",
        "id": "8",
        "merchants": "",
        "name": "coba brand",
        "similarity": 0.11249999999999998,
        "tags": "aviator,cat eye"
    },
    {
        "brand": "chanel",
        "category": "makeup",
        "description": "desk",
        "id": "9",
        "merchants": "",
        "name": "coba brand",
        "similarity": 0.11249999999999998,
        "tags": "aviator,cat eye"
    },
    {
        "brand": "chanel",
        "category": "makeup",
        "description": "desk",
        "id": "10",
        "merchants": "",
        "name": "coba brand",
        "similarity": 0.11249999999999998,
        "tags": "aviator,cat eye"
    },
    {
        "brand": "wardah",
        "category": "makeup",
        "description": "sdfsa",
        "id": "13",
        "merchants": "",
        "name": "coba brand",
        "similarity": 0.11249999999999998,
        "tags": "eyeliner"
    }
]
```

### Product Recommendation

- **URL:** `127.0.0.1:5000/recommend_product`
- **Method:** POST
- **Body:** JSON

In HTTPie, you can use the following command:

```bash
http POST http://127.0.0.1:5000/recommend_product \
    Content-Type:application/json \
    product_id=12345 "X-API-KEY: XXX"
```

It will return a recommended product based on the given product ID, for example:

```json
{
    "10": [
        "8",
        "9",
        "11",
        "13",
        "7",
        "4",
        "2"
    ],
    "11": [
        "8",
        "9",
        "10",
        "7",
        "13",
        "4",
        "2"
    ],
    "13": [
        "8",
        "9",
        "10",
        "7",
        "11",
        "4",
        "2"
    ],
    "2": [
        "8",
        "9",
        "10",
        "11",
        "4",
        "7",
        "13"
    ],
    "4": [
        "7",
        "8",
        "9",
        "10",
        "11",
        "13",
        "2"
    ],
    "7": [
        "4",
        "8",
        "9",
        "10",
        "11",
        "13",
        "2"
    ],
    "8": [
        "9",
        "10",
        "11",
        "13",
        "7",
        "4",
        "2"
    ],
    "9": [
        "8",
        "10",
        "11",
        "13",
        "7",
        "4",
        "2"
    ]
}
```

## Error Handling

If an error occurs, the API will return an error message with an appropriate status code.

Example error response:

```json
{
    "message": "Request failed with status code 401",
}
```

