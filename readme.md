# Arvigo Machine Learning API

## Table of Models

- Model for detecting human and non-human faces
- Model for detecting face shapes
- Model for analyzing personalities

## Development

### How to run

```
flask --app server run
```

### How to run with live reload

```
flask --app server --debug run
```

## Endpoints

#### Human Face Detection

```
URL: 127.0.0.1:5000/is_human
Method: POST
Body: image="base64stringhere"
```

In HTTPie, we can write this as: `http POST "127.0.0.1:5000/is_human" image="base64stringhere" "X-API-KEY: XXX`

It will return a boolean value in String format.
#### Face Shape Detection

```
URL: 127.0.0.1:5000/face_shape
Method: POST
Body: image="base64stringhere"
```

In HTTPie, we can write this as: `http POST "127.0.0.1:5000/face_shape" image="base64stringhere" "X-API-KEY: XXX`

It will return:

```
{
    "probability": 0.48920559883117676,
    "shape": "square"
}
```

#### Personality Analysis

```
URL: 127.0.0.1:5000/detect_personality
Method: POST
Body: JSON
```

In HTTPie, we can write this as: 

```
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

It _could_ return:

```
{
    "predicted_personality": [
        "Extraversion"
    ]
}
```
### Search Engine

```
URL: 127.0.0.1:5000/product_search
Method: POST
Body: JSON
```

In Curl, we can write this as: 

```
curl -X GET "http://127.0.0.1:5000/product_search?query=Emporio" -H "X-API-KEY: XXX"
```

It will return:

```
[
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 39,
        "combined": "Kacamata 8 This is the description of Product 8 Glasses Emporio heart, oval, square Optik Merah Putih, Optik Susi, Optik Sukarno, Optik tik",
        "description": "This is the description of Product 8",
        "id": 8,
        "merchants": "Optik Merah Putih, Optik Susi, Optik Sukarno, Optik tik",
        "name": "Kacamata 8",
        "similarity": 0.0475012846865365,
        "tags": "heart, oval, square"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 62,
        "combined": "Kacamata 15 This is the description of Product 15 Glasses Emporio oval, circle, oblong Optik Merah Putih, Optik tik, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 15",
        "id": 15,
        "merchants": "Optik Merah Putih, Optik tik, Optik Susi, Optik Sukarno",
        "name": "Kacamata 15",
        "similarity": 0.04657771188534719,
        "tags": "oval, circle, oblong"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 67,
        "combined": "Kacamata 18 This is the description of Product 18 Glasses Emporio heart, square, oblong Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 18",
        "id": 18,
        "merchants": "Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "name": "Kacamata 18",
        "similarity": 0.04621465491030709,
        "tags": "heart, square, oblong"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 18,
        "combined": "Kacamata 1 This is the description of Product 1 Glasses Emporio square, circle, triangle Optik Sukarno, Optik Susi, Optik tik, Optik Merah Putih",
        "description": "This is the description of Product 1",
        "id": 1,
        "merchants": "Optik Sukarno, Optik Susi, Optik tik, Optik Merah Putih",
        "name": "Kacamata 1",
        "similarity": 0.04567307692307695,
        "tags": "square, circle, triangle"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 5,
        "combined": "Kacamata 11 This is the description of Product 11 Glasses Oakley heart, triangle, square Optik Merah Putih, Optik Sukarno, Optik Susi, Optik tik",
        "description": "This is the description of Product 11",
        "id": 11,
        "merchants": "Optik Merah Putih, Optik Sukarno, Optik Susi, Optik tik",
        "name": "Kacamata 11",
        "similarity": 0.02430555555555558,
        "tags": "heart, triangle, square"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 6,
        "combined": "Kacamata 17 This is the description of Product 17 Glasses Oakley oblong, square, triangle Optik Merah Putih, Optik Sukarno, Optik tik, Optik Susi",
        "description": "This is the description of Product 17",
        "id": 17,
        "merchants": "Optik Merah Putih, Optik Sukarno, Optik tik, Optik Susi",
        "name": "Kacamata 17",
        "similarity": 0.02413793103448275,
        "tags": "oblong, square, triangle"
    },
    {
        "brand": "CHANEL",
        "category": "Glasses",
        "clicked": 50,
        "combined": "Kacamata 3 This is the description of Product 3 Glasses CHANEL heart, square, oval Optik tik, Optik Merah Putih, Optik Sukarno, Optik Susi",
        "description": "This is the description of Product 3",
        "id": 3,
        "merchants": "Optik tik, Optik Merah Putih, Optik Sukarno, Optik Susi",
        "name": "Kacamata 3",
        "similarity": 0.021739130434782594,
        "tags": "heart, square, oval"
    },
    {
        "brand": "Police",
        "category": "Glasses",
        "clicked": 66,
        "combined": "Kacamata 5 This is the description of Product 5 Glasses Police oval, oblong, heart Optik Merah Putih, Optik Susi, Optik tik, Optik Sukarno",
        "description": "This is the description of Product 5",
        "id": 5,
        "merchants": "Optik Merah Putih, Optik Susi, Optik tik, Optik Sukarno",
        "name": "Kacamata 5",
        "similarity": 0.021739130434782594,
        "tags": "oval, oblong, heart"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 46,
        "combined": "Kacamata 6 This is the description of Product 6 Glasses Oakley oblong, heart, oval Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 6",
        "id": 6,
        "merchants": "Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "name": "Kacamata 6",
        "similarity": 0.021739130434782594,
        "tags": "oblong, heart, oval"
    },
    {
        "brand": "CHANEL",
        "category": "Glasses",
        "clicked": 41,
        "combined": "Kacamata 9 This is the description of Product 9 Glasses CHANEL heart, oblong, oval Optik Sukarno, Optik Merah Putih, Optik Susi, Optik tik",
        "description": "This is the description of Product 9",
        "id": 9,
        "merchants": "Optik Sukarno, Optik Merah Putih, Optik Susi, Optik tik",
        "name": "Kacamata 9",
        "similarity": 0.021739130434782594,
        "tags": "heart, oblong, oval"
    }
]
```

### Product Recommendation

```
URL: 127.0.0.1:5000/product_recommendation
Method: POST
Body: None (no params)
```

In HTTPie, we can write this as: 

```
http GET "127.0.0.1:5000/product_recommendation" query="Emporio" "X-API-KEY: XXX"
```

It will return:

```
[
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 39,
        "combined": "Kacamata 8 This is the description of Product 8 Glasses Emporio heart, oval, square Optik Merah Putih, Optik Susi, Optik Sukarno, Optik tik",
        "description": "This is the description of Product 8",
        "id": 8,
        "merchants": "Optik Merah Putih, Optik Susi, Optik Sukarno, Optik tik",
        "name": "Kacamata 8",
        "similarity": 0.0475012846865365,
        "tags": "heart, oval, square"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 62,
        "combined": "Kacamata 15 This is the description of Product 15 Glasses Emporio oval, circle, oblong Optik Merah Putih, Optik tik, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 15",
        "id": 15,
        "merchants": "Optik Merah Putih, Optik tik, Optik Susi, Optik Sukarno",
        "name": "Kacamata 15",
        "similarity": 0.04657771188534719,
        "tags": "oval, circle, oblong"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 67,
        "combined": "Kacamata 18 This is the description of Product 18 Glasses Emporio heart, square, oblong Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 18",
        "id": 18,
        "merchants": "Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "name": "Kacamata 18",
        "similarity": 0.04621465491030709,
        "tags": "heart, square, oblong"
    },
    {
        "brand": "Emporio",
        "category": "Glasses",
        "clicked": 18,
        "combined": "Kacamata 1 This is the description of Product 1 Glasses Emporio square, circle, triangle Optik Sukarno, Optik Susi, Optik tik, Optik Merah Putih",
        "description": "This is the description of Product 1",
        "id": 1,
        "merchants": "Optik Sukarno, Optik Susi, Optik tik, Optik Merah Putih",
        "name": "Kacamata 1",
        "similarity": 0.04567307692307695,
        "tags": "square, circle, triangle"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 5,
        "combined": "Kacamata 11 This is the description of Product 11 Glasses Oakley heart, triangle, square Optik Merah Putih, Optik Sukarno, Optik Susi, Optik tik",
        "description": "This is the description of Product 11",
        "id": 11,
        "merchants": "Optik Merah Putih, Optik Sukarno, Optik Susi, Optik tik",
        "name": "Kacamata 11",
        "similarity": 0.02430555555555558,
        "tags": "heart, triangle, square"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 6,
        "combined": "Kacamata 17 This is the description of Product 17 Glasses Oakley oblong, square, triangle Optik Merah Putih, Optik Sukarno, Optik tik, Optik Susi",
        "description": "This is the description of Product 17",
        "id": 17,
        "merchants": "Optik Merah Putih, Optik Sukarno, Optik tik, Optik Susi",
        "name": "Kacamata 17",
        "similarity": 0.02413793103448275,
        "tags": "oblong, square, triangle"
    },
    {
        "brand": "CHANEL",
        "category": "Glasses",
        "clicked": 50,
        "combined": "Kacamata 3 This is the description of Product 3 Glasses CHANEL heart, square, oval Optik tik, Optik Merah Putih, Optik Sukarno, Optik Susi",
        "description": "This is the description of Product 3",
        "id": 3,
        "merchants": "Optik tik, Optik Merah Putih, Optik Sukarno, Optik Susi",
        "name": "Kacamata 3",
        "similarity": 0.021739130434782594,
        "tags": "heart, square, oval"
    },
    {
        "brand": "Police",
        "category": "Glasses",
        "clicked": 66,
        "combined": "Kacamata 5 This is the description of Product 5 Glasses Police oval, oblong, heart Optik Merah Putih, Optik Susi, Optik tik, Optik Sukarno",
        "description": "This is the description of Product 5",
        "id": 5,
        "merchants": "Optik Merah Putih, Optik Susi, Optik tik, Optik Sukarno",
        "name": "Kacamata 5",
        "similarity": 0.021739130434782594,
        "tags": "oval, oblong, heart"
    },
    {
        "brand": "Oakley",
        "category": "Glasses",
        "clicked": 46,
        "combined": "Kacamata 6 This is the description of Product 6 Glasses Oakley oblong, heart, oval Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "description": "This is the description of Product 6",
        "id": 6,
        "merchants": "Optik tik, Optik Merah Putih, Optik Susi, Optik Sukarno",
        "name": "Kacamata 6",
        "similarity": 0.021739130434782594,
        "tags": "oblong, heart, oval"
    },
    {
        "brand": "CHANEL",
        "category": "Glasses",
        "clicked": 41,
        "combined": "Kacamata 9 This is the description of Product 9 Glasses CHANEL heart, oblong, oval Optik Sukarno, Optik Merah Putih, Optik Susi, Optik tik",
        "description": "This is the description of Product 9",
        "id": 9,
        "merchants": "Optik Sukarno, Optik Merah Putih, Optik Susi, Optik tik",
        "name": "Kacamata 9",
        "similarity": 0.021739130434782594,
        "tags": "heart, oblong, oval"
    }
] 
```
