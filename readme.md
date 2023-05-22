# Arvigo Machine Learning API

## Table of Models

- Model for detecting human and non-human faces
- Model for detecting face shapes
- Model for analyzing personalities

## Development

### Dependencies

```
Flask==2.3.2
joblib==1.2.0
matplotlib==3.7.1
numpy==1.23.0
pandas==2.0.1
protobuf==4.23.0
scikit_learn==1.2.2
yellowbrick==1.5
tensorflow-estimator==2.12.0
tensorflow-macos==2.12.0
tensorflow-metal==0.8.0
keras==2.12.0
Werkzeug==2.3.3
```

If you're on macOS, you can use `tensorflow_macos==2.12.0`

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

In HTTPie, we can write this as: `http POST "127.0.0.1:5000/is_human" image="base64stringhere"`

#### Face Shape Detection

```
URL: 127.0.0.1:5000/face_shape
Method: POST
Body: image="base64stringhere"
```

In HTTPie, we can write this as: `http POST "127.0.0.1:5000/face_shape" image="base64stringhere"`

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
    OPN6:=4 OPN7:=4 OPN8:=4 OPN9:=4 OPN10:=4
```

It _could_ return:

```
{
    "predicted_personality": [
        "Extraversion"
    ]
}
```

#### **Dummy** Personality Analysis

```
URL: 127.0.0.1:5000/dummy_detect_personality
Method: POST
Body: JSON
```

In HTTPie, we can write this as: 

```
http POST http://127.0.0.1:5000/dummy_detect_personality \
    Content-Type:application/json
```

It will return:

```
{
    "input": {
        "AGR1": 1,
        "AGR10": 1,
        "AGR2": 1,
        "AGR3": 4,
        "AGR4": 3,
        "AGR5": 4,
        "AGR6": 4,
        "AGR7": 3,
        "AGR8":

 4,
        "AGR9": 1,
        "CSN1": 3,
        "CSN10": 4,
        "CSN2": 2,
        "CSN3": 2,
        "CSN4": 5,
        "CSN5": 2,
        "CSN6": 1,
        "CSN7": 3,
        "CSN8": 3,
        "CSN9": 5,
        "EST1": 4,
        "EST10": 5,
        "EST2": 2,
        "EST3": 4,
        "EST4": 5,
        "EST5": 4,
        "EST6": 3,
        "EST7": 4,
        "EST8": 5,
        "EST9": 1,
        "EXT1": 4,
        "EXT10": 4,
        "EXT2": 3,
        "EXT3": 1,
        "EXT4": 5,
        "EXT5": 3,
        "EXT6": 5,
        "EXT7": 1,
        "EXT8": 2,
        "EXT9": 4,
        "OPN1": 2,
        "OPN10": 5,
        "OPN2": 5,
        "OPN3": 5,
        "OPN4": 4,
        "OPN5": 4,
        "OPN6": 5,
        "OPN7": 1,
        "OPN8": 4,
        "OPN9": 4
    },
    "predicted_personality": [
        "Agreeableness"
    ]
}
```