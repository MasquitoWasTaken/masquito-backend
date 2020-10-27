# masquito-backend

Fullstack application to tell if you're wearing a mask right

## run this program

1. Clone this repository
2. Install the pip dependencies listed below
3. Download the latest model from the [releases](https://github.com/MasquitoWasTaken/masquito-ml-training/releases) tab
4. Place the `.h5` file in `training_data/models/`
5. Run the code with `python ./app.py`
6. Send a request to the server using the guide below

## endpoints

```
POST /mask
    Parameters:
        image={data uri}
    Example:
        POST http://127.0.0.1/mask?image=data:image/jpeg;base64,...
```

## directory structure

```
training_data/
    json/
        model_class.json
    models/
```

## pip dependencies

Note: you **must** use Python <3.8 (I recommend 3.7)

```
tensorflow<2
scipy<1.5
numpy
keras
opencv-python
pillow
imageai
flask==0.12.2
```
