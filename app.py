import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request
from flask_cors import CORS

from DivisionIntoLexemes import DivisionIntoLexemes
from photoRecognizer import PhotoRecognizer

model = tf.keras.models.load_model('My.keras')

app = Flask(__name__)

CORS(app)

EngBrailleDict = {
    "a": "100000",
    "b": "101000",
    "c": "110000",
    "d": "110100",
    "e": "100100",
    "f": "111000",
    "g": "111100",
    "h": "101100",
    "i": "011000",
    "j": "011100",
    "k": "100010",
    "l": "101010",
    "m": "110010",
    "n": "110110",
    "o": "100110",
    "p": "111010",
    "q": "111110",
    "r": "101110",
    "s": "011010",
    "t": "011110",
    "u": "100011",
    "v": "101011",
    "w": "011101",
    "x": "110011",
    "y": "110111",
    "z": "100111",
    " ": "000000",
    ",": "001000",
    ".": "001101",
    "?": "001011",
    "!": "001110",
    ";": "001010",
    ":": "001100",
    "`": "000010",
    "-": "000011",
    '"': "000111",
    "(": "001111",
    ")": "001111",
    "1": "100000",
    "2": "101000",
    "3": "110000",
    "4": "110100",
    "5": "100100",
    "6": "111000",
    "7": "111100",
    "8": "101100",
    "9": "011000",
    "0": "011100",
    "num": "010111",
    "cap": "000001"
}

EngAlph = {
    "0": "a",
    "1": "`",
    "2": "b",
    "3": "c",
    "4": "capt",
    "5": ":",
    "6": ",",
    "7": "d",
    "8": "e",
    "9": "!",
    "10": "f",
    "11": "g",
    "12": "h",
    "13": "-",
    "14": "i",
    "15": "j",
    "16": "k",
    "17": "l",
    "18": "m",
    "19": "n",
    "20": "number",
    "21": "o",
    "22": "p",
    "23": ".",
    "24": "q",
    "25": "?",
    "26": "r",
    "27": "s",
    "28": ";",
    "29": " ",
    "30": "t",
    "31": "u",
    "32": "v",
    "33": "w",
    "34": "x",
    "35": "y",
    "36": "z",
}


def check_is_number(s):
    valid_strings = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    return s in valid_strings


@app.route('/predictText', methods=['POST'])
def predictText():
    if 'file' not in request.files:
        return 'Файл не знайдено', 400
    file = request.files['file']
    if file.filename == '':
        return 'Не вибрано файл', 400

    file_stream = file.stream
    image = Image.open(file_stream)

    np_array_image = np.array(image.convert('L'), np.uint8)
    arr = PhotoRecognizer(np_array_image).matrixImage
    lexems = DivisionIntoLexemes(arr).lexemsList
    if len(lexems) == 0:
        return {"data": ""}
    for i in range(len(lexems)):
        if lexems[i].dtype != np.uint8:
            lexems[i] = lexems[i].astype(np.uint8)

    output = ''
    print(output)
    for img in lexems:
            img = Image.fromarray(img).resize((48, 48))
            x = np.expand_dims(img, axis=0)
            predict = np.argmax(model.predict(x), axis=1)[0]
            output += EngAlph[str(predict)]
    return {"data": output}


@app.route('/translateText', methods=['POST'])
def translateText():
    isDigit = False
    translateText = ''
    text = request.json.get('data')
    lexemes = list(text)
    for i in lexemes:
        try:
            l = EngBrailleDict[i.lower()]
        except KeyError:
            return {"data": ""}
        if not check_is_number(i):
            isDigit = False
        if check_is_number(i) and not isDigit:
            translateText += EngBrailleDict["num"]
            isDigit = True
        if i.isupper():
            translateText += EngBrailleDict["cap"]
        translateText += EngBrailleDict[i.lower()]
    return {"data": translateText}


@app.route('/')
def hello():
    return 'started'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
