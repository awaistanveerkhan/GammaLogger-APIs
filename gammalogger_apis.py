import numpy as np
from flask import Flask, request, jsonify
import os,joblib
from textblob import TextBlob
import tempfile
from werkzeug.utils import secure_filename


from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

text_base_dir = "text_inference"
image_base_dir = "image_inference"

with open(os.path.join(text_base_dir,'text_inference_scaler.joblib'), 'rb') as scaler_file:
    text_inference_scaler = joblib.load(scaler_file)

with open(os.path.join(text_base_dir,'text_inference_dictionary.joblib'), 'rb') as dict_file:
    dict_alpha = joblib.load(dict_file)

with open(os.path.join(text_base_dir, 'text_inference_machine_learning_model.pkl'), 'rb') as model_file:
    text_inference_model = joblib.load(model_file)


with open(os.path.join(image_base_dir, 'tokenizer.joblib'), 'rb') as tokenizer_file:
    image_inference_tokenizer = joblib.load(tokenizer_file)

vgg_model = load_model(os.path.join(image_base_dir, 'vgg16.h5'))

image_inference_model = load_model(os.path.join(image_base_dir, 'image_inference_model.h5'))

@app.route('/predict_character', methods=['POST'])
def predict_character():
    try:
        predictions = []
        words = []
        current_word = ''

        data = request.get_json(force=True)
        input_data = data['text_input_data']
    
        for input_row in input_data:
            input_row = np.array(input_row)
            scaled_input = text_inference_scaler.transform([input_row])
            prediction = text_inference_model.predict(scaled_input)[0]
            result = dict_alpha[prediction]        
            predictions.append(result)

                
        for char in predictions:
            if char != 'space':
                current_word += char
            elif char == 'space':
                if current_word:
                    words.append(current_word)
                    current_word = ''

        
        if current_word:
            words.append(current_word)

        
        sentence = ' '.join(words)

        corrected_sentence = str(TextBlob(sentence).correct())

        return jsonify({'result' : corrected_sentence})

    except Exception as e:
        return jsonify({'error': str(e)})

def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, image, tokenizer, max_length=31):
  in_text = 'startseq'
  for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], max_length)
    yhat = model.predict([image,sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = index_to_word(yhat, tokenizer)

    if word is None:
      break

    in_text += " " + word

    if word == 'endseq':
        break

    

  return in_text


@app.route('/predict_caption', methods=['POST'])
def predict_caption():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'File is not an image'})

        temp_dir = tempfile.mkdtemp()
        uploaded_image_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(uploaded_image_path)

        image = load_img(uploaded_image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        feature = vgg_model.predict(image, verbose=0)
        caption = generate_caption(image_inference_model, feature, image_inference_tokenizer, max_length=31)

        os.remove(uploaded_image_path)
        os.rmdir(temp_dir)

        return jsonify({'text': caption})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)