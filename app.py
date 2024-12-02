import tensorflow as tf
from flask import Flask, request, render_template, url_for
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from datetime import datetime

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
model = tf.saved_model.load("model")

# Görsel yükleme fonksiyonu
def load_image(img_path):
    img = tf.io.read_file(img_path)  
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Stil transferi fonksiyonu
def stylize_images(content_image, style_image):
    return model(tf.constant(content_image), tf.constant(style_image))[0]

# Yüklemeler için yeni klasör ismi oluşturma
def create_new_folder():
    # Mevcut yüklemeleri kontrol et
    upload_dir = "static/uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Klasör ismi oluştur (1, 2, 3 şeklinde)
    existing_folders = sorted([int(folder) for folder in os.listdir(upload_dir) if folder.isdigit()])
    new_folder_name = str(existing_folders[-1] + 1) if existing_folders else '1'
    
    new_folder_path = os.path.join(upload_dir, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

# Ana sayfa route'u
@app.route('/')
def index():
    return render_template('index.html')

# Görsel yükleme ve stil transferi işlemi
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'content' not in request.files or 'style' not in request.files:
        return "No file part", 400
    
    content_image = request.files['content']
    style_image = request.files['style']
    
    if content_image.filename == '' or style_image.filename == '':
        return "No selected file", 400
    
    # Dosya adı güvenli hale getirilip kaydedilecek
    content_filename = secure_filename(content_image.filename)
    style_filename = secure_filename(style_image.filename)
    
    # Yeni klasör oluştur
    folder_path = create_new_folder()
    
    # Dosyaları yeni klasöre kaydet
    content_path = os.path.join(folder_path, content_filename)
    style_path = os.path.join(folder_path, style_filename)
    
    content_image.save(content_path)
    style_image.save(style_path)
    
    # Yüklenen görselleri yükle
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    
    # Stil transferini yap
    stylized_image = stylize_images(content_image, style_image)
    
    # Çıktıyı kaydet
    output_path = os.path.join(folder_path, "generated_image.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_BGR2RGB))
    
    # Üç dosyayı (content, style, generated image) kullanıcıya göster
    return render_template('result.html', folder_name=os.path.basename(folder_path),
                           content_image=content_filename, 
                           style_image=style_filename, 
                           generated_image="generated_image.jpg")

# Uygulamanın çalıştırılması
if __name__ == "__main__":
    # Gerekli klasörlerin olup olmadığını kontrol et
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
