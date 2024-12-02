import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# VGG19 modelinin katman adları
CONTENT_LAYERS = ['block5_conv2']  # İçerik katmanı
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Stil katmanları

# İçerik ve stil kayıplarına verilen ağırlıklar
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2

def load_and_process_img(path_to_img):
    """
    Görüntüyü yükler, yeniden boyutlandırır ve VGG19 modeline uygun şekilde işler.
    """
    max_dim = 512  # Maksimum boyut
    img = load_img(path_to_img)  # Görüntüyü yükle
    long = max(img.size)  # Görüntünün en uzun kenarı
    scale = max_dim / long  # Yeniden boyutlandırma oranı
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))  # Yeniden boyutlandır
    img = img_to_array(img)  # NumPy dizisine çevir
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle
    img = tf.keras.applications.vgg19.preprocess_input(img)  # VGG19 giriş formatına uygun hale getir
    return img

def deprocess_img(processed_img):
    """
    İşlenmiş görüntüyü orijinal görüntü formatına dönüştür.
    """
    x = processed_img.copy()
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939  # VGG19'un çıkardığı ortalamaları geri ekle
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # RGB'ye geri döndür
    x = np.clip(x, 0, 255).astype('uint8')  # Değerleri 0-255 arasında tut
    return x

def get_model():
    """
    VGG19 modelini yükler ve içerik ve stil katmanlarını döndürür.
    """
    vgg = VGG19(include_top=False, weights='imagenet')  # VGG19 modelini yükle
    vgg.trainable = False  # Modeli eğitilemez hale getir
    content_outputs = [vgg.get_layer(layer).output for layer in CONTENT_LAYERS]
    style_outputs = [vgg.get_layer(layer).output for layer in STYLE_LAYERS]
    model_outputs = content_outputs + style_outputs
    return Model(vgg.input, model_outputs)

def get_content_and_style_representations(model, content_path, style_path):
    """
    İçerik ve stil görüntülerini alır ve onların temsil edici özelliklerini döndürür.
    """
    content_img = load_and_process_img(content_path)  # İçerik görüntüsünü yükle
    style_img = load_and_process_img(style_path)  # Stil görüntüsünü yükle
    content_outputs = model(content_img)
    style_outputs = model(style_img)
    content_features = [content_outputs[i] for i in range(len(CONTENT_LAYERS))]
    style_features = [style_outputs[i + len(CONTENT_LAYERS)] for i in range(len(STYLE_LAYERS))]
    return content_features, style_features

def compute_content_loss(content, target):
    """
    İçerik kaybını hesaplar.
    """
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    """
    Gram matrisi hesaplar (stil özelliklerinin korelasyonunu gösterir).
    """
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def compute_style_loss(style, target):
    """
    Stil kaybını hesaplar.
    """
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(target)))

def compute_losses(model, content_img, style_img):
    """
    Toplam içerik ve stil kaybını hesaplar.
    """
    content_features, style_features = get_content_and_style_representations(model, content_img, style_img)
    generated_outputs = model(content_img)
    generated_content_features = [generated_outputs[i] for i in range(len(CONTENT_LAYERS))]
    generated_style_features = [generated_outputs[i + len(CONTENT_LAYERS)] for i in range(len(STYLE_LAYERS))]
    
    content_loss = tf.add_n([compute_content_loss(generated_content_features[i], content_features[i]) for i in range(len(CONTENT_LAYERS))])
    style_loss = tf.add_n([compute_style_loss(generated_style_features[i], style_features[i]) for i in range(len(STYLE_LAYERS))])
    
    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
    return total_loss

@tf.function
def train_step(image, model, optimizer):
    """
    Eğitim adımını uygular.
    """
    with tf.GradientTape() as tape:
        loss = compute_losses(model, image, style_path)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

content_path = "content.jpg"  # İçerik görüntüsünün yolu
style_path = "style.jpg"  # Stil görüntüsünün yolu

# İçerik ve stil temsilcilerini oluşturmak için modeli ve optimizasyonu başlat
generated_image = tf.Variable(load_and_process_img(content_path), dtype=tf.float32)
model = get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Eğitim döngüsü
epochs = 10  # Eğitim epoch sayısı
steps_per_epoch = 100

for i in range(epochs):
    for _ in range(steps_per_epoch):
        train_step(generated_image, model, optimizer)
    if i % 10 == 0:
        plt.imshow(deprocess_img(generated_image.numpy()))
        plt.show()

